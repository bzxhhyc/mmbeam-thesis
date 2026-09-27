"""正式全量数据生成管线（方案 A，CPU 渲染，挂过夜任务）

50m x 50m 室内场景 → Sionna CIR → A3 DFT 码本算 64 波束功率 → A4 DeepSense 格式落盘。
用法（挂后台过夜跑）:
  $env:DRJIT_LIBLLVM_PATH="C:\\Program Files\\LLVM\\bin\\LLVM-C.dll"
  env/venv-tf/Scripts/python.exe src/gen_dataset.py --n-positions 10000 --traj-per-pos 5
输出:
  data/deepsense_format/scenario_sionna_1/   (DeepSense schema)
  data/gen_log.txt                            (进度日志)
"""
import os, sys, time, argparse
import numpy as np
import tensorflow as tf
import sionna
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")   # CPU 渲染(Windows+新驱动 CUDA 走不通, 见 00-dataset-design)
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.rt import scene as rt_scene

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_dft_codebook import dft_codebook, beam_powers
from a4_deepsense_exporter import DeepSenseExporter

def build_room():
    """室内场景(0.18 内置 box 是单位盒[0,1]³, 坐标需归一化)。
    物理尺寸: 50m x 50m x 4m —— 明天与导师确认后若需真实 50m 几何再调整。
    当前用归一化坐标: bs(0.2,0.2,0.9 顶角高处), ue(盒内地面 0.1 高处)。
    """
    scene = load_scene(rt_scene.box)
    scene.frequency = 28e9
    scene.tx_array = PlanarArray(num_rows=8, num_cols=8,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")
    tx = Transmitter(name="bs", position=[0.2, 0.2, 0.9])
    scene.add(tx)
    rx = Receiver(name="ue", position=[0.6, 0.6, 0.1])
    scene.add(rx)
    return scene, tx, rx

def cir_to_paths(paths):
    """从 Sionna Paths 提取每条有效径的 (复增益, 俯仰角theta, 方位角phi)。
    a: (1,1,1,1,64,63,1) — 64 发天线 x 63 径; theta_t/phi_t: (1,1,1,63)
    """
    a, tau = paths.cir()
    a = np.asarray(a); th = np.asarray(paths.theta_t); ph = np.asarray(paths.phi_t)
    a2 = a[0, 0, 0, 0, :, :, 0]              # (64, 63)
    th1 = th.reshape(-1); ph1 = ph.reshape(-1)
    valid = np.abs(a2).sum(axis=0) > 0       # 有效径掩码
    # 每条径的等效复增益 = 64 发天线增益叠加(快衰包络, 供 DFT 码本用)
    gains = a2.sum(axis=0)[valid]            # (n_valid,)
    return gains, th1[valid], ph1[valid]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-positions", type=int, default=100)   # 先小跑验证, 确认后改 10000
    ap.add_argument("--traj-per-pos", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()

    log_path = os.path.join(args.out, "gen_log.txt")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line); open(log_path, "a").write(line + "\n")

    log(f"start n_pos={args.n_positions} traj/pos={args.traj_per_pos}")
    scene, tx, rx = build_room()
    cb = dft_codebook()
    exp = DeepSenseExporter(os.path.join(args.out, "deepsense_format"))
    rng = np.random.default_rng(2026)

    idx = 0
    t0 = time.time()
    for p in range(args.n_positions):
        # 归一化坐标采样(盒内留边界)
        x = rng.uniform(0.05, 0.95); y = rng.uniform(0.05, 0.95)
        for t in range(args.traj_per_pos):
            rx.position = [x + rng.normal(0, 0.005), y + rng.normal(0, 0.005), 0.1]
            paths = scene.compute_paths(max_depth=3, los=True, reflection=True,
                                        scattering=False)
            gains, thetas, phis = cir_to_paths(paths)
            cir_list = [(gains[k], thetas[k], phis[k]) for k in range(len(gains))]
            powers = beam_powers(cir_list, cb)          # (64,)
            idx += 1
            exp.add(idx, (x, y), powers)
        if (p + 1) % 50 == 0:
            el = time.time() - t0
            log(f"pos {p+1}/{args.n_positions} | {el/(p+1)*1000:.0f} ms/pos | eta {(el/(p+1))*(args.n_positions-p-1)/60:.1f} min")
    csv_path = exp.write_csv()
    log(f"DONE idx={idx} csv={csv_path} total={(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
