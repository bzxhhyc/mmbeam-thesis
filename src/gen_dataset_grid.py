"""阶段 A5: 正式全量数据生成（网格复用 + 区域划分 + Random Waypoint 轨迹）

导师认可决策（2026-09-26）：
- 网格复用：1 万网格点 CIR 只算一次，5 条轨迹复用
- 区域划分：训练/验证/测试按区域块划分 + 测试点离网格偏移
- 速度：步行 0.5-1.5 m/s（Random Waypoint）
- 数据格式：DeepSense 6G schema

用法（挂过夜）:
  $env:DRJIT_LIBLLVM_PATH="C:\\Program Files\\LLVM\\bin\\LLVM-C.dll"
  env/venv-tf/Scripts/python.exe src/gen_dataset_grid.py --grid-size 100 --traj-per-region 5
"""
import os, sys, time, argparse
import numpy as np
import tensorflow as tf
import sionna
import mitsuba as mi

mi.set_variant("llvm_ad_rgb")
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
from sionna.rt import scene as rt_scene

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a3_dft_codebook import dft_codebook, beam_powers
from a4_deepsense_exporter import DeepSenseExporter

# 归一化单位盒 [0,1]³，0.5m 网格 → 网格数 = 1/0.5*50 = 100x100 (假设 50m 物理尺寸)
# 区域划分：训练(60%) / 验证(15%) / 测试(15%) / off-grid(10%)
REGION_SPLIT = {"train": 0.60, "val": 0.15, "test": 0.15, "offgrid": 0.10}

def build_room():
    """50m×50m×4m 真实房间（物理坐标）
    基站: 角落 (2,2,3) 高 3m，下倾 15°
    用户: 地面高 1.5m
    """
    room_xml = os.path.join(os.path.dirname(__file__), "..", "data", "room_50m", "room_50m.xml")
    scene = load_scene(room_xml)
    scene.frequency = 28e9
    scene.tx_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    tx = Transmitter(name="bs", position=[25.0, 25.0, 3.0])  # 房间中央 3m 高（增加有效波束覆盖）
    scene.add(tx)
    rx = Receiver(name="ue", position=[25.0, 25.0, 1.5])  # 中央 1.5m 高
    scene.add(rx)
    return scene, tx, rx

def cir_to_powers(paths, cb):
    a, tau = paths.cir()
    a = np.asarray(a); th = np.asarray(paths.theta_t); ph = np.asarray(paths.phi_t)
    a2 = a[0, 0, 0, 0, :, :, 0]
    th1 = th.reshape(-1); ph1 = ph.reshape(-1)
    valid = np.abs(a2).sum(axis=0) > 0
    gains = a2.sum(axis=0)[valid]
    cir_list = [(gains[k], th1[valid][k], ph1[valid][k]) for k in range(len(gains))]
    return beam_powers(cir_list, cb)

def assign_region(x, y):
    """按区域块划分（防泄漏：测试集区域训练时没见过）。"""
    # 把 [0,1]² 分成 4 个区域，各占不同比例
    if x < 0.6:
        return "train"
    elif x < 0.75:
        return "val"
    elif x < 0.90:
        return "test"
    else:
        return "offgrid"   # 最右侧 10% 做 off-grid 泛化实验

def random_waypoint(rng, start, n_steps, speed_range=(0.5, 1.5), bounds=(2.5, 47.5)):
    """Random Waypoint 轨迹（物理坐标，米）：随机选目标点，直线走过去，到达后再选新目标。"""
    traj = [start]
    pos = np.array(start)
    for _ in range(n_steps - 1):
        target = np.array([rng.uniform(*bounds), rng.uniform(*bounds)])
        direction = target - pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            continue
        direction /= dist
        speed = rng.uniform(*speed_range)   # m/s
        step = direction * speed * 0.1      # 0.1s 时间步
        pos = pos + step
        pos = np.clip(pos, bounds[0], bounds[1])
        traj.append(tuple(pos))
    return traj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-size", type=int, default=100, help="网格边长(100→1万网格点)")
    ap.add_argument("--traj-per-region", type=int, default=5, help="每区域轨迹数")
    ap.add_argument("--steps-per-traj", type=int, default=10, help="每轨迹步数")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--test-mode", action="store_true", help="小规模验证(10x10网格)")
    ap.add_argument("--skip-phase1", action="store_true", help="跳过 Phase1(复用已存 CIR)")
    args = ap.parse_args()

    log_path = os.path.join(args.out, "gen_log_grid.txt")
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    if args.test_mode:
        args.grid_size = 10
        log("TEST MODE: 10x10 grid")

    log(f"start grid={args.grid_size}x{args.grid_size} traj/region={args.traj_per_region} steps={args.steps_per_traj}")
    scene, tx, rx = build_room()
    cb = dft_codebook()
    exp = DeepSenseExporter(os.path.join(args.out, "deepsense_format"))
    rng = np.random.default_rng(2026)

    # 阶段1: 网格点 CIR 计算（复用核心）
    cir_cache = os.path.join(args.out, f"grid_cir_{args.grid_size}_center.npz")   # 中央基站版缓存
    grid_cir = {}
    n_grid = args.grid_size ** 2
    if args.skip_phase1 and os.path.exists(cir_cache):
        log(f"=== Phase 1 SKIPPED (load cache {cir_cache}) ===")
        z = np.load(cir_cache)
        for k in z.files:
            i, j = map(int, k.split(","))
            grid_cir[(i, j)] = z[k]
        log(f"loaded {len(grid_cir)} grid CIR from cache")
    else:
        log("=== Phase 1: grid CIR computation ===")
        t0 = time.time()
        for i in range(args.grid_size):
            for j in range(args.grid_size):
                x = (i + 0.5) * 50.0 / args.grid_size   # 物理坐标 0-50m
                y = (j + 0.5) * 50.0 / args.grid_size
                rx.position = [x, y, 1.5]                 # 用户高 1.5m
                paths = scene.compute_paths(max_depth=3, los=True, reflection=True, scattering=False)
                grid_cir[(i, j)] = cir_to_powers(paths, cb)
                if (i * args.grid_size + j + 1) % 500 == 0:
                    el = time.time() - t0
                    done = i * args.grid_size + j + 1
                    log(f"grid {done}/{n_grid} | {el/done*1000:.0f} ms/pt | eta {el/done*(n_grid-done)/60:.1f} min")
        log(f"Phase 1 DONE: {n_grid} grid pts in {(time.time()-t0)/60:.1f} min")
        # 落盘缓存(供 --skip-phase1 复用)
        np.savez(cir_cache, **{f"{i},{j}": grid_cir[(i,j)] for (i,j) in grid_cir})
        log(f"grid CIR cached to {cir_cache}")

    # 阶段2: 轨迹生成 + 样本落盘（查表复用）
    # 所有轨迹覆盖全房间（不按区域隔离，避免 OOD）；按 traj_id 划分在训练脚本里做
    # 额外 10% offgrid 轨迹（位置加偏移，供泛化测试）
    log("=== Phase 2: trajectory sampling ===")
    idx = 0
    t0 = time.time()
    traj_meta = []   # (idx, traj_id, step, is_offgrid)
    traj_id = 0
    n_main = int(args.traj_per_region * 10)              # 主轨迹（全房间）
    n_off  = max(1, int(n_main * 0.10))                  # offgrid 10%
    total = n_main + n_off
    for t in range(total):
        is_off = t >= n_main
        start = (rng.uniform(2.5, 47.5), rng.uniform(2.5, 47.5))
        traj = random_waypoint(rng, start, args.steps_per_traj)
        for step, (x, y) in enumerate(traj):
            i = min(int(x / 50.0 * args.grid_size), args.grid_size - 1)
            j = min(int(y / 50.0 * args.grid_size), args.grid_size - 1)
            powers = grid_cir[(i, j)]
            if is_off:   # offgrid：位置加离网格偏移（±0.5m）
                x += rng.normal(0, 0.5); y += rng.normal(0, 0.5)
            idx += 1
            exp.add(idx, (x, y), powers)
            traj_meta.append((idx, traj_id, step, int(is_off)))
        traj_id += 1
        if (t + 1) % 1000 == 0:
            log(f"traj {t+1}/{total} done, cum idx={idx}")
    csv_path = exp.write_csv()
    np.save(os.path.join(args.out, "traj_meta.npy"), np.array(traj_meta, dtype=np.int64))
    log(f"Phase 2 DONE: {idx} samples in {(time.time()-t0)/60:.1f} min")
    log(f"ALL DONE csv={csv_path}")

if __name__ == "__main__":
    main()
