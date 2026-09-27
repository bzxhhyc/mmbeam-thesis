"""阶段 A2: Sionna 0.18 室内场景冒烟（修正版，对齐 0.18 真实 API）
用法:
  $env:DRJIT_LIBLLVM_PATH="C:\\Program Files\\LLVM\\bin\\LLVM-C.dll"
  env/venv-tf/Scripts/python.exe src/a2_sionna_smoke.py
"""
import os, sys, time
import numpy as np
import tensorflow as tf
import sionna

# 先试 CUDA 变体(失败则回退 LLVM CPU)。必须在 import sionna.rt 前设定。
import mitsuba as mi
VARIANT = "cuda_ad_rgb" if "--cpu" not in sys.argv else "llvm_ad_rgb"
mi.set_variant(VARIANT)
print(f"[variant] {mi.variant()}")
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, solver_paths

N_POSITIONS = 50   # 冒烟规模(首次先小), 计时外推

def main():
    print("TF", tf.__version__, "| Sionna", sionna.__version__)
    print("GPUs:", tf.config.list_physical_devices('GPU'))

    # 0.18 没有 simple_room，用内置 box(密闭盒子，做室内)
    from sionna.rt import scene as rt_scene
    scene = load_scene(rt_scene.box)
    scene.frequency = 28e9

    scene.tx_array = PlanarArray(num_rows=8, num_cols=8,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                 vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="iso", polarization="V")

    tx = Transmitter(name="bs", position=[1.0, 1.0, 3.0])
    scene.add(tx)
    rx = Receiver(name="ue", position=[5.0, 5.0, 1.5])
    scene.add(rx)

    # 0.18: 路径求解器是 scene 内置的 SolverPaths，scene() 直接返回 Paths
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i in range(N_POSITIONS):
        rx.position = [rng.uniform(0.5, 5.5), rng.uniform(0.5, 5.5), 1.5]
        paths = scene.compute_paths(max_depth=3, los=True,
                                    reflection=True, scattering=False)
        if i == 0:
            a, tau = paths.cir()
            print("[cir] a shape:", a.shape, "| tau shape:", tau.shape)
    dt = time.time() - t0
    print(f"[timing] {N_POSITIONS} pos in {dt:.2f}s -> {dt/N_POSITIONS*1000:.1f} ms/pos")
    print(f"[extrapolate] 10k pos ~ {dt/N_POSITIONS*10000/60:.1f} min")

if __name__ == "__main__":
    main()
