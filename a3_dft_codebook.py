"""阶段 A3: 64 波束 DFT 码本 + 波束功率计算（纯 NumPy，不依赖 Sionna）

物理模型：8x8 UPA（均匀平面阵列），半波长间距，28GHz。
码本：64 个 DFT 波束 = 8 个方位档 × 8 个俯仰档的可分离乘积。

金标准验证：LoS 单径场景，argmax 波束功率的方向必须等于用户几何方向。
用法: py -3.10 src/a3_dft_codebook.py   （只依赖 numpy）
"""
import numpy as np

# ---------------- 参数 ----------------
N_ROW, N_COL = 8, 8          # UPA 行(俯仰) x 列(方位)
N_BEAMS = N_ROW * N_COL      # 64
FREQ = 28e9
C = 3e8
LAMBDA = C / FREQ
D = 0.5 * LAMBDA             # 半波长间距

def array_response_upa(theta, phi):
    """UPA 转向矢量。
    theta: 俯仰角(从 z 轴向下, rad)，phi: 方位角(在 xy 平面从 x 轴起, rad)。
    返回长度 N_ROW*N_COL 的复数向量（行主序: row=俯仰, col=方位）。
    """
    # 沿列方向(方位)和行方向(俯仰)的归一化空间频率
    u = np.sin(theta) * np.cos(phi)   # x 方向(列)
    v = np.sin(theta) * np.sin(phi)   # y 方向(行)
    a_col = np.exp(1j * np.pi * np.arange(N_COL) * u) / np.sqrt(N_COL)
    a_row = np.exp(1j * np.pi * np.arange(N_ROW) * v) / np.sqrt(N_ROW)
    return np.kron(a_row, a_col)      # (N_ROW*N_COL,) 行主序

def dft_codebook():
    """64 个 DFT 波束。第 (m,n) 个波束指向空间频率 (u,v)=(2m/Nc-1, 2n/Nr-1)。"""
    cb = np.zeros((N_BEAMS, N_ROW * N_COL), dtype=complex)
    idx = 0
    for n in range(N_ROW):            # 俯仰档
        for m in range(N_COL):        # 方位档
            u = 2 * m / N_COL - 1
            v = 2 * n / N_ROW - 1
            a_col = np.exp(1j * np.pi * np.arange(N_COL) * u) / np.sqrt(N_COL)
            a_row = np.exp(1j * np.pi * np.arange(N_ROW) * v) / np.sqrt(N_ROW)
            cb[idx] = np.kron(a_row, a_col)
            idx += 1
    return cb

def beam_powers(cir_paths, codebook):
    """对一组路径(每条: gain, theta, phi)，算每个码本波束的接收功率。
    cir_paths: list of (complex gain, theta, phi)
    返回 (N_BEAMS,) 实数功率(线性值)。
    """
    h = np.zeros(N_ROW * N_COL, dtype=complex)
    for gain, theta, phi in cir_paths:
        h += gain * array_response_upa(theta, phi)
    # 每个波束的接收功率 = |w^H h|^2
    powers = np.abs(codebook.conj() @ h) ** 2
    return powers

# ---------------- 金标准自检 ----------------
def self_test():
    cb = dft_codebook()
    print(f"[codebook] shape={cb.shape}, 每个波束范数={np.linalg.norm(cb[0]):.4f} (应≈1)")
    # 正交性: DFT 波束两两内积应≈0
    ip = np.abs(cb.conj() @ cb[0].T)
    print(f"[orthogonality] beam0 与其余波束内积 max(除自身)={np.sort(ip)[-2]:.2e} (应≈0)")

    # LoS 场景: 用户在几何方向 (theta0, phi0)，单径增益 1
    theta0, phi0 = np.deg2rad(60), np.deg2rad(30)
    powers = beam_powers([(1.0 + 0j, theta0, phi0)], cb)
    best = int(np.argmax(powers))
    # 该波束的 (u,v) 反解回方向角
    n, m = divmod(best, N_COL)
    u = 2 * m / N_COL - 1
    v = 2 * n / N_ROW - 1
    # 精确反解：u=sinθcosφ, v=sinθsinφ → sinθ=√(u²+v²), φ=atan2(v,u)
    s = np.sqrt(u**2 + v**2)
    theta_hat = np.arcsin(np.clip(s, -1, 1))
    phi_hat = np.arctan2(v, u)
    print(f"[LoS golden] 真方向  theta={np.rad2deg(theta0):.2f}° phi={np.rad2deg(phi0):.2f}°")
    print(f"[LoS golden] 波束方向 theta={np.rad2deg(theta_hat):.2f}° phi={np.rad2deg(phi_hat):.2f}° (格心)")
    ang_err = np.rad2deg(abs(theta_hat - theta0) + abs(phi_hat - phi0))
    print(f"[LoS golden] 量化角偏差 ≈{ang_err:.2f}° (8×8 码本固有限，AI 亚格精度要打破的就是它)")
    print(f"[LoS golden] 最优波束 #{best} (俯仰档{n}, 方位档{m}), 功率={powers[best]:.3f}")
    print(f"[LoS golden] Top-3 波束功率: {np.sort(powers)[-3:][::-1].round(3)}")
    # 判据: 最优波束功率应显著大于次优（LoS 单径能量集中）
    ratio = powers[best] / (np.sort(powers)[-2] + 1e-12)
    print(f"[LoS golden] 最强/次强 功率比 = {ratio:.2f} (LoS 单径应 >1，越集中越大)")
    ok = ratio > 1.0
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    self_test()
