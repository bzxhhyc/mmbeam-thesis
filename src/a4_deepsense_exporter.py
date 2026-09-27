"""阶段 A4: DeepSense 6G 格式导出器

把 (位置, 64维波束功率向量, 最优波束编号) 按 DeepSense position-aided
beam prediction 任务的 schema 落盘：
  data/deepsense_format/scenario_sionna_1/
    unit1/                     # 基站侧(功率向量按 DeepSense 惯例归在 unit1)
      pwr/power_<idx>.txt      # 每行一个样本的 64 维功率(空格分隔)
    unit2/
      gps_data/gps_<idx>.txt   # UE 位置 "x y"
    dev_data.csv               # index | unit2_loc | unit1_pwr | beam_index

功率向量存"干净值"(不加噪声)，SNR 在实验阶段注入(与 DeepSense 挑战赛做法一致)。
用法: py -3.10 src/a4_deepsense_exporter.py  （跑 100 样本合成自检）
"""
import os
import numpy as np

# DeepSense schema 的列名(对齐官网 dev_data.csv)
COLUMNS = ["index", "unit2_loc", "unit1_pwr_60GHz", "beam_index"]

class DeepSenseExporter:
    def __init__(self, root, scenario="scenario_sionna_1"):
        self.dir = os.path.join(root, scenario)
        self.pwr_dir = os.path.join(self.dir, "unit1", "pwr")
        self.gps_dir = os.path.join(self.dir, "unit2", "gps_data")
        for d in (self.pwr_dir, self.gps_dir):
            os.makedirs(d, exist_ok=True)
        self.rows = []

    def add(self, idx, loc_xy, power_vec64):
        """登记一条样本。power_vec64: (64,) 实数(线性功率或 dB 均可, 全数据集统一)。"""
        assert np.shape(power_vec64) == (64,), f"功率向量必须是 64 维, 拿到 {np.shape(power_vec64)}"
        beam_index = int(np.argmax(power_vec64)) + 1   # DeepSense 波束编号从 1 起
        pwr_path = os.path.join("unit1", "pwr", f"power_{idx}.txt")
        gps_path = os.path.join("unit2", "gps_data", f"gps_{idx}.txt")
        # 落盘功率向量(单行 64 列, 空格分隔)
        np.savetxt(os.path.join(self.dir, pwr_path), np.atleast_2d(power_vec64), fmt="%.6e")
        # 落盘位置 "x y"
        with open(os.path.join(self.dir, gps_path), "w") as f:
            f.write(f"{loc_xy[0]:.4f} {loc_xy[1]:.4f}")
        self.rows.append((idx, gps_path, pwr_path, beam_index))
        return beam_index

    def write_csv(self):
        csv_path = os.path.join(self.dir, "dev_data.csv")
        with open(csv_path, "w") as f:
            f.write(",".join(COLUMNS) + "\n")
            for idx, gps_path, pwr_path, beam_index in self.rows:
                f.write(f"{idx},{gps_path},{pwr_path},{beam_index}\n")
        return csv_path

# ---------------- 合成自检 ----------------
def self_test():
    """手造 100 条 LoS 数据(复用 A3 的码本),验证导出器格式正确且可回读。"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from a3_dft_codebook import dft_codebook, beam_powers

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "deepsense_format")
    exp = DeepSenseExporter(root)
    cb = dft_codebook()
    rng = np.random.default_rng(42)

    for i in range(1, 101):  # 100 条
        x = rng.uniform(0.5, 49.5); y = rng.uniform(0.5, 49.5)
        theta = np.deg2rad(rng.uniform(30, 80)); phi = np.deg2rad(rng.uniform(0, 360))
        powers = beam_powers([(1.0 + 0j, theta, phi)], cb)   # 干净 LoS 功率
        exp.add(i, (x, y), powers)

    csv_path = exp.write_csv()
    # 回读校验
    lines = open(csv_path).read().strip().split("\n")
    n_data = len(lines) - 1
    first = lines[1].split(",")
    pwr_back = np.loadtxt(os.path.join(root, "scenario_sionna_1", first[2]))
    gps_back = open(os.path.join(root, "scenario_sionna_1", first[1])).read().split()
    print(f"[export] CSV={csv_path}")
    print(f"[export] 数据行数={n_data} (期望 100)")
    print(f"[export] 表头={lines[0]}")
    print(f"[export] 首行样本: idx={first[0]}, beam_index={first[3]}")
    print(f"[export] 回读功率向量 shape={pwr_back.shape} (期望 (64,))")
    print(f"[export] 回读位置=({gps_back[0]}, {gps_back[1]})")
    ok = (n_data == 100 and pwr_back.shape == (64,) and len(first) == 4)
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    self_test()
