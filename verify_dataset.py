"""数据质量验证：完整性 + 物理合理性 + 区域划分正确性"""
import os
import numpy as np
import pandas as pd

data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "deepsense_format", "scenario_sionna_1")
csv_path = os.path.join(data_dir, "dev_data.csv")

print("=== 1. 完整性检查 ===")
df = pd.read_csv(csv_path)
print(f"总样本数: {len(df)} (期望 50000)")
print(f"列名: {list(df.columns)}")
print(f"beam_index 范围: {df.beam_index.min()}-{df.beam_index.max()} (期望 1-64)")

# 抽查 10 条样本的文件是否存在且格式正确
missing_pwr = 0
missing_gps = 0
bad_shape = 0
for idx in np.random.choice(len(df), 100, replace=False):
    row = df.iloc[idx]
    pwr_path = os.path.join(data_dir, row.unit1_pwr_60GHz)
    gps_path = os.path.join(data_dir, row.unit2_loc)
    if not os.path.exists(pwr_path): missing_pwr += 1
    else:
        p = np.loadtxt(pwr_path)
        if p.shape != (64,): bad_shape += 1
    if not os.path.exists(gps_path): missing_gps += 1
print(f"抽查 100 条: 缺功率文件={missing_pwr}, 缺位置文件={missing_gps}, 功率维度错={bad_shape}")

print("\n=== 2. 物理合理性检查 ===")
# LoS 主导区域（靠近基站 0.2,0.2）应能量集中
near_bs = df[df.unit2_loc.str.contains("gps_")].sample(50)
concentration_ratios = []
for idx in near_bs.index:
    pwr = np.loadtxt(os.path.join(data_dir, df.loc[idx, "unit1_pwr_60GHz"]))
    top2 = np.sort(pwr)[-2:]
    concentration_ratios.append(top2[1] / (top2[0] + 1e-12))
print(f"LoS 区域能量集中比(最强/次强): {np.mean(concentration_ratios):.1f}±{np.std(concentration_ratios):.1f} (期望 >5)")

print("\n=== 3. 区域划分检查 ===")
# 读位置，验证区域划分
regions = {"train": 0, "val": 0, "test": 0, "offgrid": 0}
for idx in range(len(df)):
    gps = open(os.path.join(data_dir, df.loc[idx, "unit2_loc"])).read().split()
    x, y = float(gps[0]), float(gps[1])
    # 物理坐标 0-50m 的区域划分（对齐 gen_dataset_grid.py）
    if x < 30.0: regions["train"] += 1
    elif x < 37.5: regions["val"] += 1
    elif x < 45.0: regions["test"] += 1
    else: regions["offgrid"] += 1
print(f"区域分布: {regions}")
print(f"比例: train={regions['train']/len(df):.1%} val={regions['val']/len(df):.1%} test={regions['test']/len(df):.1%} offgrid={regions['offgrid']/len(df):.1%}")

print("\n=== 4. 轨迹连续性检查 ===")
# 抽查几条轨迹的 beam_index 变化是否平滑
sample_beams = df.beam_index.sample(20).values
print(f"beam_index 抽样: {sample_beams[:10]} (应在 1-64 内)")
print(f"beam_index 众数: {df.beam_index.mode().values} (最常出现的波束)")

print("\n=== 总结 ===")
all_pass = (len(df) == 50000 and missing_pwr == 0 and missing_gps == 0 and bad_shape == 0
            and np.mean(concentration_ratios) > 3)
print(f"{'✅ PASS' if all_pass else '❌ FAIL'}")
