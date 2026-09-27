"""诊断 beam_index 分布：可视化空间分布，找根因"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "deepsense_format", "scenario_sionna_1")
df = pd.read_csv(os.path.join(data_dir, "dev_data.csv"))

# 解析位置
xs, ys, beams = [], [], []
for idx in range(min(5000, len(df))):  # 抽 5000 条加速
    gps = open(os.path.join(data_dir, df.loc[idx, "unit2_loc"])).read().split()
    xs.append(float(gps[0])); ys.append(float(gps[1]))
    beams.append(df.loc[idx, "beam_index"])

xs, ys, beams = np.array(xs), np.array(ys), np.array(beams)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：beam_index 空间分布
sc = axes[0].scatter(xs, ys, c=beams, cmap='tab20', s=1, alpha=0.5)
axes[0].scatter([0.2], [0.2], c='red', marker='^', s=200, label='BS')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_title(f'beam_index spatial distribution (unique={len(np.unique(beams))})')
axes[0].legend()
plt.colorbar(sc, ax=axes[0])

# 右图：beam_index 直方图
axes[1].hist(beams, bins=64, range=(1, 65), edgecolor='black')
axes[1].set_xlabel('beam_index'); axes[1].set_ylabel('count')
axes[1].set_title(f'beam_index histogram (unique={len(np.unique(beams))}/64)')
axes[1].axvline(32.5, color='red', linestyle='--', alpha=0.5)

out = os.path.join(os.path.dirname(__file__), "..", "results", "beam_dist_diagnosis.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved: {out}")
print(f"unique beams: {np.unique(beams)}")
print(f"beam counts: {np.bincount(beams)[1:]}")
