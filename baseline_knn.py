"""KNN 基线（诊断 + 开题既定基线之一）
输入 64 维功率，训练集找 k 个最近邻，位置=邻居均值，beam=邻居投票。
KNN 准 → 数据可学，MLP 结构问题；KNN 也差 → 数据本身问题。
"""
import os, sys, argparse
import numpy as np
import pandas as pd

def read_gps(data_dir, rel):
    with open(os.path.join(data_dir, rel)) as f:
        x, y = f.read().split()
    return [float(x), float(y)]

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--snr", type=float, default=10)
ap.add_argument("--n-val", type=int, default=2000)
args = ap.parse_args()

data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "deepsense_format", "scenario_sionna_1")
df = pd.read_csv(os.path.join(data_dir, "dev_data.csv"))
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
n = len(df)
train_df, val_df = df[:int(0.7*n)], df[int(0.7*n):int(0.85*n)]

print("loading powers...")
Xtr = np.stack([np.loadtxt(os.path.join(data_dir, p)) for p in train_df.unit1_pwr_60GHz])
Xv = np.stack([np.loadtxt(os.path.join(data_dir, p)) for p in val_df.unit1_pwr_60GHz])
Ytr = np.stack([read_gps(data_dir, p) for p in train_df.unit2_loc])
Yv = np.stack([read_gps(data_dir, p) for p in val_df.unit2_loc])
Btr = train_df.beam_index.values
Bv = val_df.beam_index.values

# 归一化（与 MLP 一致）+ 同 SNR 噪声
def prep(X, snr_db):
    X = X.copy()
    sig = X.max(axis=1, keepdims=True)
    noise_std = np.sqrt(sig / (10 ** (snr_db / 10)))
    X = X + np.random.normal(0, noise_std, X.shape)
    return X / (X.max(axis=1, keepdims=True) + 1e-12)

Xtr_n = prep(Xtr, args.snr)
Xv_n = prep(Xv, args.snr)

# 用对数功率（dB 尺度差异更平滑，KNN 距离更合理）
Xtr_log = np.log10(Xtr_n + 1e-12)
Xv_log = np.log10(Xv_n + 1e-12)

print(f"train={len(Xtr)} val={len(Xv)} k={args.k} snr={args.snr}dB")

# 矩阵分解算距离 ||a-b||²=||a||²+||b||²-2a·b，避免 (b,ntr,64) 内存爆炸
def knn_idx(Xtr, Xv, k):
    Xtr = Xtr.astype(np.float32); Xv = Xv.astype(np.float32)
    tr_sq = (Xtr ** 2).sum(axis=1)                 # (ntr,)
    idx_all = []
    for i in range(0, len(Xv), 500):
        blk = Xv[i:i+500]
        d = (blk ** 2).sum(axis=1, keepdims=True) + tr_sq[None, :] - 2.0 * (blk @ Xtr.T)
        idx_all.append(np.argpartition(d, k, axis=1)[:, :k])
    return np.concatenate(idx_all)

idx = knn_idx(Xtr_log, Xv_log, args.k)             # (nv, k)
preds = Ytr[idx].mean(axis=1)
rmse = np.sqrt(((preds - Yv) ** 2).sum(axis=1)).mean()
print(f"[KNN loc] k={args.k} RMSE={rmse:.2f}m")

votes = np.array([np.bincount(Btr[row], minlength=65).argmax() for row in idx])
top1 = (votes == Bv).mean()
print(f"[KNN beam] k={args.k} Top1={top1:.1%}")
