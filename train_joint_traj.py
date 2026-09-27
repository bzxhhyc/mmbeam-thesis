"""联合模型训练（按轨迹划分 + 真实历史 5 步）

核心突破：历史 5 步的测量波束索引（轨迹连续性先验）+ 按 traj_id 划分
- 历史让模型锁定当前位置在小范围，突破 beam 级 12m 分辨率
- 按轨迹划分（DeepSense 官方做法），轨迹内部历史完整

用法: venv-torch/Scripts/python.exe src/train_joint_traj.py --M 8 --snr 10 --epochs 50
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TrajDataset(Dataset):
    def __init__(self, powers, locs, beams, traj_ids, steps, hist_beams, M=8, snr_db=10):
        self.powers = powers
        self.locs = locs
        self.beams = beams
        self.traj_ids = traj_ids
        self.steps = steps
        self.hist_beams = hist_beams      # (N,5) 前 5 步的波束索引(归一化)
        self.M = M
        self.snr_db = snr_db

    def __len__(self):
        return len(self.powers)

    def __getitem__(self, i):
        pwr = self.powers[i].copy()
        # SNR 噪声注入（基准=最强波束）
        sig = pwr.max()
        noise_std = np.sqrt(sig / (10 ** (self.snr_db / 10)))
        pwr_noisy = pwr + np.random.normal(0, noise_std, 64).astype(np.float32)
        pwr_noisy = pwr_noisy / (pwr_noisy.max() + 1e-12)
        # 闭环: 训练时随机选 M 个波束
        if self.M >= 64:
            sel = np.arange(64)
        else:
            sel = np.random.choice(64, self.M, replace=False)
        pwr_m = pwr_noisy[sel]
        sel_norm = sel.astype(np.float32) / 63.0
        x = np.concatenate([pwr_m, sel_norm, self.hist_beams[i]]).astype(np.float32)
        loc_norm = self.locs[i] / 50.0
        return (torch.from_numpy(x), torch.from_numpy(loc_norm), torch.tensor(self.beams[i]))

def build_hist(beams, traj_ids, steps, hist_len=5):
    """构造真实历史：同一 traj 内，step<hist_len 的用当前 beam 补，否则取前 hist_len 步"""
    N = len(beams)
    hist = np.zeros((N, hist_len), dtype=np.float32)
    # 按 (traj_id, step) 排序索引
    order = np.lexsort((steps, traj_ids))
    sorted_traj = traj_ids[order]
    sorted_beam = beams[order]
    # 建 traj 内部位置映射: (traj_id, step) -> 原索引
    for i in range(N):
        tid, st = traj_ids[i], steps[i]
        h = []
        for k in range(1, hist_len + 1):
            prev_step = st - k
            if prev_step >= 0:
                # 找同 traj 同 step 的原索引（用 order 反查）
                # 简化：同 traj 连续存储，直接按相对位置取
                # 找到 i 在 order 中的位置
                pos_in_order = np.where(order == i)[0][0]
                prev_pos = pos_in_order - k
                if prev_pos >= 0 and sorted_traj[prev_pos] == tid:
                    h.append(sorted_beam[prev_pos] / 63.0)
                else:
                    h.append(beams[i] / 63.0)
            else:
                h.append(beams[i] / 63.0)   # 轨迹开头用当前 beam 补
        hist[i] = np.array(h[::-1])   # 逆序：最早的在最前
    return hist

class JointModel(nn.Module):
    def __init__(self, M=8, hist_len=5, hidden=256, n_beams=64):
        super().__init__()
        in_dim = 2 * M + hist_len
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
        )
        self.loc_head = nn.Linear(hidden, 2)
        self.beam_head = nn.Linear(hidden, n_beams)

    def forward(self, x):
        z = self.encoder(x)
        return self.loc_head(z), self.beam_head(z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--snr", type=float, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    powers = np.load(os.path.join(data_dir, "powers.npy"))
    locs = np.load(os.path.join(data_dir, "locs.npy"))
    traj_ids = np.load(os.path.join(data_dir, "traj_id.npy"))
    steps = np.load(os.path.join(data_dir, "step.npy"))
    is_offgrid = np.load(os.path.join(data_dir, "is_offgrid.npy"))
    df = pd.read_csv(os.path.join(data_dir, "deepsense_format", "scenario_sionna_1", "dev_data.csv"))
    beams = (df.beam_index.values - 1).astype(np.int64)

    # 只取主轨迹（is_offgrid=0），offgrid 留作泛化测试
    main_mask = is_offgrid == 0
    powers, locs, beams, traj_ids, steps = powers[main_mask], locs[main_mask], beams[main_mask], traj_ids[main_mask], steps[main_mask]
    print(f"main samples={len(powers)} offgrid={is_offgrid.sum()}")

    # 构造真实历史
    print("building history...")
    hist_beams = build_hist(beams, traj_ids, steps)

    # 按 traj_id 划分 70/15/15（DeepSense 官方做法）
    uniq_traj = np.unique(traj_ids)
    rng = np.random.default_rng(42)
    perm_traj = rng.permutation(uniq_traj)
    n = len(perm_traj)
    tr_traj = set(perm_traj[:int(0.7*n)].tolist())
    vl_traj = set(perm_traj[int(0.7*n):int(0.85*n)].tolist())
    te_traj = set(perm_traj[int(0.85*n):].tolist())
    tr_mask = np.array([t in tr_traj for t in traj_ids])
    vl_mask = np.array([t in vl_traj for t in traj_ids])
    te_mask = np.array([t in te_traj for t in traj_ids])

    if args.demo:
        tr_mask = tr_mask & (np.random.default_rng(1).random(len(tr_mask)) < 0.15)
        vl_mask = vl_mask & (np.random.default_rng(2).random(len(vl_mask)) < 0.15)

    Xtr, Ytr, Btr, Htr = powers[tr_mask], locs[tr_mask], beams[tr_mask], hist_beams[tr_mask]
    Xv, Yv, Bv, Hv = powers[vl_mask], locs[vl_mask], beams[vl_mask], hist_beams[vl_mask]
    print(f"train={len(Xtr)} val={len(Xv)} test={te_mask.sum()} M={args.M} snr={args.snr}dB")

    train_ds = TrajDataset(Xtr, Ytr, Btr, traj_ids[tr_mask], steps[tr_mask], Htr, args.M, args.snr)
    val_ds = TrajDataset(Xv, Yv, Bv, traj_ids[vl_mask], steps[vl_mask], Hv, args.M, args.snr)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = JointModel(M=args.M).to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    mse = nn.MSELoss(); ce = nn.CrossEntropyLoss()

    best_val = float("inf"); patience = 10
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); tl = 0
        for x, loc, beam in train_dl:
            x, loc, beam = x.to(device), loc.to(device), beam.to(device)
            opt.zero_grad()
            lp, bp = model(x)
            loss = args.alpha * mse(lp, loc) + args.beta * ce(bp, beam)
            loss.backward(); opt.step()
            tl += loss.item() * len(x)
        tl /= len(train_ds)
        model.eval(); vl = 0; errs = []; hits = 0
        with torch.no_grad():
            for x, loc, beam in val_dl:
                x, loc, beam = x.to(device), loc.to(device), beam.to(device)
                lp, bp = model(x)
                loss = args.alpha * mse(lp, loc) + args.beta * ce(bp, beam)
                vl += loss.item() * len(x)
                errs.append((((lp - loc) * 50.0) ** 2).sum(1).sqrt().cpu().numpy())
                t3 = bp.topk(3, 1).indices
                hits += (t3 == beam.unsqueeze(1)).any(1).sum().item()
        vl /= len(val_ds)
        rmse = np.mean(np.concatenate(errs))
        top3 = hits / len(val_ds)
        sched.step(vl)
        print(f"ep{epoch+1}/{args.epochs} train={tl:.4f} val={vl:.4f} RMSE={rmse:.2f}m Top3={top3:.1%} {time.time()-t0:.0f}s")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(data_dir, "..", "results", "model_traj_best.pth"))
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                print("early stop"); break
    print(f"DONE best_val={best_val:.4f} RMSE={rmse:.2f}m Top3={top3:.1%}")

if __name__ == "__main__":
    main()
