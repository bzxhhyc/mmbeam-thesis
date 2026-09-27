"""CNN 版联合模型：把 64 维功率向量重塑为 8×8 空间图，用卷积提取空间特征
突破 MLP 丢失的"相邻波束空间关系"，目标亚 beam 级定位。
用法: venv-torch/Scripts/python.exe src/train_cnn.py --M 64 --snr 10 --epochs 60
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TrajDataset(Dataset):
    def __init__(self, powers, locs, beams, hist_beams, M=64, snr_db=10):
        self.powers = powers; self.locs = locs; self.beams = beams
        self.hist_beams = hist_beams; self.M = M; self.snr_db = snr_db

    def __len__(self): return len(self.powers)

    def __getitem__(self, i):
        pwr = self.powers[i].copy()
        sig = pwr.max()
        noise_std = np.sqrt(sig / (10 ** (self.snr_db / 10)))
        pwr_noisy = pwr + np.random.normal(0, noise_std, 64).astype(np.float32)
        pwr_noisy = pwr_noisy / (pwr_noisy.max() + 1e-12)
        # 闭环选 M 个波束（CNN 版统一用 M=64 全功率图；M<64 时置未测为 0）
        pwr_map = np.zeros(64, dtype=np.float32)
        if self.M >= 64:
            sel = np.arange(64)
        else:
            sel = np.random.choice(64, self.M, replace=False)
        pwr_map[sel] = pwr_noisy[sel]
        img = pwr_map.reshape(1, 8, 8)               # (1,8,8) 空间图
        hist = torch.from_numpy(self.hist_beams[i])   # (5,)
        return (torch.from_numpy(img), hist, torch.from_numpy(self.locs[i] / 50.0), torch.tensor(self.beams[i]))

class CNNJoint(nn.Module):
    def __init__(self, n_beams=64, hist_len=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),   # (128,)
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + hist_len, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
        )
        self.loc_head = nn.Linear(256, 2)
        self.beam_head = nn.Linear(256, n_beams)

    def forward(self, img, hist):
        f = self.conv(img)
        z = self.fuse(torch.cat([f, hist], dim=1))
        return self.loc_head(z), self.beam_head(z)

def build_hist(beams, traj_ids, steps, hist_len=5):
    N = len(beams); hist = np.zeros((N, hist_len), dtype=np.float32)
    order = np.lexsort((steps, traj_ids)); straj = traj_ids[order]; sbeam = beams[order]
    pos_of = {int(orig): k for k, orig in enumerate(order)}
    for i in range(N):
        tid, st = traj_ids[i], steps[i]; h = []
        po = pos_of[int(i)]
        for k in range(1, hist_len + 1):
            pp = po - k
            if pp >= 0 and straj[pp] == tid: h.append(sbeam[pp] / 63.0)
            else: h.append(beams[i] / 63.0)
        hist[i] = np.array(h[::-1])
    return hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--snr", type=float, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    powers = np.load(os.path.join(data_dir, "powers.npy"))
    locs = np.load(os.path.join(data_dir, "locs.npy"))
    traj_ids = np.load(os.path.join(data_dir, "traj_id.npy"))
    steps = np.load(os.path.join(data_dir, "step.npy"))
    is_offgrid = np.load(os.path.join(data_dir, "is_offgrid.npy"))
    df = pd.read_csv(os.path.join(data_dir, "deepsense_format", "scenario_sionna_1", "dev_data.csv"))
    beams = (df.beam_index.values - 1).astype(np.int64)

    main_mask = is_offgrid == 0
    powers, locs, beams, traj_ids, steps = powers[main_mask], locs[main_mask], beams[main_mask], traj_ids[main_mask], steps[main_mask]
    hist_beams = build_hist(beams, traj_ids, steps)

    uniq = np.unique(traj_ids); rng = np.random.default_rng(42); perm = rng.permutation(uniq); n = len(perm)
    tr_t = set(perm[:int(0.7*n)].tolist()); vl_t = set(perm[int(0.7*n):int(0.85*n)].tolist())
    trm = np.array([t in tr_t for t in traj_ids]); vlm = np.array([t in vl_t for t in traj_ids])

    train_ds = TrajDataset(powers[trm], locs[trm], beams[trm], hist_beams[trm], args.M, args.snr)
    val_ds = TrajDataset(powers[vlm], locs[vlm], beams[vlm], hist_beams[vlm], args.M, args.snr)
    print(f"train={len(train_ds)} val={len(val_ds)} M={args.M} snr={args.snr}dB CNN")
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = CNNJoint().to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
    mse = nn.MSELoss(); ce = nn.CrossEntropyLoss()

    best_val = float("inf"); patience = 12
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); tl = 0
        for img, hist, loc, beam in train_dl:
            img, hist, loc, beam = img.to(device), hist.to(device), loc.to(device), beam.to(device)
            opt.zero_grad()
            lp, bp = model(img, hist)
            loss = args.alpha * mse(lp, loc) + args.beta * ce(bp, beam)
            loss.backward(); opt.step()
            tl += loss.item() * len(img)
        tl /= len(train_ds)
        model.eval(); vl = 0; errs = []; hits = 0
        with torch.no_grad():
            for img, hist, loc, beam in val_dl:
                img, hist, loc, beam = img.to(device), hist.to(device), loc.to(device), beam.to(device)
                lp, bp = model(img, hist)
                loss = args.alpha * mse(lp, loc) + args.beta * ce(bp, beam)
                vl += loss.item() * len(img)
                errs.append((((lp - loc) * 50.0) ** 2).sum(1).sqrt().cpu().numpy())
                t3 = bp.topk(3, 1).indices
                hits += (t3 == beam.unsqueeze(1)).any(1).sum().item()
        vl /= len(val_ds); rmse = np.mean(np.concatenate(errs)); top3 = hits / len(val_ds)
        sched.step(vl)
        print(f"ep{epoch+1}/{args.epochs} train={tl:.4f} val={vl:.4f} RMSE={rmse:.2f}m Top3={top3:.1%} {time.time()-t0:.0f}s", flush=True)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(data_dir, "..", "results", "model_cnn_best.pth"))
            patience = 12
        else:
            patience -= 1
            if patience == 0: print("early stop"); break
    print(f"DONE CNN best_val={best_val:.4f} RMSE={rmse:.2f}m Top3={top3:.1%}", flush=True)

if __name__ == "__main__":
    main()
