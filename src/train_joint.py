"""联合模型训练（共享编码器 + 双任务头）

输入: M 个测量波束功率 + 历史 5 步波束索引
输出: 定位 (x,y) 回归 + 64 类波束分类
损失: α·MSE + β·CE

用法:
  venv-torch/Scripts/python.exe src/train_joint.py --M 8 --snr 10 --epochs 50 --demo
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- Dataset ----------------
class BeamDataset(Dataset):
    def __init__(self, powers, locs, beams, M=8, snr_db=10):
        self.powers = powers          # (N,64) float32
        self.locs = locs              # (N,2)
        self.beams = beams            # (N,) 0-63
        self.M = M
        self.snr_db = snr_db

    def __len__(self):
        return len(self.powers)

    def __getitem__(self, i):
        pwr = self.powers[i].copy()
        # SNR 噪声注入（基准=最强波束功率，否则噪声淹没信号）
        sig_power = pwr.max()
        noise_std = np.sqrt(sig_power / (10 ** (self.snr_db / 10)))
        pwr_noisy = pwr + np.random.normal(0, noise_std, 64).astype(np.float32)
        # 功率归一化(0-1)
        pwr_noisy = pwr_noisy / (pwr_noisy.max() + 1e-12)
        # 闭环: M<64 随机选 M 个波束；M=64 全量不打乱
        if self.M >= 64:
            sel = np.arange(64)
        else:
            sel = np.random.choice(64, self.M, replace=False)
        pwr_m = pwr_noisy[sel]
        sel_norm = sel.astype(np.float32) / 63.0   # 波束索引归一化到[0,1]（关键：模型必须知道测的是哪几个波束）
        # 历史 5 步(简化为同一样本的最优波束重复，TODO: 从轨迹取)
        hist = np.full(5, self.beams[i], dtype=np.float32) / 63.0  # 归一化到[0,1]
        x = np.concatenate([pwr_m, sel_norm, hist]).astype(np.float32)
        # 位置归一化到[0,1](训练稳定，评估时缩回米)
        loc_norm = self.locs[i] / 50.0
        return (torch.from_numpy(x),
                torch.from_numpy(loc_norm),
                torch.tensor(self.beams[i]))

# ---------------- Model ----------------
class JointModel(nn.Module):
    def __init__(self, M=8, hist_len=5, hidden=256, n_beams=64):
        super().__init__()
        in_dim = 2 * M + hist_len   # M功率 + M波束索引 + 历史5步
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
        )
        self.loc_head = nn.Linear(hidden, 2)
        self.beam_head = nn.Linear(hidden, n_beams)

    def forward(self, x):
        z = self.encoder(x)
        return self.loc_head(z), self.beam_head(z)

# ---------------- Train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--snr", type=float, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--demo", action="store_true", help="只用 5000 样本快速验证")
    args = ap.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    powers = np.load(os.path.join(data_dir, "powers.npy"))
    locs = np.load(os.path.join(data_dir, "locs.npy"))
    df = pd.read_csv(os.path.join(data_dir, "deepsense_format", "scenario_sionna_1", "dev_data.csv"))
    beams = (df.beam_index.values - 1).astype(np.int64)
    # 轨迹级随机打散 70/15/15（DeepSense 官方做法；区域隔离会导致 val 完全 OOD）
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(powers))
    powers, locs, beams = powers[perm], locs[perm], beams[perm]
    n = len(powers)
    ntr, nvl = int(0.7*n), int(0.85*n)
    Xtr, Ytr, Btr = powers[:ntr], locs[:ntr], beams[:ntr]
    Xv, Yv, Bv = powers[ntr:nvl], locs[ntr:nvl], beams[ntr:nvl]
    if args.demo:
        Xtr, Ytr, Btr = Xtr[:5000], Ytr[:5000], Btr[:5000]
        Xv, Yv, Bv = Xv[:1000], Yv[:1000], Bv[:1000]
    print(f"train={len(Xtr)} val={len(Xv)} M={args.M} snr={args.snr}dB α={args.alpha} β={args.beta}")

    train_ds = BeamDataset(Xtr, Ytr, Btr, args.M, args.snr)
    val_ds = BeamDataset(Xv, Yv, Bv, args.M, args.snr)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = JointModel(M=args.M).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    best_val = float("inf")
    patience = 10
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0
        for x, loc, beam in train_dl:
            x, loc, beam = x.to(device), loc.to(device), beam.to(device)
            opt.zero_grad()
            loc_pred, beam_pred = model(x)
            loss = args.alpha * mse(loc_pred, loc) + args.beta * ce(beam_pred, beam)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0
        loc_errs = []
        beam_hits = 0
        with torch.no_grad():
            for x, loc, beam in val_dl:
                x, loc, beam = x.to(device), loc.to(device), beam.to(device)
                loc_pred, beam_pred = model(x)
                loss = args.alpha * mse(loc_pred, loc) + args.beta * ce(beam_pred, beam)
                val_loss += loss.item() * len(x)
                # RMSE 用米计算(缩回物理坐标)
                loc_errs.append((((loc_pred - loc) * 50.0) ** 2).sum(dim=1).sqrt().cpu().numpy())
                top3 = beam_pred.topk(3, dim=1).indices
                beam_hits += (top3 == beam.unsqueeze(1)).any(dim=1).sum().item()
        val_loss /= len(val_ds)
        rmse = np.mean(np.concatenate(loc_errs))
        top3_acc = beam_hits / len(val_ds)
        sched.step(val_loss)
        print(f"ep{epoch+1}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f} RMSE={rmse:.2f}m Top3={top3_acc:.1%} {time.time()-t0:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "..", "results", "model_best.pth"))
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                print("early stop")
                break

    print(f"DONE best_val={best_val:.4f} RMSE={rmse:.2f}m Top3={top3_acc:.1%}")

if __name__ == "__main__":
    main()
