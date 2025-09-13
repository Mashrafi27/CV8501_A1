#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import yaml
from torchvision.models.video import r3d_18, R3D_18_Weights

class MRI3DNPY(Dataset):
    def __init__(self, df, target_shape=(64,64,64)):
        self.df = df.reset_index(drop=True)
        self.tgt = target_shape
    def __len__(self): return len(self.df)
    def _resize_3d(self, vol):
        v = torch.from_numpy(vol[None, None, ...]).float()
        v = torch.nn.functional.interpolate(v, size=self.tgt, mode="trilinear", align_corners=False)
        return v[0]  # (1,D,H,W)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = np.load(r["std_path"]).astype(np.float32)
        v   = self._resize_3d(vol)             # (1,D,H,W)
        y   = int(r["DIAGNOSIS"]) - 1
        pid = str(r["PTID"])
        return v, y, pid

def load_cfg(p):
    with open(p, "r") as f: return yaml.safe_load(f)

def read_ids(splits_root):
    R = Path(splits_root)
    rd = lambda n: [s for s in (R/n).read_text().splitlines() if s]
    return set(rd("train_ids.txt")), set(rd("val_ids.txt")), set(rd("test_ids.txt"))

@torch.no_grad()
def eval_probs(model, loader, device):
    model.eval()
    P, Y, ID = [], [], []
    for v, y, pid in loader:
        x = v.to(device).expand(-1,3,-1,-1,-1)   # 1→3 channels
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        P.append(probs); Y.append(y.numpy()); ID.extend(pid)
    P = np.concatenate(P); Y = np.concatenate(Y); ID = np.array(ID)
    yhat = P.argmax(1)
    m = {
        "macro_f1": f1_score(Y, yhat, average="macro"),
        "balanced_acc": balanced_accuracy_score(Y, yhat)
    }
    try: m["roc_auc_ovo"] = roc_auc_score(Y, P, multi_class="ovo")
    except: m["roc_auc_ovo"] = float("nan")
    return m, P, Y, ID

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_cfg(args.cfg); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MRI index + labels (use EHR parquet for labels for consistency)
    std_idx = pd.read_csv(Path(cfg["interim_root"]) / "mri_std" / "index.csv")       # PTID, std_path, ...
    labels  = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")[["PTID","DIAGNOSIS"]]
    df = std_idx.merge(labels, on="PTID", how="inner")
    df = df[df["DIAGNOSIS"].notna()].reset_index(drop=True)

    # splits
    tr_ids, va_ids, te_ids = read_ids(cfg["splits_root"])
    df_tr = df[df["PTID"].isin(tr_ids)].reset_index(drop=True)
    df_va = df[df["PTID"].isin(va_ids)].reset_index(drop=True)
    df_te = df[df["PTID"].isin(te_ids)].reset_index(drop=True)

    dl_tr = DataLoader(MRI3DNPY(df_tr), batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(MRI3DNPY(df_va), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(MRI3DNPY(df_te), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # model
    weights = R3D_18_Weights.KINETICS400_V1 if args.pretrained else None
    model = r3d_18(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 3)
    model = model.to(device)

    # class weights
    y_tr = df_tr["DIAGNOSIS"].astype(int).values - 1
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    w = counts.sum() / (counts + 1e-9); w /= w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1, best_state, patience, bad = -1.0, None, 6, 0
    for ep in range(1, args.epochs+1):
        model.train(); loss_sum = 0.0
        for v, y, _ in tqdm(dl_tr, desc=f"MRI epoch {ep}/{args.epochs}", leave=False):
            x = v.to(device).expand(-1,3,-1,-1,-1); y = y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss)
        m_val, _, _, _ = eval_probs(model, dl_va, device)
        print(f"[{ep:03d}] loss={loss_sum/len(dl_tr):.4f} | val_f1={m_val['macro_f1']:.4f}")
        if m_val["macro_f1"] > best_f1:
            best_f1, best_state, bad = m_val["macro_f1"], model.state_dict(), 0
        else:
            bad += 1
        if bad >= patience: break

    # save & export probs
    model.load_state_dict(best_state)
    out = Path(cfg["processed_root"]).parents[0] / "Models" / "mri_only_r3d18"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out/"model.pt")

    m_val, Pv, Yv, IDv = eval_probs(model, dl_va, device)
    m_te,  Pt, Yt, IDt = eval_probs(model, dl_te, device)
    np.save(out/"val_probs.npy", Pv); np.save(out/"val_labels.npy", Yv); np.save(out/"val_ids.npy", IDv)
    np.save(out/"test_probs.npy", Pt); np.save(out/"test_labels.npy", Yt); np.save(out/"test_ids.npy", IDt)
    (out/"metrics.json").write_text(json.dumps({"val_best_f1": best_f1, "val": m_val, "test": m_te}, indent=2))
    print(f"[OK] MRI baseline saved → {out}")

if __name__ == "__main__":
    main()
