#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import yaml

# ----- tiny EHR dataset -----
class EHRDs(Dataset):
    def __init__(self, df, feat_cols):
        self.X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df["DIAGNOSIS"].astype(int).values - 1, dtype=torch.long)
        self.ids = df["PTID"].astype(str).values
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.ids[i]

# ----- simple MLP encoder+head -----
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, x): return self.net(x)

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
    for X, y, pid in loader:
        probs = torch.softmax(model(X.to(device)), dim=1).cpu().numpy()
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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_cfg(args.cfg); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EHR features & columns
    ehr = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")  # PTID, DIAGNOSIS, features...
    feat_cols = [c for c in ehr.columns if c not in {"PTID","DIAGNOSIS"}]
    ehr[feat_cols] = ehr[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    # splits
    tr_ids, va_ids, te_ids = read_ids(cfg["splits_root"])
    df_tr = ehr[ehr["PTID"].isin(tr_ids) & ehr["DIAGNOSIS"].notna()].reset_index(drop=True)
    df_va = ehr[ehr["PTID"].isin(va_ids) & ehr["DIAGNOSIS"].notna()].reset_index(drop=True)
    df_te = ehr[ehr["PTID"].isin(te_ids) & ehr["DIAGNOSIS"].notna()].reset_index(drop=True)

    dl_tr = DataLoader(EHRDs(df_tr, feat_cols), batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(EHRDs(df_va, feat_cols), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(EHRDs(df_te, feat_cols), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # model & loss
    model = MLP(in_dim=len(feat_cols), hidden=512, out_dim=3, dropout=args.dropout).to(device)
    y_tr = df_tr["DIAGNOSIS"].astype(int).values - 1
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    w = counts.sum() / (counts + 1e-9); w /= w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1, best_state, patience, bad = -1.0, None, 10, 0
    for ep in range(1, args.epochs+1):
        model.train(); loss_sum = 0.0
        pbar = tqdm(dl_tr, desc=f"EHR epoch {ep}/{args.epochs}", leave=False)
        for X, y, _ in pbar:
            X, y = X.to(device), y.to(device)
            loss = crit(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss)
            pbar.set_postfix(loss=f"{loss_sum/max(1,pbar.n):.4f}")
        m_val, _, _, _ = eval_probs(model, dl_va, device)
        print(f"[{ep:03d}] loss={loss_sum/len(dl_tr):.4f} | val_f1={m_val['macro_f1']:.4f}")
        if m_val["macro_f1"] > best_f1:
            best_f1, best_state, bad = m_val["macro_f1"], model.state_dict(), 0
        else:
            bad += 1
        if bad >= patience: break

    # save & export probs
    model.load_state_dict(best_state)
    out = Path(cfg["processed_root"]).parents[0] / "Models" / "ehr_only_mlp"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out/"model.pt")

    m_val, Pv, Yv, IDv = eval_probs(model, dl_va, device)
    m_te,  Pt, Yt, IDt = eval_probs(model, dl_te, device)
    np.save(out/"val_probs.npy", Pv); np.save(out/"val_labels.npy", Yv); np.save(out/"val_ids.npy", IDv)
    np.save(out/"test_probs.npy", Pt); np.save(out/"test_labels.npy", Yt); np.save(out/"test_ids.npy", IDt)
    (out/"metrics.json").write_text(json.dumps({"val_best_f1": best_f1, "val": m_val, "test": m_te}, indent=2))
    print(f"[OK] EHR baseline saved → {out}")

if __name__ == "__main__":
    main()
