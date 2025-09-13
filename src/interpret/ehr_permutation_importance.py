#!/usr/bin/env python3
"""
Permutation importance for EHR MLP on the test set.
- Loads cfg, EHR parquet, splits, and the saved MLP checkpoint
- Computes baseline performance, then shuffles each feature column and recomputes
- Saves a CSV with importance scores (Δ macro-F1 and Δ balanced accuracy)
- Produces a bar plot of top-K features

Usage:
  python -m src.interpret.ehr_permutation_importance --cfg config/adni.yaml \
         --model work/ADNI/Models/ehr_only_mlp/model.pt --topk 20
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch, yaml, json
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt

# ---- EHR dataset (reuse minimal) ----
class EHRDs(Dataset):
    def __init__(self, df, feat_cols):
        self.X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df["DIAGNOSIS"].astype(int).values - 1, dtype=torch.long)
        self.ids = df["PTID"].astype(str).values
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.ids[i]

# ---- same MLP as training ----
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
def evaluate(model, loader, device):
    model.eval()
    P, Y = [], []
    for X, y, _ in loader:
        probs = torch.softmax(model(X.to(device)), dim=1).cpu().numpy()
        P.append(probs); Y.append(y.numpy())
    P = np.concatenate(P); Y = np.concatenate(Y)
    yhat = P.argmax(1)
    return dict(macro_f1=f1_score(Y, yhat, average="macro"),
                bal_acc=balanced_accuracy_score(Y, yhat))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = load_cfg(args.cfg); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ehr = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")
    feat_cols = [c for c in ehr.columns if c not in {"PTID","DIAGNOSIS"}]
    ehr[feat_cols] = ehr[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    _, _, te_ids = read_ids(cfg["splits_root"])
    df_te = ehr[ehr["PTID"].isin(te_ids) & ehr["DIAGNOSIS"].notna()].reset_index(drop=True)

    ds_te = EHRDs(df_te, feat_cols)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = MLP(in_dim=len(feat_cols), hidden=512, out_dim=3, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    base = evaluate(model, dl_te, device)

    # Permutation importance
    deltas = []
    X_np = df_te[feat_cols].values.copy()
    y_np = (df_te["DIAGNOSIS"].astype(int).values - 1).copy()

    for j, col in enumerate(feat_cols):
        X_perm = X_np.copy()
        perm_idx = np.random.permutation(X_perm.shape[0])
        X_perm[:, j] = X_perm[perm_idx, j]
        ds_perm = EHRDs(pd.DataFrame({"PTID": df_te["PTID"], "DIAGNOSIS": df_te["DIAGNOSIS"]}), feat_cols)
        # hack: overwrite ds_perm tensors
        ds_perm.X = torch.tensor(X_perm, dtype=torch.float32)
        dl_perm = DataLoader(ds_perm, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
        m = evaluate(model, dl_perm, device)
        deltas.append((col, base["macro_f1"] - m["macro_f1"], base["bal_acc"] - m["bal_acc"]))

    out_dir = Path(cfg["processed_root"]).parents[0] / "Interpretability" / "EHR"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_imp = pd.DataFrame(deltas, columns=["feature", "delta_macro_f1", "delta_bal_acc"])
    df_imp.sort_values("delta_macro_f1", ascending=False, inplace=True)
    df_imp.to_csv(out_dir / "permutation_importance.csv", index=False)

    # Plot top-K
    topk = df_imp.head(args.topk)
    plt.figure(figsize=(8, max(4, 0.3*len(topk))))
    plt.barh(topk["feature"][::-1], topk["delta_macro_f1"][::-1])
    plt.xlabel("Δ Macro-F1 (higher = more important)")
    plt.tight_layout()
    plt.savefig(out_dir / "perm_importance_topk.png", dpi=180)
    print(f"[OK] saved: {out_dir}/permutation_importance.csv and perm_importance_topk.png")

if __name__ == "__main__":
    main()
