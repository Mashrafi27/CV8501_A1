#!/usr/bin/env python3
"""
Late Fusion interpretability:
- Loads unimodal EHR-only and MRI-only checkpoints + test set
- Recomputes test probabilities for both modalities
- Computes fused probabilities with weight alpha
- Exports per-sample CSV with:
    PTID, y, yhat_ehr, yhat_mri, yhat_fused, 
    p_ehr[c*], p_mri[c*], p_fused[c*], 
    contrib_mri = alpha*(p_mri[c*] - p_ehr[c*]),
    contrib_ehr = (1-alpha)*(p_ehr[c*] - p_mri[c*])
  where c* is the fused predicted class (or use true class with --use_true_class)
- Runs ablation metrics (EHR-only, MRI-only, Fused) on test set and saves JSON

Usage:
  python -m src.interpret.late_fusion_contrib --cfg config/adni.yaml \
    --ehr_model work/ADNI/Models/ehr_only_mlp/model.pt \
    --mri_model work/ADNI/Models/mri_only_r3d18/model.pt \
    --alpha 0.5 [--use_true_class]
"""

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from torchvision.models.video import r3d_18

# ---- Minimal copies of datasets/models used in training ----
class EHRDs(Dataset):
    def __init__(self, df, feat_cols):
        self.X = torch.tensor(df[feat_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df["DIAGNOSIS"].astype(int).values - 1, dtype=torch.long)
        self.ids = df["PTID"].astype(str).values
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.ids[i]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, x): return self.net(x)

class MRI3DNPY(Dataset):
    def __init__(self, df, target_shape=(64,64,64)):
        self.df = df.reset_index(drop=True); self.tgt = target_shape
    def __len__(self): return len(self.df)
    def _resize_3d(self, vol):
        v = torch.from_numpy(vol[None, None, ...]).float()
        v = torch.nn.functional.interpolate(v, size=self.tgt, mode="trilinear", align_corners=False)
        return v[0]
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
def probs_ehr(model, loader, device):
    P, Y, ID = [], [], []
    model.eval()
    for X, y, pid in loader:
        P.append(torch.softmax(model(X.to(device)), dim=1).cpu().numpy())
        Y.append(y.numpy()); ID.extend(pid)
    return np.concatenate(P), np.concatenate(Y), np.array(ID)

@torch.no_grad()
def probs_mri(model, loader, device):
    P, Y, ID = [], [], []
    model.eval()
    for v, y, pid in loader:
        x = v.to(device).expand(-1,3,-1,-1,-1)
        P.append(torch.softmax(model(x), dim=1).cpu().numpy())
        Y.append(y.numpy()); ID.extend(pid)
    return np.concatenate(P), np.concatenate(Y), np.array(ID)

def metrics(y, P):
    yhat = P.argmax(1)
    out = {
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y, yhat))
    }
    try: out["roc_auc_ovo"] = float(roc_auc_score(y, P, multi_class="ovo"))
    except: out["roc_auc_ovo"] = float("nan")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    ap.add_argument("--ehr_model", required=True)
    ap.add_argument("--mri_model", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--use_true_class", action="store_true", help="attribute wrt true class instead of fused predicted")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load tabular and MRI indices + test set ----
    ehr = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")
    feat_cols = [c for c in ehr.columns if c not in {"PTID","DIAGNOSIS"}]
    ehr[feat_cols] = ehr[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    std_idx = pd.read_csv(Path(cfg["interim_root"]) / "mri_std" / "index.csv")
    df_all  = std_idx.merge(ehr[["PTID","DIAGNOSIS"]], on="PTID", how="inner").dropna(subset=["DIAGNOSIS"])

    _, _, te_ids = read_ids(cfg["splits_root"])
    df_te_ehr = ehr[ehr["PTID"].isin(te_ids)].dropna(subset=["DIAGNOSIS"]).reset_index(drop=True)
    df_te_mri = df_all[df_all["PTID"].isin(te_ids)].reset_index(drop=True)

    # ---- Dataloaders ----
    dl_ehr = DataLoader(EHRDs(df_te_ehr, feat_cols), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    dl_mri = DataLoader(MRI3DNPY(df_te_mri), batch_size=1, shuffle=False, num_workers=2, pin_memory=True)  # keep 1 to align pids easily

    # ---- Models ----
    # EHR
    ehr_model = MLP(len(feat_cols)).to(device).eval()
    ehr_model.load_state_dict(torch.load(args.ehr_model, map_location=device))
    # MRI
    mri_model = r3d_18(weights=None); mri_model.fc = nn.Linear(mri_model.fc.in_features, 3)
    mri_model.load_state_dict(torch.load(args.mri_model, map_location=device))
    mri_model = mri_model.to(device).eval()

    # ---- Probabilities ----
    P_ehr, y_ehr, id_ehr = probs_ehr(ehr_model, dl_ehr, device)
    P_mri, y_mri, id_mri = probs_mri(mri_model, dl_mri, device)

    # Align by PTID (inner join on ids)
    df_e = pd.DataFrame({"PTID": id_ehr, "y": y_ehr, "P_ehr": list(P_ehr)})
    df_m = pd.DataFrame({"PTID": id_mri, "y_mri": y_mri, "P_mri": list(P_mri)})
    df = df_e.merge(df_m, on="PTID", how="inner")
    y = df["y"].values.astype(int)
    P_e = np.stack(df["P_ehr"].values)
    P_m = np.stack(df["P_mri"].values)

    # fused
    a = np.clip(args.alpha, 0.0, 1.0)
    P_f = a*P_m + (1.0 - a)*P_e

    # per-sample attribution wrt chosen class
    if args.use_true_class:
        c_star = y
    else:
        c_star = P_f.argmax(1)

    idx = np.arange(len(df))
    p_e_star = P_e[idx, c_star]
    p_m_star = P_m[idx, c_star]
    p_f_star = P_f[idx, c_star]

    contrib_mri = a * (p_m_star - p_e_star)
    contrib_ehr = (1.0 - a) * (p_e_star - p_m_star)

    # write CSV
    out_dir = Path(cfg["processed_root"]).parents[0] / "Interpretability" / "LateFusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for i in range(len(df)):
        out_rows.append({
            "PTID": df.iloc[i]["PTID"],
            "y": int(y[i]),
            "yhat_ehr": int(P_e[i].argmax()),
            "yhat_mri": int(P_m[i].argmax()),
            "yhat_fused": int(P_f[i].argmax()),
            "p_ehr_star": float(p_e_star[i]),
            "p_mri_star": float(p_m_star[i]),
            "p_fused_star": float(p_f_star[i]),
            "contrib_mri": float(contrib_mri[i]),
            "contrib_ehr": float(contrib_ehr[i]),
        })
    pd.DataFrame(out_rows).to_csv(out_dir/"per_sample_contrib.csv", index=False)

    # ablation metrics
    mets = {
        "ehr_only": metrics(y, P_e),
        "mri_only": metrics(y, P_m),
        f"fused_alpha_{a:.2f}": metrics(y, P_f)
    }
    (out_dir/"ablation_metrics.json").write_text(json.dumps(mets, indent=2))
    print(f"[OK] wrote {out_dir}/per_sample_contrib.csv and ablation_metrics.json")

if __name__ == "__main__":
    main()
