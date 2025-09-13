#!/usr/bin/env python3
"""
SHAP values for EHR MLP (KernelExplainer).
WARNING: KernelExplainer can be slow; use a small background set.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

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
    import yaml
    with open(p,"r") as f: return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--background", type=int, default=200)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    ehr = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")
    feat_cols = [c for c in ehr.columns if c not in {"PTID","DIAGNOSIS"}]
    ehr[feat_cols] = ehr[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    X = ehr[feat_cols].values
    y = ehr["DIAGNOSIS"].astype(int).values - 1

    # small background and sample subsets
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X), size=min(args.background, len(X)), replace=False)
    sm_idx = rng.choice(len(X), size=min(args.samples, len(X)), replace=False)
    X_bg, X_sm = X[bg_idx], X[sm_idx]

    model = MLP(in_dim=len(feat_cols), hidden=512, out_dim=3, dropout=args.dropout).eval()
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)

    def f(x_np):
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32)
            p = torch.softmax(model(x), dim=1).numpy()
        return p

    expl = shap.KernelExplainer(f, X_bg)
    shap_values = expl.shap_values(X_sm, nsamples="auto")   # list of arrays (per class)
    out_dir = Path(cfg["processed_root"]).parents[0] / "Interpretability" / "EHR"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir/"shap_values.npy", shap_values, allow_pickle=True)
    np.save(out_dir/"shap_samples.npy", X_sm)

    # summary plot for the winning class per sample
    try:
        yhat = np.argmax(f(X_sm), axis=1)
        sv_for_pred = np.array([shap_values[c][i] for i, c in enumerate(yhat)])
        shap.summary_plot(sv_for_pred, features=X_sm, feature_names=feat_cols, show=False)
        plt.tight_layout(); plt.savefig(out_dir/"shap_summary.png", dpi=180)
        print(f"[OK] saved shap_summary.png")
    except Exception as e:
        print(f"[WARN] SHAP summary plot failed: {e}")

if __name__ == "__main__":
    main()
