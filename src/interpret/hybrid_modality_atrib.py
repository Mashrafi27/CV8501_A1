#!/usr/bin/env python3
"""
Hybrid fusion interpretability:
- Loads trained HybridModel and test set
- Registers forward hooks to capture z_mri and z_ehr (pre-fusion embeddings)
- Backprops chosen logit to get grads wrt z_mri, z_ehr
- Computes per-sample modality attribution: ||∂logit/∂z_mri|| and ||∂logit/∂z_ehr||
- Also exports per-feature gradient*input scores for EHR inputs (optional)
- Saves CSV summaries and a few plots

Assumptions:
- Your HybridModel has attributes: mri_backbone, ehr_enc, and a fusion head that consumes [z_mri, z_ehr]
- The pre-fusion tensors can be captured via hooks on modules named 'mri_proj' and 'ehr_proj' (or last layers of each path).
  If names differ, pass --mri_hook and --ehr_hook as dot paths (e.g., 'mri_backbone.layer4.1.conv2').

Usage:
  python -m src.interpret.hybrid_modality_attrib --cfg config/adni.yaml \
    --model work/ADNI/Models/hybrid/model.pt \
    --mri_hook mri_proj --ehr_hook ehr_proj --per_class 3
"""

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, yaml, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, balanced_accuracy_score

# ---- Minimal datasets (reuse) ----
class EHRDsLite(Dataset):
    def __init__(self, df, feat_cols):
        self.df = df.reset_index(drop=True)
        self.X = torch.tensor(self.df[feat_cols].values, dtype=torch.float32)
        self.y = torch.tensor(self.df["DIAGNOSIS"].astype(int).values - 1, dtype=torch.long)
        self.ids = self.df["PTID"].astype(str).values
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.ids[i]

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
    with open(p,"r") as f: return yaml.safe_load(f)

def read_ids(splits_root):
    R = Path(splits_root)
    rd = lambda n: [s for s in (R/n).read_text().splitlines() if s]
    return set(rd("train_ids.txt")), set(rd("val_ids.txt")), set(rd("test_ids.txt"))

def get_module_by_path(root: nn.Module, path: str):
    m = root
    for name in path.split("."):
        m = getattr(m, name)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mri_hook", default="mri_proj", help="module name/path that outputs z_mri")
    ap.add_argument("--ehr_hook", default="ehr_proj", help="module name/path that outputs z_ehr")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--use_true_class", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data: build a single aligned test dataframe ----
    ehr = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")
    feat_cols = [c for c in ehr.columns if c not in {"PTID","DIAGNOSIS"}]
    ehr[feat_cols] = ehr[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    std_idx = pd.read_csv(Path(cfg["interim_root"]) / "mri_std" / "index.csv")
    df_all  = std_idx.merge(ehr, on="PTID", how="inner").dropna(subset=["DIAGNOSIS"]).reset_index(drop=True)
    _, _, te_ids = read_ids(cfg["splits_root"])
    df_te = df_all[df_all["PTID"].isin(te_ids)].reset_index(drop=True)

    # ---- Hybrid model import ----
    from src.models.hybrid_model import HybridModel  # uses your existing class
    model = HybridModel(n_ehr_features=len(feat_cols), pretrained_mri=False, dropout=0.2).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # ---- Register hooks on pre-fusion tensors ----
    feats = {"z_mri": None, "z_ehr": None, "g_mri": None, "g_ehr": None}

    mri_mod = get_module_by_path(model, args.mri_hook)
    ehr_mod = get_module_by_path(model, args.ehr_hook)

    def hook_mri(module, inp, out):
        feats["z_mri"] = out
        def _bwd(grad): feats["g_mri"] = grad
        out.register_hook(_bwd)
    def hook_ehr(module, inp, out):
        feats["z_ehr"] = out
        def _bwd(grad): feats["g_ehr"] = grad
        out.register_hook(_bwd)

    h1 = mri_mod.register_forward_hook(hook_mri)
    h2 = ehr_mod.register_forward_hook(hook_ehr)

    # ---- Iterate test set one-by-one (keeps alignment simple) ----
    out_dir = Path(cfg["processed_root"]).parents[0] / "Interpretability" / "Hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(len(df_te)):
        r = df_te.iloc[i]
        # inputs
        vol = np.load(r["std_path"]).astype(np.float32)
        v = torch.from_numpy(vol[None,None,...]).float()
        v = torch.nn.functional.interpolate(v, size=(64,64,64), mode="trilinear", align_corners=False)[0:1]  # (1,1,D,H,W)
        e = torch.tensor(r[feat_cols].values[None,:], dtype=torch.float32)
        y = torch.tensor([int(r["DIAGNOSIS"])-1], dtype=torch.long)

        v = v.to(device); e = e.to(device); y = y.to(device)

        # forward + choose class
        logits = model(v.expand(-1,3,-1,-1,-1), e)
        if args.use_true_class:
            c = y
        else:
            c = logits.argmax(1)

        # backprop chosen logit
        model.zero_grad(set_to_none=True)
        sel = logits.gather(1, c.view(-1,1)).sum()
        sel.backward()

        # modality attribution via grad norms at pre-fusion
        z_mri = feats["z_mri"]; z_ehr = feats["z_ehr"]
        g_mri = feats["g_mri"]; g_ehr = feats["g_ehr"]
        imp_mri = float(g_mri.norm(p=2).item()) if g_mri is not None else float("nan")
        imp_ehr = float(g_ehr.norm(p=2).item()) if g_ehr is not None else float("nan")

        # (optional) feature-level attribution for EHR: grad * input at the encoder input layer
        # Here we approximate by grad at z_ehr projected to input scale; for a quick proxy,
        # compute absolute gradient of the loss w.r.t. the encoder input using torch.autograd.grad
        # (Skip heavy exact IG for speed)
        feat_imp = None
        try:
            # one more backward pass for input grads (retain graph to avoid re-forward)
            model.zero_grad(set_to_none=True)
            logits = model(v.expand(-1,3,-1,-1,-1), e.requires_grad_(True))
            sel = logits.gather(1, c.view(-1,1)).sum()
            sel.backward()
            gi = e.grad.detach().abs().cpu().numpy()[0]  # (F,)
            feat_imp = gi
        except Exception:
            pass

        row = {
            "PTID": str(r["PTID"]),
            "y": int(y.item()),
            "yhat": int(logits.detach().argmax(1).item()),
            "imp_mri_gradnorm": imp_mri,
            "imp_ehr_gradnorm": imp_ehr
        }
        rows.append(row)

        if feat_imp is not None:
            # append per-feature importances for this sample to a separate file (wide)
            # to keep file size reasonable, only save for first 200 samples
            if i < 200:
                pd.DataFrame([feat_imp], columns=feat_cols, index=[str(r["PTID"])]).to_csv(
                    out_dir/"ehr_feature_grad_abs_sampled.csv",
                    mode="a", header=not (out_dir/"ehr_feature_grad_abs_sampled.csv").exists()
                )

    # Save modality attribution
    pd.DataFrame(rows).to_csv(out_dir/"modality_gradnorm_test.csv", index=False)

    # Simple aggregate: average normalized importance across test
    df_imp = pd.DataFrame(rows)
    s = df_imp[["imp_mri_gradnorm","imp_ehr_gradnorm"]].replace([np.inf,-np.inf], np.nan).dropna()
    if len(s) > 0:
        m = s.mean()
        tot = m["imp_mri_gradnorm"] + m["imp_ehr_gradnorm"]
        agg = {
            "mean_imp_mri": float(m["imp_mri_gradnorm"]),
            "mean_imp_ehr": float(m["imp_ehr_gradnorm"]),
            "mean_share_mri": float(m["imp_mri_gradnorm"]/tot) if tot>0 else float("nan"),
            "mean_share_ehr": float(m["imp_ehr_gradnorm"]/tot) if tot>0 else float("nan"),
            "n_samples": int(len(s))
        }
        (out_dir/"modality_gradnorm_summary.json").write_text(json.dumps(agg, indent=2))

    # cleanup hooks
    h1.remove(); h2.remove()
    print(f"[OK] saved to {out_dir}")
    
if __name__ == "__main__":
    main()
