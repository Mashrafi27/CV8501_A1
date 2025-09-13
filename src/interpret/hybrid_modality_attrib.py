#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, yaml, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------- datasets (unchanged) ----------
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
    # DEFAULTS NOW MATCH YOUR CLASS
    ap.add_argument("--mri_hook", default="mri_backbone", help="module path yielding z_mri")
    ap.add_argument("--ehr_hook", default="ehr_enc", help="module path yielding z_ehr")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--use_true_class", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- build aligned test dataframe ----------
    ehr = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")
    feat_cols = [c for c in ehr.columns if c not in {"PTID","DIAGNOSIS"}]

    # enforce numeric & float32 eagerly on EHR (pre-merge)
    ehr[feat_cols] = (
        ehr[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype("float32")
    )

    std_idx = pd.read_csv(Path(cfg["interim_root"]) / "mri_std" / "index.csv")
    df_all  = std_idx.merge(ehr, on="PTID", how="inner").dropna(subset=["DIAGNOSIS"]).reset_index(drop=True)
    _, _, te_ids = read_ids(cfg["splits_root"])
    df_te = df_all[df_all["PTID"].isin(te_ids)].reset_index(drop=True)

    # belt & suspenders: re-enforce numeric AFTER merge/filter to avoid numpy.object_
    df_te[feat_cols] = (
        df_te[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype("float32")
    )

    # quick guard: fail fast if any non-numeric slipped through
    bad = [c for c in feat_cols if str(df_te[c].dtype) == "object"]
    assert not bad, f"Non-numeric EHR feature columns after casting: {bad}"

    # ---------- import and load your HybridModel ----------
    from src.models.hybrid_model import HybridModel
    model = HybridModel(n_ehr_features=len(feat_cols), pretrained_mri=False, dropout=0.2).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state); model.eval()

    # ---------- register hooks on chosen modules ----------
    feats = {"z_mri": None, "z_ehr": None, "g_mri": None, "g_ehr": None}

    def make_fwd_hook(key_z, key_g):
        def _hook(module, inp, out):
            feats[key_z] = out
            def _bwd(grad):
                feats[key_g] = grad
            out.register_hook(_bwd)
        return _hook

    mri_mod = get_module_by_path(model, args.mri_hook)   # -> mri_backbone
    ehr_mod = get_module_by_path(model, args.ehr_hook)   # -> ehr_enc

    h_mri = mri_mod.register_forward_hook(make_fwd_hook("z_mri","g_mri"))
    h_ehr = ehr_mod.register_forward_hook(make_fwd_hook("z_ehr","g_ehr"))

    out_dir = Path(cfg["processed_root"]).parents[0] / "Interpretability" / "Hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- prebuild EHR & labels tensors to avoid per-iter dtype issues ----------
    X_ehr = torch.from_numpy(df_te[feat_cols].to_numpy(dtype=np.float32))  # (N, D)
    y_all = torch.tensor(df_te["DIAGNOSIS"].astype(int).values - 1, dtype=torch.long)

    rows = []
    for i in range(len(df_te)):
        r = df_te.iloc[i]

        # MRI
        vol = np.load(r["std_path"]).astype(np.float32)               # (D,H,W)
        v = torch.from_numpy(vol[None, None, ...]).float()            # (1,1,D,H,W)
        v = torch.nn.functional.interpolate(v, size=(64,64,64), mode="trilinear", align_corners=False)[:, :, :, :, :]
        v = v.to(device)

        # EHR & y
        e = X_ehr[i:i+1].to(device)                                   # (1, D)
        y = y_all[i:i+1].to(device)                                   # (1,)

        # forward (expand MRI to 3 channels if backbone expects RGB-like input)
        logits = model(v.expand(-1, 3, -1, -1, -1), e)
        c = y if args.use_true_class else logits.argmax(1)

        # backprop selected logit
        model.zero_grad(set_to_none=True)
        sel = logits.gather(1, c.view(-1,1)).sum()
        sel.backward()

        # modality attribution via gradient norms on pre-fusion embeddings
        g_mri = feats["g_mri"]; g_ehr = feats["g_ehr"]
        imp_mri = float(g_mri.norm(p=2).item()) if g_mri is not None else float("nan")
        imp_ehr = float(g_ehr.norm(p=2).item()) if g_ehr is not None else float("nan")

        rows.append({
            "PTID": str(r["PTID"]),
            "y": int(y.item()),
            "yhat": int(logits.detach().argmax(1).item()),
            "imp_mri_gradnorm": imp_mri,
            "imp_ehr_gradnorm": imp_ehr
        })

    pd.DataFrame(rows).to_csv(out_dir/"modality_gradnorm_test.csv", index=False)

    # aggregate summary
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

    h_mri.remove(); h_ehr.remove()
    print(f"[OK] saved to {out_dir}")

if __name__ == "__main__":
    main()
