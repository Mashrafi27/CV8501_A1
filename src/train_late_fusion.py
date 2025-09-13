#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import yaml

from src.data.mm_hybrid_dataset import MMHybridDataset
from src.models.late_fusion_model import LateFusionModel

def load_cfg(p): 
    with open(p,"r") as f: return yaml.safe_load(f)

def read_ids(splits_root):
    root = Path(splits_root)
    def _r(fname): return [s for s in (root/fname).read_text().splitlines() if s]
    return set(_r("train_ids.txt")), set(_r("val_ids.txt")), set(_r("test_ids.txt"))

def build_mm_dataframe_and_ehr_cols(cfg, ids):
    std_idx = pd.read_csv(Path(cfg["interim_root"]) / "mri_std" / "index.csv")
    ehr     = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")
    ehr_cols = [c for c in ehr.columns if c not in {"PTID","DIAGNOSIS"}]
    df = std_idx.merge(ehr, on="PTID", how="inner")
    df = df[df["DIAGNOSIS"].notna()]
    df = df[df["PTID"].isin(ids)].reset_index(drop=True)
    df.loc[:, ehr_cols] = df.loc[:, ehr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    return df, ehr_cols

@torch.no_grad()
def eval_mode(model, loader, device, mode):
    model.eval()
    P, Y = [], []
    for v3d, ehr, y in loader:
        v = v3d.to(device).expand(-1,3,-1,-1,-1)
        e = ehr.to(device); y = y.to(device)
        if mode == "mri":
            logits = model.forward_branch(x_mri=v, mode="mri")
            probs  = torch.softmax(logits, dim=1)
        elif mode == "ehr":
            logits = model.forward_branch(x_ehr=e, mode="ehr")
            probs  = torch.softmax(logits, dim=1)
        else:
            lm, le = model.forward_branch(x_mri=v, x_ehr=e, mode="fusion")
            probs  = model.fuse_logits(lm, le).exp()
        P.append(probs.cpu().numpy()); Y.append(y.cpu().numpy())
    P = np.concatenate(P); Y = np.concatenate(Y)
    yhat = P.argmax(1)
    m = {"macro_f1": f1_score(Y, yhat, average="macro"),
         "balanced_acc": balanced_accuracy_score(Y, yhat)}
    try: m["roc_auc_ovo"] = roc_auc_score(Y, P, multi_class="ovo")
    except Exception: m["roc_auc_ovo"] = float("nan")
    return m, P, Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.5, help="MRI probability weight in fusion")
    ap.add_argument("--pretrained_mri", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} | alpha={args.alpha}")

    tr_ids, va_ids, te_ids = read_ids(cfg["splits_root"])
    df_tr, ehr_cols = build_mm_dataframe_and_ehr_cols(cfg, tr_ids)
    df_va, _        = build_mm_dataframe_and_ehr_cols(cfg, va_ids)
    df_te, _        = build_mm_dataframe_and_ehr_cols(cfg, te_ids)

    dl_tr = DataLoader(MMHybridDataset(df_tr, ehr_cols), batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(MMHybridDataset(df_va, ehr_cols), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(MMHybridDataset(df_te, ehr_cols), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = LateFusionModel(n_ehr_features=len(ehr_cols),
                            n_classes=3, dropout=args.dropout,
                            alpha=args.alpha, pretrained_mri=args.pretrained_mri).to(device)
    print(f"[INFO] EHR encoder: {model.ehr_kind}")

    # class-weighted loss; fused uses NLL over log-probs; branches use CE for eval only
    y_tr = df_tr["DIAGNOSIS"].astype(int).values - 1
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    w = counts.sum() / (counts + 1e-9); w /= w.mean()
    w_t = torch.tensor(w, dtype=torch.float32, device=device)
    nll = nn.NLLLoss(weight=w_t)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"f1": -1, "state": None, "epoch": -1}
    patience, bad = 8, 0

    for ep in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(dl_tr, desc=f"late epoch {ep}/{args.epochs}", leave=False)
        for v3d, ehr, y in pbar:
            v = v3d.to(device).expand(-1,3,-1,-1,-1)
            e = ehr.to(device); y = y.to(device)
            lm, le = model.forward_branch(x_mri=v, x_ehr=e, mode="fusion")
            logp = model.fuse_logits(lm, le)
            loss = nll(logp, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss)
            pbar.set_postfix(loss=f"{loss_sum/max(1,pbar.n):.4f}")

        m_mri, _, _ = eval_mode(model, dl_va, device, "mri")
        m_ehr, _, _ = eval_mode(model, dl_va, device, "ehr")
        m_fus, _, _ = eval_mode(model, dl_va, device, "fusion")
        print(f"[{ep:03d}] loss={loss_sum/len(dl_tr):.4f} | val F1 (mri={m_mri['macro_f1']:.3f}, ehr={m_ehr['macro_f1']:.3f}, fused={m_fus['macro_f1']:.3f})")

        if m_fus["macro_f1"] > best["f1"]:
            best.update(f1=m_fus["macro_f1"], state=model.state_dict(), epoch=ep); bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"Early stop at {ep}, best {best['epoch']} (val fused F1={best['f1']:.4f})")
            break

    # test with best state
    model.load_state_dict(best["state"])
    out_dir = Path(cfg["processed_root"]).parents[0] / "Models" / "late_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir/"model.pt")

    m_mri, Pm, Y = eval_mode(model, dl_te, device, "mri")
    m_ehr, Pe, _ = eval_mode(model, dl_te, device, "ehr")
    m_fus, Pf, _ = eval_mode(model, dl_te, device, "fusion")
    (out_dir/"metrics.json").write_text(json.dumps({
        "val_best_fused_f1": best["f1"], "best_epoch": best["epoch"],
        "alpha": args.alpha,
        "test": {"mri": m_mri, "ehr": m_ehr, "fused": m_fus}
    }, indent=2))
    np.save(out_dir/"test_probs_mri.npy", Pm)
    np.save(out_dir/"test_probs_ehr.npy", Pe)
    np.save(out_dir/"test_probs_fused.npy", Pf)
    print(f"[OK] saved to {out_dir}")

if __name__ == "__main__":
    main()
