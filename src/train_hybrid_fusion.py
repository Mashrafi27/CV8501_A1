#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
import yaml
from tqdm import tqdm

from src.data.mm_hybrid_dataset import MMHybridDataset
from src.models.hybrid_model import HybridModel
from src.utils.logging_utils import try_wandb_init, wandb_log, plot_conf_mat

def load_cfg(p):
    with open(p,"r") as f: return yaml.safe_load(f)

def read_ids(root):
    def _r(name): return set((Path(root)/name).read_text().splitlines())
    return _r("train_ids.txt"), _r("val_ids.txt"), _r("test_ids.txt")

def build_mm_dataframe_and_ehr_cols(cfg, ids):
    from pathlib import Path
    import pandas as pd

    std_idx = pd.read_csv(Path(cfg["interim_root"]) / "mri_std" / "index.csv")         # PTID, std_path, out_shape_zyx, out_spacing_xyz_mm
    ehr     = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")   # PTID, DIAGNOSIS, <EHR features>

    # EHR feature columns must come ONLY from the parquet (not from std_idx)
    ehr_cols = [c for c in ehr.columns if c not in {"PTID", "DIAGNOSIS"}]

    df = std_idx.merge(ehr, on="PTID", how="inner")
    df = df[df["DIAGNOSIS"].notna()]
    df = df[df["PTID"].isin(ids)].reset_index(drop=True)

    # Coerce just the EHR columns to numeric float32 (robust to bools/ints); NaNs -> 0
    df.loc[:, ehr_cols] = df.loc[:, ehr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    return df, ehr_cols


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_all, p_all = [], []
    for v3d, ehr, y in loader:
        v = v3d.to(device).expand(-1,3,-1,-1,-1)
        ehr = ehr.to(device); y = y.to(device)
        probs = torch.softmax(model(v, ehr), dim=1).cpu().numpy()
        p_all.append(probs); y_all.append(y.cpu().numpy())
    y = np.concatenate(y_all); p = np.concatenate(p_all)
    yhat = p.argmax(1)
    metrics = {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "balanced_acc": balanced_accuracy_score(y, yhat)
    }
    try:
        metrics["roc_auc_ovo"] = roc_auc_score(y, p, multi_class="ovo")
    except Exception:
        metrics["roc_auc_ovo"] = float("nan")
    return metrics, y, yhat, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pretrained_mri", action="store_true")
    ap.add_argument("--project", default="cv8501-adni")
    ap.add_argument("--run_name", default="hybrid")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tr_ids, va_ids, te_ids = read_ids(cfg["splits_root"])
    df_tr, ehr_cols = build_mm_dataframe_and_ehr_cols(cfg, tr_ids)
    df_va, _        = build_mm_dataframe_and_ehr_cols(cfg, va_ids)
    df_te, _        = build_mm_dataframe_and_ehr_cols(cfg, te_ids)


    dl_tr = DataLoader(MMHybridDataset(df_tr, ehr_cols), batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(MMHybridDataset(df_va, ehr_cols), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(MMHybridDataset(df_te, ehr_cols), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = HybridModel(len(ehr_cols), pretrained_mri=args.pretrained_mri, dropout=args.dropout).to(device)

    # class-weighted loss
    y_tr = df_tr["DIAGNOSIS"].astype(int).values - 1
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    w = counts.sum() / (counts + 1e-9); w /= w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # wandb (optional)
    run = try_wandb_init(
        project=args.project,
        run_name=args.run_name,
        config={"epochs":args.epochs, "batch":args.batch, "lr":args.lr, "dropout":args.dropout,
                "pretrained_mri":bool(args.pretrained_mri), "ehr_dim":len(ehr_cols)}
    )
    if run is not None:
        try:
            run.watch(model, log="all", log_freq=50)
        except Exception as e:
            print(f"[wandb] watch failed: {e}")

    # training
    best = {"f1": -1, "state": None, "epoch": -1}
    patience, bad = 8, 0

    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(dl_tr, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for v3d, ehr, y in pbar:
            v = v3d.to(device).expand(-1,3,-1,-1,-1)
            ehr = ehr.to(device); y = y.to(device)
            logits = model(v, ehr)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += float(loss)
            pbar.set_postfix(loss=f"{loss_sum/max(1,pbar.n):.4f}")

        val_metrics, _, _, _ = evaluate(model, dl_va, device)
        print(f"[{epoch:03d}] train_loss={loss_sum/len(dl_tr):.4f} | val_f1={val_metrics['macro_f1']:.4f} bal={val_metrics['balanced_acc']:.4f}")
        wandb_log(run, {"train/loss": loss_sum/len(dl_tr), "val/macro_f1": val_metrics["macro_f1"],
                        "val/balanced_acc": val_metrics["balanced_acc"], "val/roc_auc_ovo": val_metrics["roc_auc_ovo"],
                        "epoch": epoch}, step=epoch)

        if val_metrics["macro_f1"] > best["f1"]:
            best.update(f1=val_metrics["macro_f1"], state=model.state_dict(), epoch=epoch)
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"Early stop at {epoch}, best epoch {best['epoch']} (val F1={best['f1']:.4f})")
            break

    # test
    model.load_state_dict(best["state"])
    test_metrics, y_true, y_pred, p_test = evaluate(model, dl_te, device)
    print("[TEST]", test_metrics)

    # save artifacts
    out_dir = Path(cfg["processed_root"]).parents[0] / "Models" / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps({
        "val_best_f1": best["f1"], "best_epoch": best["epoch"], "test": test_metrics
    }, indent=2))

    # confusion matrix fig
    try:
        import matplotlib.pyplot as plt
        fig, cm = plot_conf_mat(y_true, y_pred, labels=("CN","MCI","AD"))
        fig.savefig(out_dir / "cm_test.png", bbox_inches="tight")
        plt.close(fig)
        wandb_log(run, {"test/macro_f1": test_metrics["macro_f1"],
                        "test/balanced_acc": test_metrics["balanced_acc"],
                        "test/roc_auc_ovo": test_metrics["roc_auc_ovo"]})
        if run is not None:
            run.log({"cm_test": wandb.Image(str(out_dir / "cm_test.png"))})
    except Exception as e:
        print(f"[warn] could not save/log confusion matrix: {e}")

    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()
