#!/usr/bin/env python3
"""
Grad-CAM for 3D ResNet-18:
- Loads r3d_18 checkpoint and test set volumes (64^3 expected)
- Captures activations & gradients from the last conv block
- Saves heatmaps overlaid on three orthogonal planes for N examples per class

Usage:
  python -m src.interpret.mri_gradcam3d --cfg config/adni.yaml \
         --model work/ADNI/Models/mri_only_r3d18/model.pt --per_class 3
"""

import argparse
from pathlib import Path
import numpy as np, pandas as pd, yaml, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18, R3D_18_Weights
import matplotlib.pyplot as plt

# ---- dataset ----
class MRI3DNPY(Dataset):
    def __init__(self, df, target_shape=(64,64,64)):
        self.df = df.reset_index(drop=True); self.tgt = target_shape
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
    with open(p,"r") as f: return yaml.safe_load(f)

def read_ids(splits_root):
    R = Path(splits_root)
    rd = lambda n: [s for s in (R/n).read_text().splitlines() if s]
    return set(rd("train_ids.txt")), set(rd("val_ids.txt")), set(rd("test_ids.txt"))

def register_cam_hooks(model, layer):
    feats = {"acts": None, "grads": None}
    def fwd_hook(m, inp, out): feats["acts"] = out.detach()
    def bwd_hook(m, grad_in, grad_out): feats["grads"] = grad_out[0].detach()
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    return feats, (h1, h2)

def make_cam(acts, grads):
    # acts, grads: (N, C, T, H, W) or (N, C, D, H, W) – for r3d it's (N,C,T,H,W)
    # We treat T as depth (D)
    weights = grads.mean(dim=(2,3,4), keepdim=True)          # global-average over spatial
    cam = (weights * acts).sum(dim=1)                        # (N, T, H, W)
    cam = torch.relu(cam)
    # normalize each sample to [0,1]
    cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1,1,1,1)
    cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1,1,1,1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam  # (N, D, H, W)

def save_triplet_overlay(vol, cam, out_png, title=""):
    """
    vol: (D,H,W) numpy, cam: (D,H,W) numpy in [0,1]
    Saves mid-slices overlay in axial (z), coronal (y), sagittal (x)
    """
    D,H,W = vol.shape
    z, y, x = D//2, H//2, W//2
    planes = [
        ("Axial (Z)", vol[z], cam[z]),            # (H,W)
        ("Coronal (Y)", vol[:, y, :], cam[:, y, :]),  # (D,W)
        ("Sagittal (X)", vol[:, :, x], cam[:, :, x]), # (D,H)
    ]
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    for ax, (name, img, hm) in zip(axs, planes):
        ax.imshow(img, cmap="gray")
        ax.imshow(hm, alpha=0.45)   # default colormap
        ax.set_title(name); ax.axis("off")
    if title: fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(out_png, dpi=180); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--per_class", type=int, default=3, help="examples per class")
    ap.add_argument("--pretrained_flag", action="store_true", help="ignored; kept for API symmetry")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    std_idx = pd.read_csv(Path(cfg["interim_root"]) / "mri_std" / "index.csv")
    labels  = pd.read_parquet(Path(cfg["processed_root"]) / "tabular_feats.parquet")[["PTID","DIAGNOSIS"]]
    df = std_idx.merge(labels, on="PTID", how="inner")
    df = df[df["DIAGNOSIS"].notna()].reset_index(drop=True)

    _, _, te_ids = read_ids(cfg["splits_root"])
    df_te = df[df["PTID"].isin(te_ids)].reset_index(drop=True)

    dl = DataLoader(MRI3DNPY(df_te), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = r3d_18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 3)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state); model = model.to(device).eval()

    # choose last conv layer block for CAM
    target_layer = model.layer4[-1].conv2
    feats, hooks = register_cam_hooks(model, target_layer)

    out_dir = Path(cfg["processed_root"]).parents[0] / "Interpretability" / "MRI_GradCAM"
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect per-class examples
    saved_count = {0:0,1:0,2:0}
    for v, y, pid in dl:
        x = v.to(device).expand(-1,3,-1,-1,-1)
        x.requires_grad_(True)
        logits = model(x)
        pred = logits.argmax(1)
        # Backprop CAM for the predicted class (or true y if you prefer)
        chosen = pred  # or y.to(device)
        sel = logits.gather(1, chosen.unsqueeze(1)).sum()
        model.zero_grad(set_to_none=True)
        sel.backward()

        cam = make_cam(feats["acts"], feats["grads"]).detach().cpu().numpy()  # (N,D,H,W)
        vol = v[0,0].numpy()
        cam0 = cam[0]

        cls = int(y[0].item())
        if saved_count[cls] < args.per_class:
            save_triplet_overlay(vol, cam0, out_dir / f"{pid[0]}_cls{cls}_cam.png",
                                 title=f"PTID {pid[0]} | true {cls} | pred {int(pred[0])}")
            saved_count[cls] += 1

        if all(saved_count[c] >= args.per_class for c in saved_count):
            break

    for h in hooks: h.remove()
    print(f"[OK] saved examples to {out_dir} (per_class={args.per_class})")

if __name__ == "__main__":
    main()
