#!/usr/bin/env python3
"""
MRI preprocessing for ADNI:
- Reads paths from work/ADNI/Interim/master_index.csv (subject_id, mri_path)
- Resamples each NIfTI to isotropic spacing (default: 2.0 mm) with BSpline
- Builds a robust brain mask (Otsu with fallback)
- Clips intensities to [0.5, 99.5] percentiles within mask
- Z-scores within mask; sets background to 0
- Saves standardized volumes as .npy under work/ADNI/Interim/mri_std/{PTID}.npy
- Writes an index CSV with PTID, std_path, out_shape, out_spacing
"""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
import sys

def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def resample_isotropic(img: sitk.Image, target_spacing=(2.0, 2.0, 2.0)) -> sitk.Image:
    img = sitk.Cast(img, sitk.sitkFloat32)
    in_spacing = np.array(list(img.GetSpacing()), dtype=float)
    in_size    = np.array(list(img.GetSize()), dtype=float)
    tgt        = np.array(list(target_spacing), dtype=float)

    # compute output size: new_size = old_size * (in_spacing / target_spacing)
    out_size = np.round(in_size * (in_spacing / tgt)).astype(int).tolist()
    # guard against zeros
    out_size = [int(max(1, s)) for s in out_size]

    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(tuple(tgt))
    res.SetSize(out_size)
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetTransform(sitk.Transform())
    res.SetDefaultPixelValue(0.0)
    res.SetInterpolator(sitk.sitkBSpline)  # linear or BSpline are fine for MRI
    return res.Execute(img)

def build_mask(img_iso: sitk.Image) -> np.ndarray:
    """Otsu mask with a little smoothing; fallback to percentile if tiny."""
    sm = sitk.SmoothingRecursiveGaussian(img_iso, sigma=1.0)
    mask_img = sitk.OtsuThreshold(sm, 0.0, 1.0)  # returns {0,1}
    mask = sitk.GetArrayFromImage(mask_img).astype(bool)

    # Fallback if mask is too small (e.g., Otsu failed)
    if mask.sum() < 1000:
        vol = sitk.GetArrayFromImage(img_iso).astype(np.float32)
        thr = np.percentile(vol, 60.0)
        mask = vol > thr
        if mask.sum() < 1000:
            mask = vol > np.percentile(vol, 30.0)
    return mask

def clip_and_znorm(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = vol[mask]
    if vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(vals, [0.5, 99.5])
    vol = vol.copy()
    vol[mask] = np.clip(vol[mask], lo, hi)
    vals = vol[mask]
    mu, sd = float(vals.mean()), float(vals.std())
    if sd > 1e-9:
        vol[mask] = (vals - mu) / sd
    else:
        vol[mask] = vals - mu
    vol[~mask] = 0.0
    return vol.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml", help="Path to config")
    ap.add_argument("--spacing", type=float, default=2.0, help="Target isotropic spacing in mm")
    ap.add_argument("--out-kind", choices=["npy"], default="npy", help="Output format")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N subjects (debug)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    master_csv = Path(cfg["interim_root"]) / "master_index.csv"
    if not master_csv.exists():
        print(f"ERROR: {master_csv} not found. Build index first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(master_csv)
    out_root = ensure_dir(Path(cfg["interim_root"]) / "mri_std")
    rows = []
    n_total = len(df)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    for i, r in df.iterrows():
        sid = str(r["subject_id"])
        p = str(r["mri_path"])
        try:
            img = sitk.ReadImage(p)
            img_iso = resample_isotropic(img, target_spacing=(args.spacing, args.spacing, args.spacing))
            vol = sitk.GetArrayFromImage(img_iso).astype(np.float32)  # (z,y,x)
            mask = build_mask(img_iso)
            vol_std = clip_and_znorm(vol, mask)

            outp = out_root / f"{sid}.npy"
            np.save(outp, vol_std)

            rows.append({
                "PTID": sid,
                "std_path": str(outp.resolve()),
                "out_shape_zyx": json.dumps(list(vol_std.shape)),
                "out_spacing_xyz_mm": json.dumps([args.spacing, args.spacing, args.spacing])
            })
            if (i+1) % 50 == 0:
                print(f"[{i+1}/{n_total}] {sid} ✓")
        except Exception as e:
            print(f"[WARN] {sid}: {e}")

    idx_out = out_root / "index.csv"
    pd.DataFrame(rows).to_csv(idx_out, index=False)
    print(f"[OK] standardized {len(rows)}/{n_total} MRIs → {out_root}")
    print(f"[OK] wrote index: {idx_out}")

if __name__ == "__main__":
    main()
