#!/usr/bin/env python3
"""
MRI preprocessing (no masking):
- Read subject paths from work/ADNI/Interim/master_index.csv
- Resample each NIfTI to isotropic spacing (default: 2.0 mm) with BSpline
- Intensity normalize globally: percentile clip [0.5, 99.5] then z-score over the whole volume
- Save as .npy under work/ADNI/Interim/mri_std/{PTID}.npy
- Write an index CSV: work/ADNI/Interim/mri_std/index.csv
"""

from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

def load_cfg(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def resample_isotropic(img: sitk.Image, target_spacing=(2.0, 2.0, 2.0)) -> sitk.Image:
    """Resample to isotropic spacing using BSpline (good for MRI)."""
    img = sitk.Cast(img, sitk.sitkFloat32)
    in_spacing = np.array(list(img.GetSpacing()), dtype=float)   # (sx, sy, sz)
    in_size    = np.array(list(img.GetSize()), dtype=float)      # (nx, ny, nz)
    tgt        = np.array(list(target_spacing), dtype=float)

    out_size = np.round(in_size * (in_spacing / tgt)).astype(int).tolist()
    out_size = [int(max(1, s)) for s in out_size]

    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(tuple(tgt))
    res.SetSize(out_size)
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetTransform(sitk.Transform())
    res.SetDefaultPixelValue(0.0)            # numeric fill
    res.SetInterpolator(sitk.sitkBSpline)    # or sitk.sitkLinear
    return res.Execute(img)

def clip_and_znorm_global(vol: np.ndarray) -> np.ndarray:
    """Clip to [0.5, 99.5] percentiles and z-score over the whole volume (no masking)."""
    vol = np.asarray(vol, dtype=np.float32)
    v = vol.ravel()
    # guard against NaNs/Infs
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros_like(vol, dtype=np.float32)

    lo, hi = np.percentile(v, [0.5, 99.5])
    vol = np.clip(vol, lo, hi)

    m = float(vol.mean())
    s = float(vol.std())
    if s > 1e-9:
        vol = (vol - m) / s
    else:
        vol = vol - m
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml", help="Path to config")
    ap.add_argument("--spacing", type=float, default=2.0, help="Target isotropic spacing in mm")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N subjects (debug)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    master_csv = Path(cfg["interim_root"]) / "master_index.csv"
    if not master_csv.exists():
        print(f"ERROR: {master_csv} not found. Build index first.", file=sys.stderr)
        sys.exit(1)

    out_root = ensure_dir(Path(cfg["interim_root"]) / "mri_std")
    df = pd.read_csv(master_csv)
    n_total = len(df)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    rows = []
    for i, r in df.iterrows():
        sid = str(r["subject_id"])
        p = str(r["mri_path"])
        try:
            img = sitk.ReadImage(p)
            img_iso = resample_isotropic(img, target_spacing=(args.spacing, args.spacing, args.spacing))
            vol = sitk.GetArrayFromImage(img_iso).astype(np.float32)  # (z,y,x)
            vol_std = clip_and_znorm_global(vol)

            outp = out_root / f"{sid}.npy"
            np.save(outp, vol_std)

            rows.append({
                "PTID": sid,
                "std_path": str(outp.resolve()),
                "out_shape_zyx": json.dumps(list(vol_std.shape)),
                "out_spacing_xyz_mm": json.dumps([args.spacing, args.spacing, args.spacing]),
            })

            if (i+1) % 50 == 0:
                print(f"[{i+1}/{len(df)}] {sid} ✓")

        except Exception as e:
            print(f"[WARN] {sid}: {e}")

    idx_out = out_root / "index.csv"
    pd.DataFrame(rows).to_csv(idx_out, index=False)
    print(f"[OK] standardized {len(rows)}/{n_total} MRIs → {out_root}")
    print(f"[OK] wrote index: {idx_out}")

if __name__ == "__main__":
    main()
