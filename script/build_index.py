#!/usr/bin/env python3
"""
Builds a master index linking subject folders (MRI) to Excel metadata.
Output: work/ADNI/Interim/master_index.csv
"""

from pathlib import Path
import argparse, yaml, pandas as pd

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path="config/adni.yaml"):
    cfg = load_cfg(cfg_path)

    images_root = Path(cfg["images_root"])
    excel_path  = Path(cfg["excel_path"])
    interim     = Path(cfg["interim_root"]); interim.mkdir(parents=True, exist_ok=True)

    # Load Excel file
    df = pd.read_excel(excel_path)
    if "PTID" not in df.columns or "DIAGNOSIS" not in df.columns:
        raise SystemExit("❌ Excel must contain PTID and DIAGNOSIS columns.")

    # Normalize PTID (string, strip spaces)
    df["PTID"] = df["PTID"].astype(str).str.strip()

    # Optional: map diagnosis int → string
    label_map = {int(k): v for k, v in cfg.get("labels", {}).get("map_int_to_str", {}).items()}
    if label_map:
        df["DX_STR"] = df["DIAGNOSIS"].map(label_map)

    rows = []
    for subj_dir in sorted(images_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj = subj_dir.name.strip()
        nii_files = list(subj_dir.glob("*.nii.gz"))
        if len(nii_files) == 0:
            print(f"[WARN] no .nii.gz for {subj}")
            continue

        # In your case: exactly one file per folder
        nii_path = str(nii_files[0].resolve())

        # Match Excel
        hit = df[df["PTID"] == subj]
        if len(hit) == 0:
            print(f"[WARN] {subj} not found in Excel")
            diag, dxs = None, None
        else:
            diag = int(hit["DIAGNOSIS"].iloc[0]) if pd.notna(hit["DIAGNOSIS"].iloc[0]) else None
            dxs  = hit["DX_STR"].iloc[0] if "DX_STR" in hit.columns else None

        rows.append({
            "subject_id": subj,
            "mri_path": nii_path,
            "diagnosis": diag,
            "dx_str": dxs
        })

    out_csv = interim / "master_index.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} with {len(rows)} subjects")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/adni.yaml")
    args = ap.parse_args()
    main(args.cfg)
