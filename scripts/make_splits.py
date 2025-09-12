#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    root = Path("work/ADNI")
    master = pd.read_csv(root / "Interim" / "master_index.csv")
    df = master.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int) - 1   # {0,1,2}

    train_ids, test_ids = train_test_split(
        df["subject_id"], stratify=df["diagnosis"], test_size=0.2, random_state=42
    )
    train_ids, val_ids = train_test_split(
        train_ids, stratify=df.loc[df["subject_id"].isin(train_ids), "diagnosis"],
        test_size=0.2, random_state=42
    )

    split_dir = root / "Splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_ids.txt").write_text("\n".join(train_ids))
    (split_dir / "val_ids.txt").write_text("\n".join(val_ids))
    (split_dir / "test_ids.txt").write_text("\n".join(test_ids))

    print(f"[OK] Saved splits to {split_dir}")

if __name__ == "__main__":
    main()
