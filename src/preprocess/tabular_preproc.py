#!/usr/bin/env python3
from pathlib import Path
import yaml, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

CFG = "config/adni.yaml"

def load_cfg(p):
    with open(p,"r") as f: return yaml.safe_load(f)

def main():
    cfg = load_cfg(CFG)
    df  = pd.read_excel(cfg["excel_path"]).copy()

    assert "PTID" in df.columns and "DIAGNOSIS" in df.columns, "Excel must have PTID and DIAGNOSIS"
    df["PTID"] = df["PTID"].astype(str).str.strip()

    # 1. Drop ID-like columns that are not features
    drop_cols = {"RID","SITEID","VISCODE"}  # /TODO: expand if you see other ID cols
    df = df[[c for c in df.columns if c not in drop_cols]]

    # 2. Remove columns with too many NAs (e.g., > 40%)
    na_thresh = 0.4
    keep_cols = [c for c in df.columns if df[c].isna().mean() <= na_thresh]

    df = df[keep_cols]

    # 3. Identify numeric vs categorical
    num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()

    # Remove PTID + DIAGNOSIS from feature sets
    num_cols = [c for c in num_cols if c not in ["DIAGNOSIS"]]
    cat_cols = [c for c in cat_cols if c not in ["PTID"]]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["tabular"]["impute_numeric"])),
        ("scaler", StandardScaler() if cfg["tabular"]["standardize"] else "passthrough")
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg["tabular"]["impute_categorical"])),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    X = pre.fit_transform(df)

    # feature names
    feat_names = []
    feat_names += num_cols
    if cat_cols:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        feat_names += list(ohe.get_feature_names_out(cat_cols))

    arr = X.toarray() if hasattr(X, "toarray") else X
    out = pd.DataFrame(arr, columns=feat_names)
    out.insert(0, "PTID", df["PTID"].values)
    out.insert(1, "DIAGNOSIS", df["DIAGNOSIS"].values)

    out_path = Path(cfg["processed_root"]) / "tabular_feats.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[OK] wrote {out_path} | shape={out.shape}")
    print(f" kept {len(num_cols)} numeric + {len(cat_cols)} categorical columns (after dropping >{na_thresh*100:.0f}% NA)")

if __name__ == "__main__":
    main()
