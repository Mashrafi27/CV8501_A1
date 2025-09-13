# ADNI Multimodal (MRI + EHR)

End-to-end pipelines for AD classification (CN/MCI/AD) on ADNI using:

* **Unimodal baselines:** MRI-only (3D ResNet-18), EHR-only (MLP)
* **Hybrid fusion:** feature-level concatenation of MRI+EHR embeddings
* **Late fusion:** decision-level weighted averaging of unimodal softmaxes

> **Data location (read-only on HPC):**
> `/apps/local/shared/CV8501/assignment1/data/ADNI/`
> (contains `Raw/Images/*/*.nii.gz` and `ADNI_Subset_Diagnosis_Filtered.xlsx`)

> **Project workspace (writeable):**
> `work/ADNI/Interim` and `work/ADNI/Processed` (created automatically)

---

## 0) Quickstart

```bash
# Clone (HPC and/or local)
git clone git@github.com:Mashrafi27/CV8501_A1.git
cd CV8501_A1

# Create env
conda create -n cv8501-mm python=3.10 -y
conda activate cv8501-mm
pip install -r requirements.txt    # (torch/torchvision must match your CUDA)

# Optional: login to Weights & Biases (for charts)
wandb login
```

---

## 1) Configuration

Edit `config/adni.yaml` to match your machine:

```yaml
# config/adni.yaml
hpc_adni_root: "/apps/local/shared/CV8501/assignment1/data/ADNI"
work_root:      "work/ADNI"
interim_root:   "work/ADNI/Interim"
processed_root: "work/ADNI/Processed"
splits_root:    "work/ADNI/Splits"

# training defaults
seed: 42
mri_target_spacing_mm: 2.0
mri_target_shape: [64, 64, 64]    # D,H,W
num_classes: 3
```

> 💡 All internal artifacts (indexes, npy volumes, parquet features, models) are written under `work/ADNI/*`.
> You **do not** move or copy the raw ADNI data into the repo. Just adjust the ADNI dataset location in the config file.

---

## 2) Build indexes & preprocess

### 2.1 Master index (PTID ↔ MRI path ↔ label)

```bash
python -m src.preprocess.build_index --cfg config/adni.yaml
# writes: work/ADNI/Interim/master_index.csv
```

### 2.2 EHR preprocessing (keep all columns unless >40% missing)

```bash
python -m src.preprocess.tabular_preproc --cfg config/adni.yaml
# writes: work/ADNI/Processed/tabular_feats.parquet
```

### 2.3 MRI standardization (resample+normalize+downsample)

```bash
python -m src.preprocess.mri_resample_norm --cfg config/adni.yaml --spacing 2.0
# writes: work/ADNI/Interim/mri_std/{PTID}.npy and index.csv
```

### 2.4 Create splits (stratified by diagnosis)

```bash
python -m src.preprocess.make_splits --cfg config/adni.yaml --seed 42 --val 0.15 --test 0.15
# writes: work/ADNI/Splits/{train_ids.txt,val_ids.txt,test_ids.txt}
```

---

## 3) Training


### 3.1 Unimodal baselines

**EHR-only (MLP)**

```bash
python -m src.train_ehr --cfg config/adni.yaml --epochs 100 --batch 256 --lr 1e-3 --dropout 0.2
# saves: work/ADNI/Models/ehr_only_mlp/{model.pt, metrics.json, *probs.npy}
```

**MRI-only (3D ResNet-18)**

```bash
python -m src.train_mri --cfg config/adni.yaml --epochs 50 --batch 2 --lr 1e-4 --dropout 0.2 --pretrained_mri
# saves: work/ADNI/Models/mri_only_r3d18/{model.pt, metrics.json, *probs.npy}
```

### 3.2 Hybrid Fusion (feature-level)

```bash
python -m src.train_hybrid_fusion --cfg config/adni.yaml --epochs 50 --batch 2 --lr 1e-4 --dropout 0.2 --pretrained_mri
# saves: work/ADNI/Models/hybrid/{model.pt, metrics.json}
```

### 3.3 Late Fusion (decision-level)

1. Train the two unimodals above.
2. Fuse on validation to tune α; evaluate on test:

```bash
python -m src.train_late_fusion --cfg config/adni.yaml --epochs 50 --batch 2 --lr 1e-4 --dropout 0.2 --alpha 0.5
# reads probs from unimodal folders; saves metrics → work/ADNI/Models/late_fusion/metrics.json
```

---

## 4) Evaluation & Metrics

All trainers log:

* **Macro-F1 (primary)**, **Balanced Accuracy**, **ROC-AUC (OVO)**
* **Confusion matrices** at validation/test
* `metrics.json` and `{val,test}_{probs,labels,ids}.npy`

W\&B (optional): metrics/curves are logged automatically if you’re logged in.

---

## 5) Interpretability

### 5.1 Late Fusion (modality contribution)

```bash
python -m src.interpret.late_fusion_contrib \
  --cfg config/adni.yaml \
  --ehr_model work/ADNI/Models/ehr_only_mlp/model.pt \
  --mri_model work/ADNI/Models/mri_only_r3d18/model.pt \
  --alpha 0.5
# → work/ADNI/Interpretability/LateFusion/{per_sample_contrib.csv, ablation_metrics.json}
```

### 5.2 Hybrid (modality attribution at fusion)

```bash
python -m src.interpret.hybrid_modality_attrib \
  --cfg config/adni.yaml \
  --model work/ADNI/Models/hybrid/model.pt \
  --mri_hook mri_backbone --ehr_hook ehr_enc
# → work/ADNI/Interpretability/Hybrid/{modality_gradnorm_test.csv, modality_gradnorm_summary.json}
```

---


## 6) Tips & Troubleshooting

* **Path moves / different users:**
  If you move the repo, ensure `work/ADNI/Interim/mri_std/index.csv` points to valid files.
  Prefer **relative paths** in index (we already reconstruct from config at runtime).
  Quick fix: `sed -i 's|OLD_BASE|NEW_BASE|g' work/ADNI/Interim/mri_std/index.csv`.

* **CUDA OOM (MRI):**
  Use `--batch 1 or 2`, keep input at `64³`, freeze BN (`torch.set_grad_enabled` for head only), or reduce LR.

* **Dataloader workers:**
  If warned that workers > cores, set `num_workers=<max_recommended_in_warning>`.

* **W\&B offline:**
  `wandb offline` to avoid internet, logs still saved locally.

---

## 7) Reproducibility

* Fixed `seed=42` (configurable).
* All artifacts saved under `work/ADNI/*`.
* Checkpoints saved in `work/ADNI/Models/*`.

---

## 8) Citation

ADNI: [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)

---

**License:** for academic coursework.
