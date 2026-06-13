# Project Plan — Multi-Omics Cancer Classification
## 3-Person Collaboration Guide

| Person | Name | Role |
|--------|------|------|
| Person 1 | Bojana | Data download & preprocessing |
| Person 2 | Sandra | MethylGPT embeddings |
| Person 3 | Aleksandar | Geneformer embeddings + Classifier |

---

## Overview

We are building a system that classifies cancer types using two types of biological data:
- **DNA Methylation** → processed by MethylGPT (a foundation model)
- **RNA-seq (gene expression)** → processed by Geneformer (a foundation model)

Both models convert raw biological data into **embeddings** (numerical vectors that capture biological meaning). We then combine these embeddings and train a classifier to predict cancer type.

**6 cancer types:** BRCA (breast), LUAD (lung), COAD (colon), KIRC (kidney), LIHC (liver), THCA (thyroid)

**Goal: ~600 samples** (100 per cancer type), expanded from the current 30-sample test subset.

---

## Dependencies Between People

```
Person 1 (Data)
    ↓ provides processed data
Person 2 (MethylGPT) ──┐
Person 3 (Geneformer) ──┤ both provide embeddings
                        ↓
              All 3 together → Classifier + Evaluation
```

Person 2 and Person 3 **cannot run the full pipeline** until Person 1 finishes downloading.
But both can prepare and test on the existing 30 samples while waiting.

---

---

# PERSON 1 — Data Download & Preprocessing

## Your Role
You are responsible for getting the full dataset ready and making it available to the whole team. Everything depends on you finishing first.

## What You Are Doing and Why
Right now the project only has 5 samples per cancer type (30 total) — this is a test subset. 30 samples is far too few to train a meaningful model. You need to download 100 samples per cancer type (600 total) from the TCGA database via the GDC API.

---

## Step 1 — Change the Download Limit

Open the file `scripts/download_tcga_data.py` and find line 11:

```python
LIMIT_PER_PROJECT = 5  # Target roughly 3000 total (SET TO 5 FOR TEST)
```

Change it to:

```python
LIMIT_PER_PROJECT = 100
```

This tells the script to download 100 samples per cancer type instead of 5.

---

## Step 2 — Run the Download Script

Make sure you are inside the project folder, then run:

```bash
cd multi-omics-cancer-classification
python scripts/download_tcga_data.py
```

**What this does:**
- Queries the GDC (Genomic Data Commons) API to find patients who have BOTH methylation AND RNA-seq data available
- Downloads the matched files for each patient
- Creates a new manifest file at `data/manifests/matched_samples.csv` listing all 600 samples

**How long it takes:** 1-3 hours depending on your internet speed (each sample is ~16MB, 600 samples = ~9.6GB total)

**Important:** Do not close the terminal while it is running. If it gets interrupted, just run it again — the script skips files that are already downloaded.

---

## Step 3 — Preprocess the Methylation Data

```bash
python scripts/preprocess_methylation.py
```

**What this does:**
- Reads all 600 methylation files
- Filters to keep only the CpG probes that MethylGPT was trained on (49,156 probes)
- Combines everything into one file: `data/processed/tcga_methylation.h5ad`
- This is the input file that Person 2 needs

**How long it takes:** ~10-20 minutes

---

## Step 4 — Preprocess the RNA-seq Data

```bash
python scripts/preprocess_geneformer.py
```

**What this does:**
- Reads all 600 RNA-seq files
- Filters genes, removes version numbers from gene IDs, aggregates duplicates
- Combines everything into one file: `data/processed/tcga_rna_seq.h5ad`
- This is the input file that Person 3 needs

**How long it takes:** ~5-10 minutes

---

## Step 5 — Upload to Google Drive (Shared Folder)

Create a shared Google Drive folder with the team. Upload these files:

```
data/manifests/matched_samples.csv
data/processed/tcga_methylation.h5ad       ← Person 2 needs this
data/processed/tcga_rna_seq.h5ad           ← Person 3 needs this
data/processed/probe_ids_type3.csv         ← Person 2 needs this
data/processed/token_dictionary.pkl        ← Person 3 needs this
data/processed/gene_median_dictionary.pkl  ← Person 3 needs this
```

Also upload the `pretrained_models/methylgpt-medium/` folder — Person 2 needs the model weights.

---

## While You Wait for the Download

Person 2 and 3 need you to finish before they can run the full pipeline. While downloading:
- Help Person 3 write the classifier code (it can be written and tested on 30 samples)
- Make sure the shared Google Drive folder is set up and shared with both teammates

---

## Your Deliverables
- [ ] `data/processed/tcga_methylation.h5ad` — 600 samples × 49,156 CpG probes
- [ ] `data/processed/tcga_rna_seq.h5ad` — 600 samples × ~60,000 genes
- [ ] `data/manifests/matched_samples.csv` — 600 rows
- [ ] All files uploaded to shared Google Drive

---

---

# PERSON 2 — MethylGPT Embeddings

## Your Role
You take the preprocessed methylation data and run it through the MethylGPT foundation model to extract embeddings. These embeddings are rich numerical representations of each patient's epigenetic profile.

## What You Are Doing and Why
Raw methylation data has 49,156 values per patient — one beta value per CpG probe. Instead of using these raw values directly (which would be too sparse and high-dimensional), we pass them through MethylGPT, a model pretrained on large methylation datasets. MethylGPT compresses each patient's profile into a dense vector that captures biological patterns. This is called **transfer learning**.

---

## While Waiting for Person 1

Set up your Colab environment and test it on the existing 30 samples.

### Step 1 — Open Google Colab

Go to **colab.research.google.com** → New Notebook

### Step 2 — Enable GPU

**Runtime → Change runtime type → T4 GPU → Save**

### Step 3 — Install Dependencies

In the first cell, run:

```python
!pip install anndata tqdm
!pip install scgpt
!git clone https://github.com/yoyolicoris/MethylGPT.git
%cd MethylGPT
!pip install -e .
%cd ..
```

### Step 4 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 5 — Test on 30 Samples

Copy `extract_methylation_embeddings.py` into a Colab cell and update the paths to point to your Drive:

```python
INPUT_H5AD = "/content/drive/MyDrive/shared_folder/data/processed/tcga_methylation.h5ad"
MODEL_DIR = "/content/drive/MyDrive/shared_folder/pretrained_models/methylgpt-medium"
CPG_LIST_PATH = "/content/drive/MyDrive/shared_folder/data/processed/probe_ids_type3.csv"
OUTPUT_EMBEDDINGS = "/content/drive/MyDrive/shared_folder/data/processed/methylgpt_embeddings.npy"
```

Run it on the 30 samples to confirm it works.

---

## Once Person 1 Uploads the 600-Sample Data

### Step 6 — Run on Full 600 Samples

The paths are already set up. Just run the full script — it will automatically use the new 600-sample file.

**What this does:**
- Loads `tcga_methylation.h5ad` (600 samples)
- Tokenizes each sample's CpG probe values
- Passes them through MethylGPT
- Extracts the `[CLS]` token embedding for each sample — a single vector summarizing the whole methylation profile
- Saves all 600 embeddings as `methylgpt_embeddings.npy`

**How long it takes:** ~1.5-2 hours on T4 GPU (49K probes per sample is a very long sequence)

**Expected output shape:** `(600, embedding_dim)` — one vector per patient

### Step 7 — Verify the Output

```python
import numpy as np
emb = np.load("/content/drive/MyDrive/shared_folder/data/processed/methylgpt_embeddings.npy")
print(f"Shape: {emb.shape}")         # should be (600, some_dim)
print(f"Any NaN: {np.isnan(emb).any()}")  # should be False
```

### Step 8 — Save to Shared Drive

The output path already points to Drive, so it saves automatically. Notify the team that `methylgpt_embeddings.npy` is ready.

---

## Your Deliverables
- [ ] `data/processed/methylgpt_embeddings.npy` — shape (600, embedding_dim), saved to shared Drive

---

---

# PERSON 3 — Geneformer Embeddings + Classifier

## Your Role
You have two tasks:
1. Regenerate the RNA-seq processed file and extract Geneformer embeddings
2. Build the full classifier pipeline (can start immediately on 30 samples)

## What You Are Doing and Why
Gene expression data (RNA-seq) tells us how active each gene is in a tumor. Geneformer is a BERT-based model pretrained on 30 million single-cell transcriptomes. It understands gene interactions and dependencies that simple normalization misses. Like Person 2, you extract embeddings — one vector per patient summarizing their gene expression profile.

The classifier takes both embedding types and learns to distinguish cancer types.

---

## Task A — Geneformer Embeddings

### While Waiting for Person 1

The file `data/processed/tcga_rna_seq.h5ad` is currently missing (it was deleted or never saved). But the raw files for 30 samples are still there. Regenerate it:

```bash
cd multi-omics-cancer-classification
python scripts/preprocess_geneformer.py
```

This creates `tcga_rna_seq.h5ad` from the existing 30 raw files.

Then test the Geneformer embedding extraction on 30 samples:

```bash
python scripts/extract_geneformer_embeddings.py
```

This will overwrite the existing `geneformer_embeddings.npy` with a fresh version. Confirm it works.

### Once Person 1 Uploads the 600-Sample Data

Set up Colab the same way as Person 2 (GPU T4, mount Drive), then run `extract_geneformer_embeddings.py` with updated paths:

```python
INPUT_H5AD = "/content/drive/MyDrive/shared_folder/data/processed/tcga_rna_seq.h5ad"
TOKEN_DICT_PATH = "/content/drive/MyDrive/shared_folder/data/processed/token_dictionary.pkl"
OUTPUT_EMBEDDINGS = "/content/drive/MyDrive/shared_folder/data/processed/geneformer_embeddings.npy"
```

**How long it takes:** ~30-45 minutes on T4 GPU

**Expected output shape:** `(600, 1152)` — Geneformer produces 1152-dimensional embeddings

---

## Task B — Classifier (Start This Now)

You can build and test the classifier immediately using the existing 30 samples and the already-extracted Geneformer embeddings. When the 600-sample embeddings are ready, just re-run with the new files.

### Step 1 — Create the Classifier Script

Create `scripts/train_classifier.py`:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────
GENEFORMER_EMB = "data/processed/geneformer_embeddings.npy"
METHYLGPT_EMB  = "data/processed/methylgpt_embeddings.npy"   # add when ready
MANIFEST       = "data/manifests/matched_samples.csv"

# ── Load data ──────────────────────────────────────────────────────────────
manifest = pd.read_csv(MANIFEST)
labels   = manifest["project"].values

gf_emb = np.load(GENEFORMER_EMB)   # (N, 1152)

# Comment this out until MethylGPT embeddings are ready:
# mg_emb = np.load(METHYLGPT_EMB)
# X = np.concatenate([gf_emb, mg_emb], axis=1)  # Early Fusion
X = gf_emb  # Geneformer only for now

le = LabelEncoder()
y  = le.fit_transform(labels)
print(f"Classes: {le.classes_}")
print(f"Feature matrix shape: {X.shape}")

# ── K-Fold Cross Validation ────────────────────────────────────────────────
n_splits = 5
skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

acc_scores, f1_scores, auc_scores = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average="macro")

    y_val_bin = label_binarize(y_val, classes=list(range(len(le.classes_))))
    auc = roc_auc_score(y_val_bin, y_proba, multi_class="ovr", average="macro")

    acc_scores.append(acc)
    f1_scores.append(f1)
    auc_scores.append(auc)
    print(f"Fold {fold+1}: Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

print(f"\n── Final Results ({n_splits}-Fold CV) ──")
print(f"Accuracy : {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
print(f"F1 Score : {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
print(f"AUC-ROC  : {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
```

### Step 2 — Run on 30 Samples (Geneformer Only)

```bash
pip install xgboost scikit-learn
python scripts/train_classifier.py
```

Results on 30 samples won't be meaningful (too few), but this confirms the pipeline works.

### Step 3 — Add Early Fusion (Once MethylGPT Embeddings Arrive)

When Person 2 delivers `methylgpt_embeddings.npy`, uncomment these lines in the classifier:

```python
mg_emb = np.load(METHYLGPT_EMB)
X = np.concatenate([gf_emb, mg_emb], axis=1)  # Early Fusion
```

This concatenates both embedding vectors per patient and re-runs the classifier. Compare results to Geneformer-only.

---

## Your Deliverables
- [ ] `data/processed/geneformer_embeddings.npy` — shape (600, 1152), saved to shared Drive
- [ ] `scripts/train_classifier.py` — working classifier with k-fold CV
- [ ] Results table: Accuracy / F1 / AUC for Geneformer-only and Early Fusion

---

---

# All 3 Together — Final Evaluation

Once both embeddings are ready and the classifier works:

1. **Geneformer only** — baseline, already done by Person 3
2. **MethylGPT only** — swap `X = mg_emb`, re-run
3. **Early Fusion** — concatenate both, re-run
4. **Compare results** in a table:

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Geneformer only | ? | ? | ? |
| MethylGPT only | ? | ? | ? |
| Early Fusion (both) | ? | ? | ? |

The goal is to show that **Early Fusion outperforms either modality alone** — this validates the multi-omics approach.

---

# Quick Reference — File Locations

| File | Created by | Used by |
|------|-----------|---------|
| `data/manifests/matched_samples.csv` | Person 1 | Everyone |
| `data/processed/tcga_methylation.h5ad` | Person 1 | Person 2 |
| `data/processed/tcga_rna_seq.h5ad` | Person 1 | Person 3 |
| `data/processed/methylgpt_embeddings.npy` | Person 2 | Person 3 (classifier) |
| `data/processed/geneformer_embeddings.npy` | Person 3 | Person 3 (classifier) |
| `data/processed/probe_ids_type3.csv` | Already exists | Person 2 |
| `data/processed/token_dictionary.pkl` | Already exists | Person 3 |
| `pretrained_models/methylgpt-medium/` | Already exists | Person 2 |
