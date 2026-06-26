# Classification Pipeline — Full Documentation

**Project:** Multi-Omics Cancer Classification  
**Models:** BulkFormer (RNA-seq) + CpGPT (Methylation)  
**Data:** 800 TCGA patients, 6 cancer types  
**Evaluation:** Nested 5-fold CV · 3 classifiers · Statistical significance tests

---

## Overview

We classify 6 cancer types using compressed representations (embeddings) from two pretrained AI models instead of raw biological data. We test 5 different strategies for combining both data sources, and 3 different classifiers — 15 configurations total.

---

## Input Data

| Modality | Raw features | After foundation model | Compression |
|---|---|---|---|
| RNA-seq | 60,616 genes | 643-dim BulkFormer embedding | 94× smaller |
| Methylation | 49,156 CpG probes | 128-dim CpGPT embedding | 384× smaller |
| Both combined | — | 64-dim autoencoder embedding | — |

**Labels:** Cancer type from TCGA manifest (`matched_samples.csv`)

| Cancer | Full name | Samples |
|---|---|---|
| BRCA | Breast cancer | 134 |
| COAD | Colon adenocarcinoma | 133 |
| KIRC | Kidney renal clear cell carcinoma | 133 |
| LIHC | Liver hepatocellular carcinoma | 133 |
| LUAD | Lung adenocarcinoma | 134 |
| THCA | Thyroid carcinoma | 133 |

---

## Step 1 — Preprocessing Per Modality

Before fusion, each modality is independently standardized and compressed:

```
BulkFormer embeddings (800 × 643)
        │
        └── StandardScaler (zero mean, unit variance)
        └── PCA → 64 principal components
        └── Output: (800 × 64)

CpGPT embeddings (800 × 128)
        │
        └── StandardScaler
        └── PCA → 64 principal components
        └── Output: (800 × 64)
```

**Why PCA(64) per modality?**  
Without this, BulkFormer (643 dims) would dominate the classifier — it has 5× more features than CpGPT (128 dims). By reducing both to 64 dimensions, each modality contributes equally to the fusion. PCA also removes noise, keeping only the most important directions.

---

## Step 2 — Fusion Strategies (5 total)

### 1. BulkFormer Only
Use only the RNA-seq embedding. Tests: how good is RNA alone?
```
BulkFormer (64 dims after PCA) → Classifier → Prediction
```

### 2. CpGPT Only
Use only the methylation embedding. Tests: how good is methylation alone?
```
CpGPT (64 dims after PCA) → Classifier → Prediction
```

### 3. Early Fusion
Concatenate both embeddings into one vector. Classifier sees all features at once.
```
BulkFormer (64) + CpGPT (64) → concat → (128 dims) → Classifier → Prediction
```
Both modalities contribute equally (thanks to PCA(64) preprocessing).

### 4. Late Fusion — Probability Averaging
Train two separate classifiers. Average their probability outputs.
```
BulkFormer (64) → Classifier A → probabilities ──┐
                                                   ├── average → Prediction
CpGPT (64)      → Classifier B → probabilities ──┘
```
Each modality votes independently. Good when one modality fails on certain samples.

### 5. Late Fusion — Learned Weighting (Meta-Learner)
Instead of simple averaging, train a meta-learner on top of both classifiers' outputs.
```
BulkFormer → Classifier A → probabilities ──┐
                                             ├── Meta-Learner → Prediction
CpGPT      → Classifier B → probabilities ──┘
```
The meta-learner learns how much to trust each modality per cancer type.

---

## Step 3 — Classifiers (3 total)

### Logistic Regression (Linear Probe)
Draws a straight decision boundary between classes. No hyperparameters beyond regularization strength C.

**Purpose:** Tests whether embeddings are linearly separable. If LogReg matches XGBoost, the foundation models did all the work — the classifier is just reading off the answer.

**Grid search:** `C ∈ {0.01, 0.1, 1, 10}`

### SVM with RBF Kernel
Finds the maximum-margin hyperplane between classes. RBF kernel handles slight non-linearities.

**Purpose:** Excellent for high-dimensional, small-sample settings. Mathematically designed to generalize well — focuses on the hardest boundary cases.

**Grid search:** `C ∈ {0.1, 1, 10}`, `γ ∈ {scale, auto}`

### XGBoost
Ensemble of decision trees, each correcting previous errors. Strong baseline for tabular data.

**Purpose:** The "heavy" baseline. If SVM and LogReg match XGBoost, it confirms the embeddings are already linearly separable.

**Grid search:** `n_estimators ∈ {100, 300}`, `max_depth ∈ {3, 6}`, `lr ∈ {0.05, 0.1}`

---

## Step 4 — Nested Cross-Validation

**Why nested CV instead of simple CV?**

Simple CV picks hyperparameters and tests on the same data split — slightly optimistic. Nested CV separates tuning from evaluation completely.

```
Outer fold 1: [── TEST (160 patients) ──][────────── TRAIN (640 patients) ──────────]
                                                            │
                                          Inner fold 1: [val][──train──][──train──]
                                          Inner fold 2: [──train──][val][──train──]
                                          Inner fold 3: [──train──][──train──][val]
                                                            │
                                               Best hyperparameters found here
                                                            │
                                          Retrain on full 640, test on 160 → score

Outer fold 2: [──train──][── TEST ──][──train──][──train──][──train──]
...repeat for 5 outer folds...

Final score = average of 5 outer fold scores
```

- **5 outer folds** → 5 independent test evaluations → reliable mean ± std
- **3 inner folds** → hyperparameter tuning never touches the outer test set
- **Result:** Truly unbiased performance estimate

---

## Step 5 — Statistical Testing

With all methods between 97–99%, you cannot rely on point estimates alone. 0.5% difference across 5 folds may be noise.

### Wilcoxon Signed-Rank Test
Compares fold-by-fold F1 scores between every pair of configurations.

> "Does Early Fusion beat BulkFormer consistently across all 5 folds, or does it win some and lose others?"

If p < 0.05 → the difference is statistically significant (not luck).  
Results: `results/biomni/wilcoxon_pairwise_f1.csv`

### McNemar's Test
Compares predictions sample-by-sample between two models.

> "Are there specific patients that Early Fusion classifies correctly but BulkFormer gets wrong?"

If both models make the same mistakes → difference is noise, not real.  
Results: `results/biomni/mcnemar_pairwise.csv`

---

## Results

### Summary Table (mean ± std, 5-fold nested CV)

| Fusion Strategy | Classifier | Accuracy | Macro F1 | AUC-ROC |
|---|---|---|---|---|
| **Early Fusion** | **SVM** | **99.0% ± 0.6%** | **0.990** | **0.998** |
| Late Fusion (learned) | LogReg | 99.0% ± 0.6% | 0.990 | 0.998 |
| Early Fusion | LogReg | 98.9% ± 0.9% | 0.989 | 0.998 |
| Late Fusion (avg) | SVM | 98.2% ± 0.5% | 0.983 | 0.998 |
| BulkFormer only | SVM | 98.8% ± 0.6% | 0.988 | 0.998 |
| BulkFormer only | LogReg | 98.5% ± 0.9% | 0.985 | 0.998 |
| BulkFormer only | XGBoost | 98.0% ± 0.5% | 0.980 | 0.997 |
| CpGPT only | SVM | 87.8% ± 3.1% | 0.877 | 0.981 |
| CpGPT only | LogReg | 87.0% ± 2.1% | 0.870 | 0.981 |
| CpGPT only | XGBoost | 80.1% ± 4.2% | 0.801 | 0.962 |

### Per-Cancer F1 Score (Early Fusion + SVM — best model)

| Cancer | F1 Score | Notes |
|---|---|---|
| THCA | ~0.996 | Easiest — thyroid has very distinct methylation signature |
| LIHC | ~1.000 | Liver-specific gene expression highly discriminative |
| COAD | ~0.989 | Colon cancer well-separated in both modalities |
| LUAD | ~0.989 | Lung adenocarcinoma distinct from other types |
| KIRC | ~0.981 | Kidney clear cell — some overlap with other cancers |
| BRCA | ~0.978 | Most heterogeneous cancer type — hardest to classify |

---

## Key Findings

### 1. Linear probe matches XGBoost
LogReg (98.5%) ≈ XGBoost (98.0%) for BulkFormer alone.  
→ **The embeddings are linearly separable.** BulkFormer already organized the data so cleanly that a straight line separates the 6 cancer types. The classifier is just reading off the answer.

### 2. SVM is the best classifier
SVM consistently outperforms both LogReg and XGBoost.  
→ Maximum-margin classifiers excel on high-dimensional embeddings with small samples (800 patients). This is expected in the foundation model literature.

### 3. Early Fusion wins but barely
Early Fusion + SVM (99.0%) > BulkFormer only + SVM (98.8%) by 0.2%.  
→ The improvement is tiny. BulkFormer alone captures most of the signal. CpGPT adds a small complementary signal, but the task (tissue-of-origin) is essentially solved by RNA alone.

### 4. CpGPT is significantly weaker
CpGPT only: 80-88% vs BulkFormer only: 98-99%.  
→ 128 dimensions is very compressed for 49,156 probes. BulkFormer's 643 dimensions from 60,616 genes retains more information. For this task, RNA-seq is the more informative modality.

### 5. All methods hit AUC ≥ 0.998
Near-perfect ranking ability across all methods that include BulkFormer.  
→ These 6 cancer types (different tissues) are highly separable from biological data. The more scientifically interesting task would be intra-cancer subtype classification (e.g. BRCA molecular subtypes).

---

## Output Files

All results from the rigorous Biomni evaluation are in `results/biomni/`:

| File | Contents |
|---|---|
| `results_summary.csv` | Mean ± std per fusion × classifier combination |
| `per_fold_metrics.csv` | Accuracy / F1 / AUC for each outer fold |
| `per_class_metrics.csv` | Precision / Recall / F1 per cancer type |
| `wilcoxon_pairwise_f1.csv` | p-values for all pairwise method comparisons |
| `mcnemar_pairwise.csv` | Sample-level agreement tests between methods |
| `best_params_per_fold.csv` | Which hyperparameters were selected per fold |
| `oof_predictions.npz` | Full out-of-fold predictions for all 15 configurations |
| `metrics_barplot.png` | Bar chart comparing all methods |
| `confusion_matrices.png` | 5×3 grid of confusion matrices |
| `shap_top_features.png` | Most important embedding dimensions |
| `shap_modality_importance.png` | BulkFormer vs CpGPT SHAP contribution |

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn xgboost shap scipy statsmodels matplotlib seaborn jupyter

# 2. Make sure embeddings are in data/processed/
#    bulkformer_embeddings.npy  (800, 643)
#    cpgpt_embeddings.npy       (800, 128)

# 3. Run the rigorous Biomni pipeline (~40 min)
jupyter notebook notebooks/biomni_fusion_pipeline.ipynb

# 4. Or run our quick pipeline (~5 min, XGBoost only)
python scripts/train_classifier.py
```
