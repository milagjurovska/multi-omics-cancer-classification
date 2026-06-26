# Multi-Omics Cancer Classification Using Foundation Models

Classifying 6 cancer types from patient biology data using pretrained AI models (BulkFormer + CpGPT) and comparing fusion strategies — a university research project using TCGA data.

**Team:** Mila Todorovska · Bojana Andonova · Sandra Zivanovska  
**University:** Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University · 2024

---

## What This Project Does

We take 800 cancer patients, each with two types of biological measurements:
- **RNA-seq** — which genes are active in the patient's cells
- **DNA Methylation** — chemical markers that control gene activity

Instead of feeding raw biological data (60,000+ features) directly into a classifier, we use two **pretrained AI transformers** to compress this data into compact, meaningful representations called **embeddings**. We then train a classifier on these embeddings and test 5 different ways of combining both data sources.

**The 6 cancer types:** Breast (BRCA), Lung (LUAD), Colon (COAD), Kidney (KIRC), Liver (LIHC), Thyroid (THCA)

---

## Results

| Method | Classifier | Accuracy | AUC-ROC |
|---|---|---|---|
| **Early Fusion** | **SVM** | **99.0%** | **0.998** |
| Late Fusion (learned) | LogReg | 99.0% | 0.998 |
| Early Fusion | LogReg | 98.9% | 0.998 |
| BulkFormer only | SVM | 98.8% | 0.998 |
| BulkFormer only | LogReg | 98.5% | 0.998 |
| CpGPT only | SVM | 87.8% | 0.981 |
| CpGPT only | XGBoost | 80.1% | 0.962 |

> Evaluated with nested 5-fold cross-validation (5 outer × 3 inner folds). Statistical significance tested with Wilcoxon and McNemar tests. Full results in `results/biomni/`.

**Key finding:** BulkFormer compresses RNA data **94×** (60,616 genes → 643 numbers) and still achieves 98.5% accuracy. Logistic Regression matches XGBoost — proving the foundation model embeddings are linearly separable.

---

## Pipeline Overview

```
TCGA Data (800 patients)
        │
        ├── RNA-seq ──────► BulkFormer ──► 643-dim embeddings
        │
        └── Methylation ──► CpGPT ────────► 128-dim embeddings
                                │
                                └── Autoencoder ──► 64-dim embeddings
                                        │
                    ┌───────────────────────────────────┐
                    │        Fusion Strategies           │
                    │  Early: concat (771-dim)           │
                    │  Late: two classifiers vote        │
                    │  Deep: autoencoder latent (64-dim) │
                    └───────────────────────────────────┘
                                        │
                              XGBoost / SVM / LogReg
                                        │
                         6-class cancer type prediction
```

---

## Foundation Models

### BulkFormer (RNA-seq)
A transformer pretrained on ~500k bulk RNA-seq samples. Rank-orders genes by expression level per patient and outputs a 643-dimensional embedding capturing gene interaction patterns.

> Download model files from [Zenodo doi:10.5281/zenodo.15744294](https://doi.org/10.5281/zenodo.15744294) and place in `BulkFormer/data/`. Run `python scripts/setup_bulkformer.py` to set up.

### CpGPT (DNA Methylation)
A transformer pretrained on large-scale methylation array data. Takes CpG probe beta values (0–1) as input and outputs a 128-dimensional embedding per patient.

> Run `notebooks/colab/cpgpt_colab.ipynb` on Google Colab (GPU required).

### Cross-Modal Autoencoder
A neural network (PyTorch) that takes both BulkFormer + CpGPT embeddings as input and compresses them into a shared 64-dimensional latent space. Trained with reconstruction loss on both modalities.

---

## Project Structure

```
multi-omics-cancer-classification/
│
├── data/
│   ├── manifests/
│   │   └── matched_samples.csv        ← 800 patients with cancer type labels
│   └── processed/
│       ├── bulkformer_embeddings.npy  ← (800, 643) RNA embeddings
│       ├── cpgpt_embeddings.npy       ← (800, 128) methylation embeddings
│       └── autoencoder_embeddings.npy ← (800, 64) joint embeddings
│
├── scripts/
│   ├── train_classifier.py            ← main classification pipeline
│   ├── combine_results.py             ← merges all results into one table
│   └── preprocess_methylation.py      ← methylation preprocessing
│
├── notebooks/
│   ├── biomni_fusion_pipeline.ipynb   ← rigorous evaluation (nested CV + stats)
│   └── colab/
│       ├── cpgpt_colab.ipynb          ← CpGPT embedding extraction (Colab)
│       └── preprocess_colab.ipynb     ← data preprocessing (Colab)
│
└── results/
    ├── biomni/                        ← full rigorous results (use these)
    │   ├── results_summary.csv        ← mean±std per method/classifier
    │   ├── per_fold_metrics.csv       ← per fold breakdown
    │   ├── per_class_metrics.csv      ← F1/precision/recall per cancer type
    │   ├── wilcoxon_pairwise_f1.csv   ← statistical significance tests
    │   ├── mcnemar_pairwise.csv       ← pairwise McNemar tests
    │   ├── metrics_barplot.png        ← bar chart all methods
    │   ├── confusion_matrices.png     ← all 15 confusion matrices
    │   └── shap_modality_importance.png ← BulkFormer vs CpGPT SHAP
    ├── mofa/                          ← MOFA+ classical baseline results
    └── kegg_top_pathways.png          ← KEGG pathway enrichment
```

---

## How to Run

### Requirements
```bash
pip install numpy pandas scikit-learn xgboost shap umap-learn torch scipy statsmodels matplotlib seaborn jupyter
```

### Quick run (our pipeline)
```bash
python scripts/train_classifier.py
```
Runs 5 fusion strategies with XGBoost, 5-fold CV. Completes in ~5 minutes.

### Rigorous run (Biomni pipeline — recommended)
```bash
jupyter notebook notebooks/biomni_fusion_pipeline.ipynb
```
Runs 5 fusion strategies × 3 classifiers with nested CV and statistical tests. Takes ~40 minutes.

### Prerequisites
Embeddings must already be generated and placed in `data/processed/`. Embedding extraction requires Google Colab (GPU).

---

## Methods

| Component | Choice | Why |
|---|---|---|
| Foundation model (RNA) | BulkFormer | Pretrained on bulk RNA-seq — compatible with TCGA data |
| Foundation model (Methylation) | CpGPT | Compatible with Illumina 450K array format |
| Classifier | XGBoost + SVM + LogReg | SVM/LogReg best for linearly separable embeddings |
| Cross-validation | Nested 5×3-fold | Prevents data leakage from hyperparameter tuning |
| Statistical tests | Wilcoxon + McNemar | Required when methods differ by <1% |
| Baseline | MOFA+ (30 factors) | Classical multi-omics method for comparison |

---

## Classical Baseline — MOFA+

MOFA+ (Multi-Omics Factor Analysis) was run by Mila as a classical comparison. It learns shared and modality-specific factors from both omics without using foundation models. Results in `results/mofa/`.

## Biological Interpretation — KEGG Pathways

Top 500 most discriminative CpG probes (by ANOVA F-statistic) were mapped to genes and submitted to KEGG enrichment analysis. Key enriched pathways: complement/coagulation cascades (LIHC-specific), thyroid hormone synthesis (THCA-specific), metabolic reprogramming.

---

## Dependencies

```
numpy · pandas · scikit-learn · xgboost · shap · umap-learn
torch · scipy · statsmodels · matplotlib · seaborn · anndata · jupyter
```
