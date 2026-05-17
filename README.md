# Multi-Omics Cancer Classification using Foundation Models

This project implements a multi-omics approach to cancer classification, leveraging transfer learning from foundation models specialized in biological data. We integrate DNA Methylation and RNA-seq data from TCGA (The Cancer Genome Atlas) to improve diagnostic accuracy across multiple cancer types.

## Motivation
Cancer is a complex disease requiring analysis across multiple molecular levels. Foundation models like **MethylGPT** (or CpGPT) for methylation and **Geneformer** for transcriptomics offer rich representations that capture intricate biological dependencies better than traditional methods.

## Methodology
The project follows a multi-phase approach:
1.  **Parallel Preprocessing**:
    *   **RNA-seq**: Processed for Geneformer to capture gene interactions.
    *   **DNA Methylation**: Processed for MethylGPT to extract epigenetic embeddings.
2.  **Fusion Strategies**:
    *   **Early Fusion**: Concatenating embeddings before classification.
    *   **Late Fusion**: Ensemble of specialized models for each modality.
3.  **Classification**: Final layer utilizing Neural Networks or XGBoost with k-fold cross-validation.

## Data Acquisition
We target 5-6 primary cancer types from TCGA:
*   Breast (TCGA-BRCA)
*   Lung (TCGA-LUAD)
*   Colon (TCGA-COAD)
*   Kidney (TCGA-KIRC)
*   Liver (TCGA-LIHC)
*   Thyroid (TCGA-THCA)

Data is downloaded using the GDC API, specifically filtering for patients with matched RNA-seq and Methylation 450k data.

## Initial Setup

### Prerequisites
*   Python 3.8+
*   Disk Space: ~50-60GB for full dataset (currently using subset for testing)

### Installation
```bash
pip install -r requirements.txt
```

### External Foundation Models
This project uses **MethylGPT** as an external foundation model for DNA methylation embeddings. MethylGPT is developed by other researchers and is not included in this repository.

Recommended install from PyPI:
```bash
pip install methylgpt
```

For pinned dependency versions or development installs, use the upstream repository:
```bash
git clone https://github.com/albert-ying/MethylGPT.git
cd MethylGPT
pip install -r requirements.txt
pip install -e .
```

Alternatively, create the conda environment from the upstream repository:
```bash
git clone https://github.com/albert-ying/MethylGPT.git
cd MethylGPT
conda env create -f environment.yml
conda activate methylgpt
pip install -e .
```

If the upstream repository is cloned inside this project as `MethylGPT_repo/`, that folder is ignored by Git so its source code and model files are not committed here.

### Data Download
To download the initial test subset:
```bash
python scripts/download_tcga_data.py
```

## Classification Pipeline

After preprocessing and embedding extraction, the modeling pipeline trains cancer-type classifiers from the foundation-model embeddings.

Expected inputs:
* `data/processed/geneformer_embeddings.npy` - RNA-seq embeddings from Geneformer.
* DNA methylation embeddings from MethylGPT or CpGPT. The pipeline will automatically use the first file it finds from:
  * `data/processed/methylgpt_embeddings.npy`
  * `data/processed/cpgpt_embeddings.npy`
  * `data/processed/dna_methylation_embeddings.npy`
* `data/processed/tcga_rna_seq.h5ad` and `data/processed/tcga_methylation.h5ad` - metadata with `case_id` and `cancer_type`.

Run the full classification workflow:
```bash
python scripts/train_classification_pipeline.py
```

The script evaluates:
* RNA-only classification.
* DNA methylation-only classification.
* Early fusion by concatenating RNA and methylation embeddings.
* Late fusion by averaging class probabilities from separate RNA and methylation classifiers.

The default classifier is XGBoost:
```bash
python scripts/train_classification_pipeline.py --models xgboost
```

Other available classifiers are logistic regression, random forest, and a neural network classifier:
```bash
python scripts/train_classification_pipeline.py --models logistic_regression random_forest mlp
```

Results are written to `outputs/classification/`:
* `metrics_summary.csv` - cross-validation Accuracy, macro F1, weighted F1, and one-vs-rest ROC-AUC.
* `predictions.csv` - fold-level predictions.
* `aligned_samples.csv` - samples shared by both omics modalities.
* `run_config.json` - paths and settings used for the run.

