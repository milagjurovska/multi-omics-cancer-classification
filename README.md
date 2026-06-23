# Multi-Omics Cancer Classification using Foundation Models

This project implements a multi-omics approach to cancer classification, leveraging transfer learning from foundation models specialized in biological data. We integrate DNA Methylation and RNA-seq data from TCGA (The Cancer Genome Atlas) to improve diagnostic accuracy across multiple cancer types.

## Motivation
Cancer is a complex disease requiring analysis across multiple molecular levels. Foundation models like **CpGPT** for methylation and **BulkFormer** for transcriptomics offer rich representations that capture intricate biological dependencies better than traditional methods.

## Methodology
The project follows a multi-phase approach:
1.  **Parallel Preprocessing**:
    *   **RNA-seq**: Processed for BulkFormer to capture gene interactions.
    *   **DNA Methylation**: Processed for CpGPT to extract epigenetic embeddings.
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

#### BulkFormer (RNA-seq)
BulkFormer is a transformer pretrained on ~500k bulk RNA-seq samples. The architecture files are not included in this repository. Download them by running:
```bash
python scripts/setup_bulkformer.py
```

Then place the following data files in `BulkFormer/data/` (download from [Zenodo doi:10.5281/zenodo.15744294](https://doi.org/10.5281/zenodo.15744294)):
- `G_tcga.pt`, `G_tcga_weight.pt`, `esm2_feature_concat.pt`, `interested_gene_list.pt`

And place `BulkFormer_147M.pt` in `pretrained_models/`.

#### CpGPT (DNA methylation)
CpGPT is used for DNA methylation embeddings. Run `notebooks/colab/cpgpt_colab.ipynb` on Google Colab.

#### Note on additional embeddings
We also have precomputed embeddings from **Geneformer** (RNA-seq, 512 dims) and **MethylGPT** (DNA methylation, 128 dims) stored in `data/processed/`. These were extracted on Google Colab via `notebooks/colab/geneformer_colab.ipynb` and `notebooks/colab/methylgpt_colab.ipynb` and are available for comparison experiments.

### Data Download
To download the initial test subset:
```bash
python scripts/download_tcga_data.py
```

## Classification Pipeline

After preprocessing and embedding extraction, the modeling pipeline trains cancer-type classifiers from the foundation-model embeddings.

Expected inputs:
* `data/processed/bulkformer_embeddings.npy` - RNA-seq embeddings from BulkFormer (800 samples, 643 dims).
* `data/processed/cpgpt_embeddings.npy` - DNA methylation embeddings from CpGPT.
* Additional available embeddings: `geneformer_embeddings.npy`, `methylgpt_embeddings.npy`.

Run the full classification workflow:
```bash
python scripts/train_classifier.py
```

The script evaluates:
* RNA-only classification.
* DNA methylation-only classification.
* Early fusion by concatenating RNA and methylation embeddings.
* Late fusion by averaging class probabilities from separate RNA and methylation classifiers.

Results are written to `outputs/classification/`:
* `metrics_summary.csv` - cross-validation Accuracy, macro F1, weighted F1, and one-vs-rest ROC-AUC.
* `predictions.csv` - fold-level predictions.
* `aligned_samples.csv` - samples shared by both omics modalities.
* `run_config.json` - paths and settings used for the run.
