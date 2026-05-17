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

Clone the MethylGPT repository separately into the project folder when needed:
```bash
git clone <MethylGPT repository URL> MethylGPT_repo
```

The `MethylGPT_repo/` folder is ignored by Git so its source code and model files are not committed to this repository.

### Data Download
To download the initial test subset:
```bash
python scripts/download_tcga_data.py
```

