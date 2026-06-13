# ============================================================
# Geneformer Embedding Extraction — Google Colab Script
# Person 3 (Aleksandar) — Run this in a Colab notebook
# Runtime: T4 GPU | ~30-45 min for 600 samples
# ============================================================

# ── CELL 1: Enable GPU ───────────────────────────────────────
# Before running anything:
# Runtime → Change runtime type → T4 GPU → Save

# ── CELL 2: Install dependencies ────────────────────────────
# !pip install -q anndata transformers tqdm

# ── CELL 3: Mount Google Drive ──────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

# ── CELL 4: Check GPU ───────────────────────────────────────
# import torch
# print(torch.cuda.is_available())       # must be True
# print(torch.cuda.get_device_name(0))   # should say Tesla T4

# ── CELL 5: Run everything below ────────────────────────────

import anndata as ad
import pickle
import numpy as np
import torch
from transformers import AutoModel
from pathlib import Path
from tqdm import tqdm

# ── Paths — update DRIVE_ROOT to your shared folder path ────
DRIVE_ROOT      = Path("/content/drive/MyDrive/shared_folder")
INPUT_H5AD      = DRIVE_ROOT / "data/processed/tcga_rna_seq.h5ad"
TOKEN_DICT_PATH = DRIVE_ROOT / "data/processed/token_dictionary.pkl"
OUTPUT_EMB      = DRIVE_ROOT / "data/processed/geneformer_embeddings.npy"
OUTPUT_LABELS   = DRIVE_ROOT / "data/processed/geneformer_labels.npy"

MODEL_NAME = "ctheodoris/Geneformer"
MAX_LEN    = 2048
BATCH_SIZE = 8   # T4 has 16GB — use batch size 8 for speed


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading AnnData and token dictionary...")
    adata      = ad.read_h5ad(INPUT_H5AD)
    token_dict = load_pickle(TOKEN_DICT_PATH)
    print(f"Samples: {adata.shape[0]}  Genes: {adata.shape[1]}")

    print(f"Loading Geneformer from HuggingFace...")
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval().to(device)

    # Match genes to Geneformer vocabulary
    valid_genes   = [g for g in adata.var_names if g in token_dict]
    gene_to_token = {g: token_dict[g] for g in valid_genes}
    print(f"Matched {len(valid_genes)}/{adata.shape[1]} genes to Geneformer vocab")
    adata = adata[:, valid_genes].copy()

    X      = adata.X
    labels = adata.obs["cancer_type"].values

    embeddings = []

    print(f"Extracting embeddings for {adata.shape[0]} samples...")
    for batch_start in tqdm(range(0, adata.shape[0], BATCH_SIZE)):
        batch_end    = min(batch_start + BATCH_SIZE, adata.shape[0])
        batch_tokens = []

        for i in range(batch_start, batch_end):
            counts = np.array(X[i]).flatten()

            # Rank genes by expression (Geneformer's input format)
            nonzero_mask   = counts > 0
            nonzero_counts = counts[nonzero_mask]
            nonzero_genes  = np.array(valid_genes)[nonzero_mask]
            sort_idx       = np.argsort(-nonzero_counts)
            sorted_genes   = nonzero_genes[sort_idx]

            tokens = [gene_to_token[g] for g in sorted_genes[:MAX_LEN]]
            if len(tokens) < MAX_LEN:
                tokens += [0] * (MAX_LEN - len(tokens))

            batch_tokens.append(tokens)

        input_ids = torch.tensor(batch_tokens).to(device)

        with torch.no_grad():
            outputs        = model(input_ids)
            hidden_states  = outputs.last_hidden_state       # (B, seq_len, hidden)
            mean_embedding = hidden_states.mean(dim=1)       # (B, hidden)
            embeddings.append(mean_embedding.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    np.save(OUTPUT_EMB,    embeddings)
    np.save(OUTPUT_LABELS, labels)

    print(f"\nDone!")
    print(f"Embeddings shape : {embeddings.shape}")
    print(f"Labels shape     : {labels.shape}")
    print(f"Saved to         : {OUTPUT_EMB}")

    # Quick sanity check
    print(f"\nSamples per cancer type:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")


if __name__ == "__main__":
    extract_embeddings()
