import torch
import numpy as np
import anndata as ad
import json
from pathlib import Path
from tqdm import tqdm
from methylgpt.model.methyl_model import MethylGPTModel
from methylgpt.model.methyl_vocab import MethylVocab
from scgpt.tokenizer import tokenize_and_pad_batch

# Configuration
INPUT_H5AD = Path("data/processed/tcga_methylation.h5ad")
MODEL_DIR = Path("pretrained_models/methylgpt-medium")
MODEL_WEIGHTS = MODEL_DIR / "small-best_model_epoch6.pt"
CONFIG_FILE = MODEL_DIR / "args.json"
CPG_LIST_PATH = Path("data/processed/probe_ids_type3.csv")
OUTPUT_EMBEDDINGS = Path("data/processed/methylgpt_embeddings.npy")

BATCH_SIZE = 1

def extract_methylation_embeddings():
    print("Loading preprocessed methylation data...")
    adata = ad.read_h5ad(INPUT_H5AD)
    print(f"Data shape: {adata.shape}")

    # Load model config
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    # Keep fast_transformer=True to match pretrained weight keys (Wqkv vs in_proj)
    # The model falls back to PyTorch attention if flash_attn is not installed
    config["load_model"] = True
    config["pretrained_file"] = str(MODEL_WEIGHTS)

    # Build vocabulary
    print("Building MethylVocab...")
    vocab = MethylVocab(
        probe_id_dir=str(CPG_LIST_PATH),
        pad_token="<pad>",
        special_tokens=["<pad>", "<cls>", "<eoc>"],
        save_dir=None,
    )
    print(f"Vocab size: {len(vocab)}, CpG probes: {len(vocab.CpG_ids)}")

    # Create model and load pretrained weights
    print(f"Loading model (layer_size={config['layer_size']}, nlayers={config['nlayers']})...")
    model = MethylGPTModel(config, vocab)

    # The pretrained weights use flash_attn's fused Wqkv format, but without
    # flash_attn installed the model uses standard PyTorch in_proj_weight/bias.
    # Convert the key names to match.
    state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
    converted = {}
    for k, v in state_dict.items():
        new_key = k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
        new_key = new_key.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
        converted[new_key] = v
    model.load_state_dict(converted)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Prepare data: replace NaN with pad_value (-2)
    data = adata.X.copy()
    data = np.nan_to_num(data, nan=-2.0)

    max_len = len(vocab.CpG_ids) + 1  # +1 for <cls> token

    embeddings = []

    print(f"Extracting embeddings for {adata.shape[0]} samples...")
    with torch.no_grad():
        for start in tqdm(range(0, adata.shape[0], BATCH_SIZE)):
            end = min(start + BATCH_SIZE, adata.shape[0])
            batch_data = data[start:end]

            tokenized = tokenize_and_pad_batch(
                batch_data,
                vocab.CpG_ids,
                max_len=max_len,
                vocab=vocab,
                pad_token="<pad>",
                pad_value=-2,
                append_cls=True,
                include_zero_gene=True,
            )

            gene_ids = tokenized["genes"].to(device)
            values = tokenized["values"].to(device)

            cell_embs = model.get_cell_embeddings(gene_ids, values)
            embeddings.append(cell_embs.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    extract_methylation_embeddings()
