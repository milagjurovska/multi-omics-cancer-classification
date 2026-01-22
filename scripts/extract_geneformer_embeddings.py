import anndata as ad
import pickle
import numpy as np
import torch
from transformers import BertModel, BertConfig
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_H5AD = Path("data/processed/tcga_rna_seq.h5ad")
TOKEN_DICT_PATH = Path("data/processed/token_dictionary.pkl")
MODEL_NAME = "ctheodoris/Geneformer"
OUTPUT_EMBEDDINGS = Path("data/processed/geneformer_embeddings.npy")
MAX_LEN = 2048

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_embeddings():
    print("Loading data and dictionaries...")
    adata = ad.read_h5ad(INPUT_H5AD)
    token_dict = load_pickle(TOKEN_DICT_PATH)
    
    print(f"Loading model: {MODEL_NAME}...")
    model = BertModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    gene_to_token = {gene: token_dict.get(gene) for gene in adata.var_names if gene in token_dict}
    valid_genes = list(gene_to_token.keys())
    print(f"Matched {len(valid_genes)} genes with Geneformer vocabulary.")
    
    adata = adata[:, valid_genes].copy()
    
    embeddings = []
    
    print("Extracting embeddings for samples...")
    for i in tqdm(range(adata.shape[0])):
        counts = adata.X[i]
        nonzero_mask = counts > 0
        nonzero_counts = counts[nonzero_mask]
        nonzero_genes = np.array(valid_genes)[nonzero_mask]
        
        sort_idx = np.argsort(-nonzero_counts)
        sorted_genes = nonzero_genes[sort_idx]
        
        tokens = [gene_to_token[g] for g in sorted_genes[:MAX_LEN]]
        
        if len(tokens) < MAX_LEN:
            tokens = tokens + [0] * (MAX_LEN - len(tokens))
            
        input_ids = torch.tensor([tokens])
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            
        with torch.no_grad():
            outputs = model(input_ids)
            hidden_states = outputs.last_hidden_state 
            mean_embedding = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_embedding[0])
            
    np.save(OUTPUT_EMBEDDINGS, np.array(embeddings))
    print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")
    print(f"Embedding shape: {np.array(embeddings).shape}")

if __name__ == "__main__":
    extract_embeddings()
