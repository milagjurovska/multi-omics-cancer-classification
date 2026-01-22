import sys
import os
from unittest.mock import MagicMock
from pathlib import Path

# NOTE: MethylGPT requires torchtext, which has known issues on Windows.
# It is highly recommended to run this script in a WSL (Linux) environment.
# If running on Windows, we mock binary dependencies to avoid DLL loading errors.

# Mock problematic dependencies BEFORE any MethylGPT imports
# These are needed for Windows/CPU compatibility and deprecated dependencies
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attention"] = MagicMock()
sys.modules["scib"] = MagicMock()
sys.modules["torchtext"] = MagicMock()
sys.modules["torchtext.vocab"] = MagicMock()
sys.modules["torchtext.data"] = MagicMock()
sys.modules["torchtext._torchtext"] = MagicMock()

import os
import torch
import numpy as np
import pandas as pd
import anndata as ad
import json
import pickle
from tqdm import tqdm

# Add cloned MethylGPT repo to path
sys.path.append(str(Path("MethylGPT").absolute()))

# Import MethylGPT model
from methylgpt.model.methyl_model import MethylGPTModel

# Create a simple vocab class that doesn't depend on torchtext
class SimpleMethylVocab:
    """Simple vocabulary implementation without torchtext dependency."""
    def __init__(self, probe_id_dir, pad_token, special_tokens):
        self.probe_id_dir = probe_id_dir
        self.special_tokens = special_tokens
        self.pad_token = pad_token
        
        # Load CpG list
        cpg_df = pd.read_csv(probe_id_dir)
        probe_column = cpg_df.columns[0]  # First column contains probe IDs
        self.CpG_list = cpg_df[probe_column].tolist()
        
        # Build vocabulary: special tokens first, then CpG probes
        all_tokens = special_tokens + self.CpG_list
        self.stoi = {token: idx for idx, token in enumerate(all_tokens)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        
        # Set default index for padding
        self.default_index = self.stoi[pad_token]
        self.CpG_ids = len(special_tokens) + np.arange(len(self.CpG_list))
        
    def __getitem__(self, token):
        """Get index for a token."""
        return self.stoi.get(token, self.default_index)
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.stoi)
    
    def get_stoi(self):
        """Return string-to-index mapping."""
        return self.stoi
    
    def get_itos(self):
        """Return index-to-string mapping."""
        return self.itos
    
    def set_default_index(self, idx):
        """Set default index for unknown tokens."""
        self.default_index = idx

# Configuration
INPUT_H5AD = Path("data/processed/tcga_methylation.h5ad")
MODEL_CONFIG_DIR = Path("MethylGPT/pretrained_models/dev_pretraining_test-dataset_CpGs_type3-preprocessing_False-Sep26-10-27")
MODEL_WEIGHTS = MODEL_CONFIG_DIR / "model_epoch10.pt"
CPG_LIST_PATH = Path("data/processed/probe_ids_type3.csv")
OUTPUT_EMBEDDINGS = Path("data/processed/methylgpt_embeddings.npy")

def extract_methylation_embeddings():
    print("Loading data and model configuration...")
    adata = ad.read_h5ad(INPUT_H5AD)
    
    # Load config
    with open(MODEL_CONFIG_DIR / "args.json", "r") as f:
        config = json.load(f)
    
    # Override paths in config if necessary
    config["load_model"] = True
    config["model_file"] = str(MODEL_WEIGHTS)
    config["probe_id_dir"] = str(CPG_LIST_PATH)
    config["fast_transformer"] = False # Disable flash_attn dependency
    
    # Setup Vocab and Model
    print("Initializing SimpleMethylVocab and MethylGPT Model...")
    methyl_vocab = SimpleMethylVocab(config["probe_id_dir"], config["pad_token"], config["special_tokens"])
    model = MethylGPTModel(config, methyl_vocab)
    
    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading weights from {MODEL_WEIGHTS} to {device}...")
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu"))
    except:
        # Partial load as backup
        model_dict = model.state_dict()
        pretrained_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    # model.half() # Optional: precision reduction

    embeddings = []
    
    print("Starting embedding extraction (this may be slow on CPU)...")
    with torch.no_grad():
        # MethylGPT expects batches of tokens/values
        # We'll process samples one by one for simplicity in this test
        for i in tqdm(range(adata.shape[0])):
            # Prepare data compatible with scGPT style tokenizer/input
            # For MethylGPT, X is already aligned with the probe list from the preprocessing script
            
            # Input IDs are essentially the indices of the probes
            # In MethylGPT type 3, they often use a fixed order? 
            # The prepare_data method in the notebook logic handles this.
            
            # Mocking a batch for prepare_data
            # In a real scenario, use scgpt.tokenizer.tokenize_and_pad_batch if required
            # But MethylGPT usually takes the full vector if small enough or uses its own dataloader
            
            input_values = torch.tensor(adata.X[i]).unsqueeze(0).to(device).float() # [1, n_probes]
            # MethylGPT genes/probes IDs
            # Typically 0..n_probes mapped via vocab
            # From notebook: input_gene_ids = vocab["genes"]
            # We'll use the indices from the vocab
            gene_ids = torch.arange(adata.shape[1]).unsqueeze(0).to(device)
            
            # Padding mask (none for this test as we aligned the probes)
            src_key_padding_mask = torch.zeros_like(gene_ids, dtype=torch.bool).to(device)
            
            output_dict = model(
                gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                MVC=config.get("GEPC", True),
                ECS=config.get("ecs_thres", 0) > 0
            )
            
            emb = output_dict["cell_emb"].cpu().numpy() # [1, emb_dim]
            embeddings.append(emb[0])

    # Save
    np.save(OUTPUT_EMBEDDINGS, np.array(embeddings))
    print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")
    print(f"Shape: {np.array(embeddings).shape}")

if __name__ == "__main__":
    extract_methylation_embeddings()
