import sys
import os
from unittest.mock import MagicMock
from pathlib import Path

# Add MethylGPT_repo to path so we can import from it
REPO_PATH = Path("MethylGPT_repo").absolute()
sys.path.insert(0, str(REPO_PATH))

# Mock problematic dependencies with concrete classes for typing safety before any imports
class MockVocab: 
    def __init__(self, *args, **kwargs): pass
    def __getitem__(self, key): return 0
    def __len__(self): return 1000
    def set_default_index(self, idx): pass
    def get_stoi(self): return {}

mock_tt = MagicMock()
mock_tt.vocab.Vocab = MockVocab
mock_tt._torchtext.Vocab = MockVocab # This is the key for MethylVocab
sys.modules["torchtext"] = mock_tt
sys.modules["torchtext.vocab"] = mock_tt.vocab
sys.modules["torchtext._torchtext"] = mock_tt._torchtext

mock_sc_tok = MagicMock()
mock_sc_tok.GeneVocab = MockVocab
mock_sc_tok.GeneTokenizer = MagicMock()
mock_sc_tok.tokenize_and_pad_batch = MagicMock()
sys.modules["scgpt.tokenizer"] = mock_sc_tok
sys.modules["scgpt.tokenizer.gene_tokenizer"] = mock_sc_tok

sys.modules["IPython"] = MagicMock()
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attention"] = MagicMock()

# Force anndata version string for scanpy compatibility
try:
    import anndata
    anndata.__version__ = "0.9.1"
except ImportError:
    pass

import torch
import numpy as np
import pandas as pd
import anndata as ad
import json
from tqdm import tqdm

# Import and PATCH MethylVocab before use
from methylgpt.model.methyl_vocab import MethylVocab

# Redefine MethylVocab to not use VocabPybind
class PurePythonMethylVocab:
    def __init__(self, probe_id_dir, pad_token, special_tokens, save_dir=None):
        self.probe_id_dir = probe_id_dir
        self.special_tokens = special_tokens
        self.save_dir = save_dir
        self.pad_token = pad_token
        
        self.CpG_list = pd.read_csv(self.probe_id_dir)["illumina_probe_id"].tolist()
        self.tokens = self.special_tokens + self.CpG_list
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for i, t in enumerate(self.tokens)}
        self.default_index = self.stoi.get(pad_token, 0)
        self.CpG_ids = len(self.special_tokens) + np.arange(len(self.CpG_list))
    
    def __getitem__(self, key):
        return self.stoi.get(key, self.default_index)
    
    def __len__(self):
        return len(self.tokens)
    
    def get_stoi(self):
        return self.stoi

# Swap it out
import methylgpt.model.methyl_model
import methylgpt.model.methyl_vocab
methylgpt.model.methyl_vocab.MethylVocab = PurePythonMethylVocab
MethylVocab = PurePythonMethylVocab
from methylgpt.model.methyl_vocab import MethylVocab

# Configuration
INPUT_H5AD = Path("data/processed/tcga_methylation.h5ad")
# We'll use the model structure from tutorials/finetuning_age_prediction or similar
# For now, we need to download/link the checkpoint correctly.
# The user already has a checkpoint folder in the original MethylGPT dir, let's see if we can use it.
MODEL_CONFIG_DIR = Path("MethylGPT/pretrained_models/dev_pretraining_test-dataset_CpGs_type3-preprocessing_False-Sep26-10-27")
MODEL_WEIGHTS = MODEL_CONFIG_DIR / "model_epoch10.pt"
CPG_LIST_PATH = Path("data/processed/probe_ids_type3.csv")
OUTPUT_EMBEDDINGS = Path("data/processed/methylgpt_embeddings.npy")

def extract_methylation_embeddings():
    print("Loading data and model configuration...")
    adata = ad.read_h5ad(INPUT_H5AD)
    
    # Load config from the pretrained dir
    args_path = MODEL_CONFIG_DIR / "args.json"
    if args_path.exists():
        with open(args_path, "r") as f:
            config = json.load(f)
    else:
        # Default config if args.json is missing or we need to override
        config = {
            "emb_dim": 128,
            "n_layers": 6,
            "n_heads": 4,
            "fast_transformer": False,
            "load_model": True,
            "model_file": str(MODEL_WEIGHTS),
            "probe_id_dir": str(CPG_LIST_PATH),
            "pad_token": "<pad>",
            "special_tokens": ["<pad>", "<mask>", "<cls>"],
            "MVC": True,
            "ecs_thres": 0
        }
    
    # Override paths in config for the current environment
    config["load_model"] = True
    config["model_file"] = str(MODEL_WEIGHTS)
    config["probe_id_dir"] = str(CPG_LIST_PATH)
    config["fast_transformer"] = False # Disable flash_attn dependency
    
    # Setup Vocab and Model
    print("Initializing MethylVocab and MethylGPT Model...")
    # NOTE: MethylVocab in the new repo might have a different signature. 
    # Let's check methylgpt/model/methyl_vocab.py
    methyl_vocab = MethylVocab(config["probe_id_dir"], config["pad_token"], config["special_tokens"], save_dir=None)
    model = MethylGPTModel(config, methyl_vocab)
    
    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading weights from {MODEL_WEIGHTS} to {device}...")
    
    if MODEL_WEIGHTS.exists():
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu"))
            print("Successfully loaded model weights.")
        except Exception as e:
            print(f"Warning: Could not load full state dict: {e}")
            model_dict = model.state_dict()
            pretrained_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} layers from checkpoint.")
    else:
        print(f"ERROR: Model weights not found at {MODEL_WEIGHTS}")
        return

    model.to(device)
    model.eval()

    embeddings = []
    
    print("Starting embedding extraction...")
    with torch.no_grad():
        for i in tqdm(range(adata.shape[0])):
            # Prepare data
            input_values = torch.tensor(adata.X[i].toarray() if hasattr(adata.X[i], 'toarray') else adata.X[i]).unsqueeze(0).to(device).float()
            
            # Use vocab to get gene IDs (indices)
            gene_ids = torch.arange(adata.shape[1]).unsqueeze(0).to(device)
            
            # Padding mask
            src_key_padding_mask = torch.zeros_like(gene_ids, dtype=torch.bool).to(device)
            
            output_dict = model(
                gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                MVC=config.get("MVC", True),
                ECS=config.get("ecs_thres", 0) > 0
            )
            
            emb = output_dict["cell_emb"].cpu().numpy()
            embeddings.append(emb[0])

    # Save
    np.save(OUTPUT_EMBEDDINGS, np.array(embeddings))
    print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")
    print(f"Shape: {np.array(embeddings).shape}")

if __name__ == "__main__":
    extract_methylation_embeddings()
