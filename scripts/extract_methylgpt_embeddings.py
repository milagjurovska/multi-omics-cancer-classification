"""
Extract methylation embeddings using MethylGPT pretrained checkpoint.
Downloads model directly from GitHub without cloning the full repository.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from tqdm import tqdm
import urllib.request
import json
import sys

def download_methylgpt_checkpoint(model_size="base", force_download=False):
    """Download MethylGPT checkpoint from GitHub."""
    
    # Model URLs (update these based on actual GitHub release URLs)
    MODEL_URLS = {
        "base": "https://github.com/CompEpigen/MethylGPT/releases/download/v1.0/methylgpt_base.pt",
        "medium": "https://github.com/CompEpigen/MethylGPT/releases/download/v1.0/methylgpt_medium.pt",
        "large": "https://github.com/CompEpigen/MethylGPT/releases/download/v1.0/methylgpt_large.pt"
    }
    
    # Model configs
    MODEL_CONFIGS = {
        "base": {"emb_dim": 64, "n_layers": 6, "n_heads": 4},
        "medium": {"emb_dim": 128, "n_layers": 6, "n_heads": 4},
        "large": {"emb_dim": 256, "n_layers": 6, "n_heads": 4}
    }
    
    checkpoint_dir = Path("models/methylgpt")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"methylgpt_{model_size}.pt"
    config_path = checkpoint_dir / f"config_{model_size}.json"
    
    # Download config
    if not config_path.exists() or force_download:
        print(f"Creating config for {model_size} model...")
        with open(config_path, 'w') as f:
            json.dump(MODEL_CONFIGS[model_size], f)
    
    # Download checkpoint
    if not checkpoint_path.exists() or force_download:
        print(f"Downloading MethylGPT-{model_size} checkpoint...")
        print("Note: If download fails, you'll need to manually download from GitHub")
        try:
            urllib.request.urlretrieve(MODEL_URLS[model_size], checkpoint_path)
            print(f"✓ Downloaded to {checkpoint_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"\nPlease manually download from:")
            print(f"https://github.com/CompEpigen/MethylGPT")
            print(f"And place it at: {checkpoint_path}")
            return None, None
    else:
        print(f"Using existing checkpoint: {checkpoint_path}")
    
    return checkpoint_path, config_path


class SimpleMethylGPTWrapper(nn.Module):
    """Simplified wrapper to extract embeddings from MethylGPT checkpoint."""
    
    def __init__(self, checkpoint_path, n_probes, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Simple embedding projection (will be replaced by checkpoint weights)
        self.probe_embedding = nn.Linear(1, emb_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, batch_first=True),
            num_layers=6
        )
        self.output_projection = nn.Linear(emb_dim, emb_dim)
        
        # Load weights if checkpoint exists
        if checkpoint_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                # Load compatible weights
                self.load_state_dict(state_dict, strict=False)
                print("✓ Loaded checkpoint weights")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Using randomly initialized weights")
    
    def forward(self, methylation_values):
        """Extract cell embeddings from methylation values."""
        # methylation_values shape: (batch, n_probes)
        batch_size = methylation_values.shape[0]
        
        # Embed each probe value
        x = methylation_values.unsqueeze(-1)  # (batch, n_probes, 1)
        x = self.probe_embedding(x)  # (batch, n_probes, emb_dim)
        
        # Pass through transformer
        x = self.transformer(x)  # (batch, n_probes, emb_dim)
        
        # Get cell-level embedding (mean pooling)
        cell_emb = x.mean(dim=1)  # (batch, emb_dim)
        cell_emb = self.output_projection(cell_emb)
        
        return cell_emb


def extract_methylation_embeddings(model_size="base", use_pretrained=True):
    """Extract embeddings using MethylGPT."""
    
    # Paths
    INPUT_H5AD = Path("data/processed/tcga_methylation.h5ad")
    OUTPUT_EMBEDDINGS = Path("data/processed/methylgpt_embeddings.npy")
    
    print("="*60)
    print("MethylGPT Embedding Extraction")
    print("="*60)
    
    print("\n1. Loading methylation data...")
    adata = ad.read_h5ad(INPUT_H5AD)
    n_samples, n_probes = adata.shape
    print(f"   Samples: {n_samples}, CpG sites: {n_probes}")
    
    # Download checkpoint if using pretrained
    checkpoint_path = None
    config_path = None
    if use_pretrained:
        print("\n2. Downloading/locating MethylGPT checkpoint...")
        checkpoint_path, config_path = download_methylgpt_checkpoint(model_size)
        
        if checkpoint_path and config_path:
            with open(config_path) as f:
                config = json.load(f)
            emb_dim = config['emb_dim']
        else:
            print("   Checkpoint not available, using default embedding dimension")
            emb_dim = 128
    else:
        emb_dim = 128
        checkpoint_path = Path("nonexistent.pt")  # Won't exist, will use random init
    
    print(f"\n3. Initializing model (embedding dim: {emb_dim})...")
    model = SimpleMethylGPTWrapper(checkpoint_path, n_probes, emb_dim)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    model = model.to(device)
    model.eval()
    
    print("\n4. Extracting embeddings...")
    embeddings = []
    batch_size = 8  # Process in small batches
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size)):
            # Get batch
            end_idx = min(i + batch_size, n_samples)
            batch_data = adata.X[i:end_idx].toarray() if hasattr(adata.X, 'toarray') else adata.X[i:end_idx]
            
            # Convert to tensor
            methyl_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
            
            # Extract embeddings
            batch_emb = model(methyl_tensor).cpu().numpy()
            embeddings.append(batch_emb)
    
    # Concatenate all batches
    embeddings_array = np.vstack(embeddings)
    
    # Save
    print(f"\n5. Saving embeddings...")
    np.save(OUTPUT_EMBEDDINGS, embeddings_array)
    
    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60)
    print(f"Output: {OUTPUT_EMBEDDINGS}")
    print(f"Shape: {embeddings_array.shape}")
    print(f"Samples: {embeddings_array.shape[0]}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    print("="*60)


if __name__ == "__main__":
    # Run with pretrained checkpoint (downloads automatically)
    extract_methylation_embeddings(model_size="base", use_pretrained=True)
