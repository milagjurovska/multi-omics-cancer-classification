import os
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from tqdm import tqdm

# Configuration
MANIFEST_PATH = Path("data/manifests/matched_samples.csv")
RAW_DATA_DIR = Path("data/raw")
PROBE_LIST_PATH = Path("data/processed/probe_ids_type3.csv")
OUTPUT_PATH = Path("data/processed/tcga_methylation.h5ad")

def preprocess_methylation():
    if not MANIFEST_PATH.exists() or not PROBE_LIST_PATH.exists():
        print("Required files (manifest or probe list) missing.")
        return

    # Load manifest
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Loading {len(manifest)} samples from manifest...")

    # Load target CpG probe list (single column with Probe_ID)
    # The Dropbox file usually has probe ids in the first column
    probes_df = pd.read_csv(PROBE_LIST_PATH)
    probe_column = "illumina_probe_id" if "illumina_probe_id" in probes_df.columns else probes_df.columns[-1]
    target_probes = probes_df[probe_column].tolist()
    print(f"Targeting {len(target_probes)} CpG probes from MethylGPT list.")

    data_list = []
    obs_list = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        project = row["project"]
        file_name = row["meth_file_name"]
        case_id = row["case_id"]
        
        file_path = RAW_DATA_DIR / project / "methylation" / file_name
        
        if not file_path.exists():
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        # TCGA level 3 betas are typically space/tab delimited: [CpG_ID] [Beta_Value]
        # Some have headers, some don't. We'll try to infer.
        try:
            sample_df = pd.read_csv(file_path, sep="\t", header=None, names=["probe", "beta"])
            # Handle possible conversion issues (NA strings)
            sample_df["beta"] = pd.to_numeric(sample_df["beta"], errors="coerce")
            sample_df = sample_df.set_index("probe")
            
            # Reindex to match target probe list, filling missing with NaN
            sample_betas = sample_df.reindex(target_probes)["beta"].values
            
            data_list.append(sample_betas)
            obs_list.append({
                "case_id": case_id,
                "cancer_type": project,
                "file_id": row["meth_file_id"]
            })
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if not data_list:
        print("No methylation data processed.")
        return

    # Create AnnData
    print("Creating AnnData object...")
    X = np.stack(data_list)
    
    # Handle missing values (simple mean imputation per site or zeroing)
    # For foundation models, usually zeros or mean is used if not too many
    print("Handling missing values (filling with 0)...")
    X = np.nan_to_num(X, nan=0.0)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(obs_list),
        var=pd.DataFrame(index=target_probes)
    )

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUTPUT_PATH)
    print(f"Saved processed methylation data to {OUTPUT_PATH}")
    print(f"Shape: {adata.shape}")

if __name__ == "__main__":
    preprocess_methylation()
