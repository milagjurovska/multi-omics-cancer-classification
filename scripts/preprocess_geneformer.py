import os
import pandas as pd
import anndata as ad
from pathlib import Path
from tqdm import tqdm

# Configuration
MANIFEST_PATH = Path("data/manifests/matched_samples.csv")
RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/tcga_rna_seq.h5ad")

def preprocess_rna_seq():
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found at {MANIFEST_PATH}. Please run the download script first.")
        return

    # Load manifest
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Loading {len(manifest)} samples from manifest...")

    # Lists to store data
    data_list = []
    obs_list = []
    var_names = None

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        project = row["project"]
        file_name = row["rna_file_name"]
        case_id = row["case_id"]
        
        file_path = RAW_DATA_DIR / project / "rna_seq" / file_name
        
        if not file_path.exists():
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        # Read TSV, skipping the first row (metadata comment)
        # Column names are on line 2 (header=0 after skipping 1)
        df = pd.read_csv(file_path, sep="\t", skiprows=1)
        
        # Filter out the first 4 rows which are summary stats (N_unmapped, etc.)
        df = df[df["gene_id"].str.startswith("ENSG")]
        
        # Remove version from gene_id
        df["gene_id"] = df["gene_id"].str.split(".").str[0]
        
        # Aggregate duplicates (summing counts)
        df_aggregated = df.groupby("gene_id").agg({
            "unstranded": "sum",
            "gene_name": "first",
            "gene_type": "first"
        })
        
        # Use aggregated data
        counts = df_aggregated["unstranded"]
        
        # Store data
        data_list.append(counts.values)
        
        # Store metadata
        obs_list.append({
            "case_id": case_id,
            "cancer_type": project,
            "file_id": row["rna_file_id"]
        })
        
        # Get variable names (genes) from the first file
        if var_names is None:
            var_names = df_aggregated.index.tolist()
            gene_names = df_aggregated["gene_name"].tolist()
            gene_types = df_aggregated["gene_type"].tolist()

    if not data_list:
        print("No data found to process.")
        return

    # Create AnnData object
    print("Creating AnnData object...")
    adata = ad.AnnData(
        X=pd.DataFrame(data_list, columns=var_names).values,
        obs=pd.DataFrame(obs_list),
        var=pd.DataFrame(index=var_names, data={"gene_name": gene_names, "gene_type": gene_types})
    )

    # Save to H5AD
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUTPUT_PATH)
    print(f"Saved processed RNA-seq data to {OUTPUT_PATH}")
    print(f"Shape: {adata.shape}")

if __name__ == "__main__":
    preprocess_rna_seq()
