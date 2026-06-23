"""
Downloads only the BulkFormer files needed for inference — no git clone required.
Run once before opening extract_bulkformer_embeddings.ipynb.

    python scripts/setup_bulkformer.py
"""
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/KangBoming/BulkFormer/main"

FILES = {
    "utils/BulkFormer.py":       f"{BASE_URL}/utils/BulkFormer.py",
    "utils/BulkFormer_block.py": f"{BASE_URL}/utils/BulkFormer_block.py",
    "utils/Rope.py":             f"{BASE_URL}/utils/Rope.py",
    "model/config.py":           f"{BASE_URL}/model/config.py",
    "data/bulkformer_gene_info.csv": f"{BASE_URL}/data/bulkformer_gene_info.csv",
    "data/gene_length_df.csv":       f"{BASE_URL}/data/gene_length_df.csv",
}

DEST_ROOT = Path(__file__).parent.parent / "BulkFormer"

def main():
    print(f"Setting up BulkFormer at: {DEST_ROOT}\n")
    for rel_path, url in FILES.items():
        dest = DEST_ROOT / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"  already exists  {rel_path}")
            continue
        print(f"  downloading     {rel_path} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
    print("\nDone. Place the 4 Zenodo .pt files in BulkFormer/data/ and you are ready.")

if __name__ == "__main__":
    main()
