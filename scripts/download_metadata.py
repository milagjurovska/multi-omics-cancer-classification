import requests
from pathlib import Path

urls = {
    "data/processed/token_dictionary.pkl": "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary_gc104M.pkl?download=true",
    "data/processed/gene_median_dictionary.pkl": "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_median_dictionary_gc104M.pkl?download=true",
    "data/processed/probe_ids_type3.csv": "https://www.dropbox.com/scl/fi/2n6bx7j8v0aon0kwfsghp/probe_ids_type3.csv?rlkey=ly133xlce1xxjiku6tiski6qq&st=pig4e41h&dl=1"
}

def download_metadata():
    for rel_path, url in urls.items():
        path = Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} to {path}...")
        try:
            r = requests.get(url, stream=True, allow_redirects=True)
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {path}")
        except Exception as e:
            print(f"Error downloading {rel_path}: {e}")

if __name__ == "__main__":
    download_metadata()
