import requests
import json
import os
import pandas as pd
from pathlib import Path

# Configuration
PROJECTS = ["TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD", "TCGA-KIRC", "TCGA-LIHC", "TCGA-THCA"]
DATA_DIR = Path("data/raw")
MANIFEST_DIR = Path("data/manifests")
PROJECT_SAMPLE_LIMITS = {
    "TCGA-BRCA": 134,
    "TCGA-LUAD": 134,
    "TCGA-COAD": 133,
    "TCGA-KIRC": 133,
    "TCGA-LIHC": 133,
    "TCGA-THCA": 133,
}

# GDC API Endpoints
FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

def query_gdc_for_matched_samples(project_id):
    """
    Finds cases in a project that have both RNA-seq and DNA Methylation data.
    """
    print(f"Querying GDC for matched samples in {project_id}...")
    
    rna_filter = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}}
        ]
    }
    
    meth_filter = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Methylation Beta Value"]}},
            {"op": "in", "content": {"field": "files.platform", "value": ["Illumina Human Methylation 450"]}}
        ]
    }

    def get_files(filters):
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.submitter_id,cases.case_id",
            "format": "JSON",
            "size": "2000"
        }
        response = requests.get(FILES_ENDPOINT, params=params)
        response.raise_for_status()
        return response.json()["data"]["hits"]

    rna_files = get_files(rna_filter)
    meth_files = get_files(meth_filter)
   
    rna_map = {hit["cases"][0]["case_id"]: hit for hit in rna_files if hit.get("cases")}
    meth_map = {hit["cases"][0]["case_id"]: hit for hit in meth_files if hit.get("cases")}
  
    common_cases = sorted(set(rna_map.keys()) & set(meth_map.keys()))
    print(f"Found {len(common_cases)} cases with both RNA-seq and Methylation in {project_id}.")
    
    matched_pairs = []
    sample_limit = PROJECT_SAMPLE_LIMITS[project_id]
    for case_id in common_cases[:sample_limit]:
        matched_pairs.append({
            "project": project_id,
            "case_id": case_id,
            "rna_file_id": rna_map[case_id]["file_id"],
            "rna_file_name": rna_map[case_id]["file_name"],
            "meth_file_id": meth_map[case_id]["file_id"],
            "meth_file_name": meth_map[case_id]["file_name"]
        })
        
    return matched_pairs

def download_file(file_id, dest_path):
    if dest_path.exists():
        return

    response = requests.post(DATA_ENDPOINT, data=json.dumps({"ids": [file_id]}), headers={"Content-Type": "application/json"}, stream=True)
    response.raise_for_status()
    
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    temp_path.replace(dest_path)

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    
    all_matched = []
    
    for project in PROJECTS:
        matched = query_gdc_for_matched_samples(project)
        all_matched.extend(matched)

    manifest_df = pd.DataFrame(all_matched)
    manifest_df.to_csv(MANIFEST_DIR / "matched_samples.csv", index=False)
    print(f"Manifest saved to {MANIFEST_DIR / 'matched_samples.csv'} with {len(manifest_df)} matched samples.")

    for project in PROJECTS:
        matched = [pair for pair in all_matched if pair["project"] == project]
        project_dir = DATA_DIR / project
        (project_dir / "rna_seq").mkdir(parents=True, exist_ok=True)
        (project_dir / "methylation").mkdir(parents=True, exist_ok=True)
        
        print(f"Beginning download for {project}...")
        for i, pair in enumerate(matched):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(matched)}")
            
            download_file(pair["rna_file_id"], project_dir / "rna_seq" / pair["rna_file_name"])
            download_file(pair["meth_file_id"], project_dir / "methylation" / pair["meth_file_name"])
            
    print("Finished downloading data.")

if __name__ == "__main__":
    main()
