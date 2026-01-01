#!/usr/bin/env python3
"""
===============================================================================
Script Name   : merge_scenes.py
Description   : Scene Metadata Merger
                This script merges external scene-level metadata (from a JSON file) into the 
                training manifest. It creates a new CSV file containing a 'scene_id' column.
            
                This is Step 3 of the pipeline, required for Forensic Evaluation to ensure 
                data is split by scene (video/location) rather than by frame, preventing 
                data leakage.

Usage         :
                python scripts/merge_scenes.py \
                --manifest data/manifests/df2023_manifest.csv \
                --metadata data/raw/coco_scene_labels.json \
                --out data/manifests/df2023_manifest_with_scenes.csv

Arguments     :
              --manifest  : Path to the input training manifest CSV.
              --metadata  : Path to the JSON file containing scene labels.
              --out       : Path where the enriched CSV will be saved.

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119
Created On    : 2025-Dec-31

License       : MIT License
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.

Dependencies  :
- pandas
===============================================================================
"""

import pandas as pd
import json
import argparse
import sys
from pathlib import Path

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Merge scene metadata into the dataset manifest."
    )
    parser.add_argument(
        "--manifest", 
        type=str, 
        required=True,
        help="Path to the input manifest CSV (e.g., df2023_manifest.csv)"
    )
    parser.add_argument(
        "--metadata", 
        type=str, 
        required=True,
        help="Path to the JSON file containing scene/video IDs."
    )
    parser.add_argument(
        "--out", 
        type=str, 
        required=True,
        help="Path to save the new manifest with 'scene_id' column."
    )
    args = parser.parse_args()

    # 2. Load Manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)
    
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    print(f"  Rows: {len(df):,}")

    # 3. Load Metadata
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print(f"[ERROR] Metadata file not found: {meta_path}")
        sys.exit(1)

    print(f"Loading metadata: {meta_path}")
    try:
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {e}")
        sys.exit(1)

    # 4. Perform Merge
    # NOTE: Adjust this logic if your JSON structure differs.
    # Assumption: JSON is a dict like {"image_id_1": "scene_A", "image_id_2": "scene_A"}
    
    print("Merging scene data...")
    
    # Check if 'image_id' exists in manifest
    if 'image_id' not in df.columns:
        print("[ERROR] Manifest missing 'image_id' column. Cannot merge.")
        sys.exit(1)

    # Map the scene_id using the image_id
    # If the metadata is a list or complex object, pre-process it into a dict here.
    if isinstance(meta_data, list):
        # Example handling if it's a list of dicts
        # meta_dict = {item['id']: item['scene'] for item in meta_data}
        print("[WARNING] Metadata is a list. Assuming conversion is needed (Update script if logic fails).")
        pass 
    
    # Apply mapping
    # This creates a new column 'scene_id'. If missing in JSON, fills with 'unknown'
    df['scene_id'] = df['image_id'].map(meta_data)
    
    # Check coverage
    missing_count = df['scene_id'].isna().sum()
    if missing_count > 0:
        print(f"[WARNING] {missing_count} images did not have a matching scene ID in the JSON.")
        df['scene_id'] = df['scene_id'].fillna('unknown')
    else:
        print("✔ All images successfully matched to a scene.")

    # 5. Save Output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out_path, index=False)
    print(f"✔ Saved enriched manifest: {out_path}")
    print(f"  Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
