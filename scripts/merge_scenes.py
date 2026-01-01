#!/usr/bin/env python3
"""
===============================================================================
Script Name   : merge_scenes.py
Description   : Explicitly adds 'source_scene_id' to the manifest by parsing
                filenames. eliminating the need for external JSON lookups.

Usage         :
                python scripts/merge_scenes.py \
                --manifest data/manifests/df2023_manifest.csv \
                --metadata data/raw/coco_scene_labels.json \
                --out data/manifests/df2023_manifest_with_scenes.csv

Arguments     :
              --manifest  : Input : data/manifests/df2023_manifest.csv
              --metadata  : Path to the JSON file containing scene labels.
              --out       : Output: data/manifests/df2023_manifest.csv (Overwrites with new col)

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
import argparse
from pathlib import Path
import sys

def get_scene_id(row):
    """
    Parses scene ID from image_id string.
    Format: COCO_DF_MethodID_SceneID -> returns SceneID
    Example: COCO_DF_I000B00000_00918198 -> 00918198
    """
    # If already exists, keep it
    if 'source_scene_id' in row and pd.notna(row['source_scene_id']):
        return row['source_scene_id']
        
    # Parse from image_id
    if 'image_id' in row:
        val = str(row['image_id'])
        # Split by underscore and take the last segment
        parts = val.split('_')
        if len(parts) > 1:
            return parts[-1]
    return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Add scene IDs to manifest.")
    parser.add_argument("--manifest", default="data/manifests/df2023_manifest.csv", 
                        help="Path to input manifest")
    parser.add_argument("--out", default=None, 
                        help="Path to output manifest (default: overwrite input)")
    args = parser.parse_args()

    input_path = Path(args.manifest)
    output_path = Path(args.out) if args.out else input_path

    if not input_path.exists():
        print(f"[ERROR] Manifest not found: {input_path}")
        sys.exit(1)

    print(f"[*] Loading: {input_path}")
    df = pd.read_csv(input_path)
    
    print("[*] Extracting Scene IDs from filenames...")
    df['source_scene_id'] = df.apply(get_scene_id, axis=1)
    
    # Validation stats
    n_unique = df['source_scene_id'].nunique()
    print(f"[*] Stats: {len(df)} images derived from {n_unique} unique scenes.")
    
    print(f"[*] Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("[OK] Manifest updated successfully.")

if __name__ == "__main__":
    main()
