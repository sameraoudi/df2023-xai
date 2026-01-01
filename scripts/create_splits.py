#!/usr/bin/env python3
"""
===============================================================================
Script Name   : create_splits.py
Description   : Data Split Generator
                This script deterministically splits a master manifest CSV into separate physical 
                CSV files (e.g., `train_split.csv`, `val_split.csv`) based on the 'split' column. 
                This ensures that the training pipeline loads a fixed, version-controlled subset 
                of the data, rather than relying on random runtime splitting.

Usage        :
                # Standard usage
                python scripts/create_splits.py data/manifests/df2023_manifest.csv
            
                # Custom output directory
                python scripts/create_splits.py data/manifests/df2023_manifest.csv --out-dir data/my_splits

Arguments      :
                master_csv  : Path to the master manifest CSV file. Must contain a 'split' column.
                --out-dir   : (Optional) Directory where split files will be saved. 
                              Default: data/manifests/splits

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
import os
import argparse
import sys
from pathlib import Path

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Deterministically split the master manifest into separate train/val files."
    )
    parser.add_argument(
        "master_csv", 
        type=str,
        help="Path to the master manifest CSV file (must contain a 'split' column)."
    )
    parser.add_argument(
        "--out-dir", 
        type=str,
        default="data/manifests/splits", 
        help="Directory where the resulting split CSVs will be saved."
    )
    args = parser.parse_args()

    # 2. Validate Input
    input_path = Path(args.master_csv)
    if not input_path.exists():
        print(f"[ERROR] Master manifest not found at: {input_path}")
        sys.exit(1)

    print(f"Loading master manifest: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        sys.exit(1)

    if 'split' not in df.columns:
        print(f"[ERROR] The file {input_path} is missing the required 'split' column.")
        sys.exit(1)

    # 3. Ensure Output Directory Exists
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Process Splits
    unique_splits = df['split'].unique()
    print(f"Found {len(unique_splits)} unique splits: {list(unique_splits)}")

    for split_name in unique_splits:
        # Filter data for this split
        split_df = df[df['split'] == split_name]
        
        # Define output filename (e.g., train_split.csv)
        output_filename = f"{split_name}_split.csv"
        output_path = out_dir / output_filename
        
        # Save to disk
        split_df.to_csv(output_path, index=False)
        print(f"✔ Created {split_name:<10} split: {output_path} ({len(split_df):,} rows)")

if __name__ == "__main__":
    main()
