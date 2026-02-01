"""
===============================================================================
PROJECT      : DF2023-XAI 
SCRIPT       : create_splits.py
VERSION      : 1.1.0
DESCRIPTION  : Scene-Disjoint dataset partitioning (Methodology Section 3.1).
-------------------------------------------------------------------------------
FUNCTIONALITY:
    Partitions the DF2023 dataset into Train (80%), Val (10%), and Test (10%) 
    subsets while guaranteeing scene isolation. By splitting on unique 
    Source Scene IDs rather than individual images, it ensures that no 
    background or lighting environment from training appears in the 
    evaluation sets, forcing the model to learn forensic features rather 
    than scene context.

USAGE:
    python scripts/create_splits.py data/manifests/df2023_manifest.csv

INPUTS       :
    - data/manifests/df2023_manifest.csv (Master manifest)

OUTPUTS      :
    - data/manifests/splits/train_split.csv
    - data/manifests/splits/val_split.csv
    - data/manifests/splits/test_split.csv

AUTHOR       : Dr. Samer Aoudi
AFFILIATION  : Higher Colleges of Technology (HCT), UAE
ROLE         : Assistant Professor & Division Chair (CIS)
EMAIL        : cybersecurity@sameraoudi.com
ORCID        : 0000-0003-3887-0119
CREATED      : 2026-01-11
UPDATED      : 2026-02-01

LICENSE      : MIT License
CITATION     : If used in academic research, please cite:
               Aoudi, S. (2026). "Beyond Accuracy — A Risk-Centric 
               Comparative Evaluation of Deep Intrusion Detection Systems."

DESIGN NOTES:
    - Leakage Prevention: Implements scene-level isolation to ensure 
      geographical/environmental independence between splits.
    - Robust Parsing: Uses a multi-tiered Scene ID extractor that parses 
      image_id strings as a fallback for missing metadata.
    - Statistical Integrity: Uses a fixed seed (1337) and a two-stage 
      train_test_split to maintain a strict 80/10/10 ratio on unique scenes.

DEPENDENCIES:
    - Python >= 3.10
    - pandas, numpy, scikit-learn
===============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Config
MANIFEST_PATH = "data/manifests/df2023_manifest.csv"
OUT_DIR = Path("data/manifests/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 1337

def get_scene_id(row):
    """
    Robustly extract scene ID from image_id.
    Format: COCO_DF_MethodID_SceneID -> returns SceneID
    """
    # 1. Trust existing columns if present
    if 'source_scene_id' in row and pd.notna(row['source_scene_id']): 
        return row['source_scene_id']
    if 'scene_id' in row and pd.notna(row['scene_id']): 
        return row['scene_id']
    
    # 2. Parse from image_id (e.g. COCO_DF_I000_00918198 -> 00918198)
    if 'image_id' in row:
        return str(row['image_id']).split('_')[-1]
        
    return "unknown"

def main():
    print(f"[*] Loading manifest: {MANIFEST_PATH}")
    if not Path(MANIFEST_PATH).exists():
        print(f"[ERROR] Manifest not found at {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    # 1. Extract Scene IDs
    print("[*] Extracting Scene IDs...")
    df['scene_id'] = df.apply(get_scene_id, axis=1)
    
    # 2. Get Unique Scenes
    unique_scenes = df['scene_id'].unique()
    print(f"[*] Found {len(df)} images from {len(unique_scenes)} unique scenes.")
    
    # 3. Perform Disjoint Split on SCENES (Strict 80/10/10)
    # Step A: Split Train (80%) vs Temp (20%)
    train_scenes, temp_scenes = train_test_split(unique_scenes, test_size=0.2, random_state=SEED)
    
    # Step B: Split Temp into Val (10%) and Test (10%)
    # Note: 0.5 of 20% = 10%
    val_scenes, test_scenes = train_test_split(temp_scenes, test_size=0.5, random_state=SEED)
    
    print(f"[*] Split Scenes Breakdown:")
    print(f"    Train Scenes: {len(train_scenes)}")
    print(f"    Val Scenes:   {len(val_scenes)}")
    print(f"    Test Scenes:  {len(test_scenes)}")
    
    # 4. Map Scenes back to Images
    train_df = df[df['scene_id'].isin(train_scenes)].copy()
    val_df   = df[df['scene_id'].isin(val_scenes)].copy()
    test_df  = df[df['scene_id'].isin(test_scenes)].copy()
    
    # 5. Save Splits
    print("[*] Saving CSVs...")
    train_df.to_csv(OUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUT_DIR / "val_split.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_split.csv", index=False)
    
    print(f"[OK] Done. Splits saved to {OUT_DIR}")
    print(f"    Train Images: {len(train_df)} ({(len(train_df)/len(df))*100:.1f}%)")
    print(f"    Val Images:   {len(val_df)} ({(len(val_df)/len(df))*100:.1f}%)")
    print(f"    Test Images:  {len(test_df)} ({(len(test_df)/len(df))*100:.1f}%)")

if __name__ == "__main__":
    main()
