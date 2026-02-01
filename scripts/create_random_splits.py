"""
===============================================================================
PROJECT      : DF2023-XAI (Explainable AI for Deepfake Detection)
SCRIPT       : create_random_splits.py
VERSION      : 1.0.0
DESCRIPTION  : Utility to generate randomized dataset splits (80/10/10).
-------------------------------------------------------------------------------
FUNCTIONALITY:
    Reads the master dataset manifest and partitions it into Training (80%), 
    Validation (10%), and Testing (10%) subsets. This ensures a standardized 
    data foundation for model benchmarking across different architectures.

USAGE:
    python scripts/create_random_splits.py

INPUTS       :
    - data/manifests/df2023_manifest.csv (Master manifest)

OUTPUTS      :
    - data/manifests/splits/train_split_random.csv
    - data/manifests/splits/val_split_random.csv
    - data/manifests/splits/test_split_random.csv

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
    - Reproducibility: Enforces a fixed random seed (42) to ensure that 
      splits remain consistent across different execution environments.
    - Shuffle: Dataset is shuffled prior to splitting to remove any 
      potential ordering bias from the master manifest.
    - Structure: Implements a two-stage split via scikit-learn to achieve 
      the desired 80/10/10 distribution.

DEPENDENCIES:
    - Python >= 3.10
    - pandas
    - scikit-learn
===============================================================================
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your full dataset
df = pd.read_csv("data/manifests/df2023_manifest.csv")

# Randomly split 80/10/10 (ignoring scene IDs)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# Save them
train_df.to_csv("data/manifests/splits/train_split_random.csv", index=False)
val_df.to_csv("data/manifests/splits/val_split_random.csv", index=False)
test_df.to_csv("data/manifests/splits/test_split_random.csv", index=False)

print("Random splits created in data/manifests/splits/")
