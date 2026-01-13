#!/usr/bin/env python3
"""
===============================================================================
Script Name   : create_random_splits.py
Description   : Generates random splits.
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
