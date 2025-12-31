"""
===============================================================================
Script Name   : split_scenes.py
Description   : Generates scientifically rigorous, scene-disjoint data splits 
                for the DF2023 dataset. 
                
                Unlike random shuffling, this script splits data based on 
                'source_scene_id'. This ensures that all manipulations derived 
                from a specific background image (scene) reside in the same 
                partition (Train, Val, or Test), preventing "Background Leakage"
                where models memorize the background rather than the forgery.

How to Run    :
                python -m df2023xai.cli.split_scenes \
                    --manifest data/manifests/df2023_manifest.csv \
                    --outdir data/manifests/splits \
                    --seed 1337

Inputs        :
                --manifest : Path to the master CSV (from build_manifest.py)
                --outdir   : Directory to save the split CSVs
                --seed     : Random seed for reproducibility (Default: 1337)

Outputs       :
                - train_v2.csv, val_v2.csv, test_v2.csv
                - split_log_v2.json (Audit trail with SHA256 hashes)

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119
Created On    : 2025-Dec-31

License       : MIT License
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.

Design Notes  :
- Enforces strict disjointness: A scene ID found in Train will NEVER appear
  in Val or Test.
- Reproducibility: Uses a fixed numpy seed. The same seed always produces
  the exact same file lists.
- Auditability: Dumps a JSON log containing the SHA256 hash of the input
  manifest, ensuring the exact input data version is tracked.

Dependencies  :
- Python >= 3.10
- pandas, numpy
===============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Set

import numpy as np
import pandas as pd


# Columns required to perform a safe split
REQUIRED_COLS = {"image_id", "source_scene_id", "image_path", "mask_path", "manip_type"}


def sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    """Compute file hash for reproducibility logging."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_required(df: pd.DataFrame) -> List[str]:
    """Verify presence of metadata columns needed for stratification."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


def summarize(df: pd.DataFrame) -> Dict:
    """Generate summary statistics for the audit log."""
    out = {}
    out["n_rows"] = int(len(df))
    out["n_scenes"] = int(df["source_scene_id"].nunique())
    # Track class balance in this split
    out["manip_type_counts"] = (
        df["manip_type"].value_counts(dropna=False).sort_index().to_dict()
        if "manip_type" in df.columns
        else {}
    )
    return out


def split_scenes(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    """
    Core Logic: Split unique Scene IDs first, then assign rows based on Scene ID.
    This guarantees 0% leakage of background information.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1.0"

    # 1. Extract unique scenes
    scenes = df["source_scene_id"].drop_duplicates().values
    
    # 2. Shuffle scenes deterministically
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)

    n = len(scenes)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # 3. Calculate test remainder to avoid rounding errors dropping scenes
    n_test = n - n_train - n_val

    # 4. Partition Scene IDs
    train_scenes = set(scenes[:n_train])
    val_scenes = set(scenes[n_train : n_train + n_val])
    test_scenes = set(scenes[n_train + n_val :])

    # 5. Filter main dataframe
    d_train = df[df["source_scene_id"].isin(train_scenes)].copy()
    d_val = df[df["source_scene_id"].isin(val_scenes)].copy()
    d_test = df[df["source_scene_id"].isin(test_scenes)].copy()

    # 6. Verify Disjointness (The "Sleep Well at Night" Check)
    assert train_scenes.isdisjoint(val_scenes)
    assert train_scenes.isdisjoint(test_scenes)
    assert val_scenes.isdisjoint(test_scenes)

    return {"train": d_train, "val": d_val, "test": d_test}


def assert_disjoint_ids(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> None:
    """Double-check that no image OR scene leaks between splits."""
    def to_set(x: pd.DataFrame) -> Set:
        return set(x["image_id"].tolist())

    # Check Image ID leakage
    sa, sb, sc = to_set(a), to_set(b), to_set(c)
    if sa & sb or sa & sc or sb & sc:
        raise AssertionError("CRITICAL: Image overlap detected across splits.")
    
    # Check Scene ID leakage
    for col in ("source_scene_id",):
        pa, pb, pc = set(a[col].tolist()), set(b[col].tolist()), set(c[col].tolist())
        if pa & pb or pa & pc or pb & pc:
            raise AssertionError(f"CRITICAL: Scene overlap detected across splits on column={col}.")


def write_txt_csv(
    parts: Dict[str, pd.DataFrame],
    out_dir: str,
    suffix: str,
) -> Dict[str, Dict[str, str]]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, Dict[str, str]] = {}
    for name, df in parts.items():
        txt = os.path.join(out_dir, f"{name}_{suffix}.txt")
        csv = os.path.join(out_dir, f"{name}_{suffix}.csv")
        # Legacy TXT format (Image IDs only)
        df[["image_id"]].to_csv(txt, index=False, header=False)
        # Full CSV format (Standard usage)
        df.to_csv(csv, index=False)
        paths[name] = {"txt": txt, "csv": csv}
    return paths


def dist_table(df: pd.DataFrame, by: str = "manip_type") -> pd.DataFrame:
    if by not in df.columns:
        return pd.DataFrame()
    return df[by].value_counts().sort_index().rename("count").to_frame()


def main():
    ap = argparse.ArgumentParser(description="DF2023-XAI: Scene-Level Disjoint Splitter")
    ap.add_argument("--manifest", required=True, help="Path to master manifest CSV")
    ap.add_argument("--outdir", required=True, help="Directory to save split CSVs")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed (default: 1337)")
    ap.add_argument("--suffix", default=None, help="Custom suffix for output files (e.g., 'v1')")
    ap.add_argument("--train", type=float, default=0.8, help="Train ratio (default 0.8)")
    ap.add_argument("--val", type=float, default=0.1, help="Val ratio (default 0.1)")
    ap.add_argument("--test", type=float, default=0.1, help="Test ratio (default 0.1)")
    args = ap.parse_args()

    # Generate timestamped suffix if none provided to prevent accidental overwrites
    if args.suffix:
        suffix = args.suffix
    else:
        suffix = "v2_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load Data
    print(f"[*] Loading manifest: {args.manifest}")
    df = pd.read_csv(args.manifest)
    
    missing = check_required(df)
    if missing:
        print(f"[ERROR] Manifest missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    # 2. Sanity Check: Duplicates
    if df[["image_path", "mask_path"]].duplicated().any():
        print("[WARN] Duplicate file pairs found. Check build_manifest logic.", file=sys.stderr)

    # 3. Perform Split
    print(f"[*] Splitting scenes with seed={args.seed}...")
    parts = split_scenes(df, seed=args.seed, train_ratio=args.train, val_ratio=args.val, test_ratio=args.test)

    # 4. Verify Integrity
    assert_disjoint_ids(parts["train"], parts["val"], parts["test"])

    # 5. Save Outputs
    paths = write_txt_csv(parts, args.outdir, suffix)

    # 6. Generate Audit Log
    dist = {
        "train": summarize(parts["train"]),
        "val": summarize(parts["val"]),
        "test": summarize(parts["test"]),
        "all": summarize(df),
    }

    # Save distribution stats for the paper
    for k in ("train", "val", "test"):
        dt = dist_table(parts[k], by="manip_type")
        if not dt.empty:
            dt.to_csv(os.path.join(args.outdir, f"{k}_manip_type_{suffix}.csv"))

    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "input_manifest": os.path.abspath(args.manifest),
        "input_sha256": sha256_of_file(args.manifest),
        "output_dir": os.path.abspath(args.outdir),
        "outputs": paths,
        "distributions": dist,
        "notes": [
            "Scene-level disjointness enforced.",
            "Paths preserved exactly as in input manifest.",
        ],
        "cmdline": " ".join(sys.argv),
        "version": {"script": "split_scenes.py"},
    }
    
    log_path = os.path.join(args.outdir, f"split_log_{suffix}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # 7. Final Report
    print(f"[OK] Split Complete. Seed: {args.seed}")
    for k in ("train", "val", "test"):
        print(f"  {k.upper():<5} | Scenes: {dist[k]['n_scenes']:>6} | Images: {dist[k]['n_rows']:>7}")
    print(f"\nAudit Log: {log_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
