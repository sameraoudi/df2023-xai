#!/usr/bin/env python3
# ===============================================================================
# PROJECT      : DF2023-XAI (Explainable AI for Deepfake Detection)
# SCRIPT       : prepare_data.py
# VERSION      : 1.0.0
# DESCRIPTION  : Unified data preparation pipeline for the DF2023-XAI project.
# -------------------------------------------------------------------------------
# FUNCTIONALITY:
#     Combines three data preparation steps into a single entry point:
#     (1) add-scene-ids  — adds source_scene_id column to raw manifest
#     (2) scene-splits   — generates scene-disjoint 80/10/10 splits
#     (3) random-splits  — generates random 80/10/10 splits (ablation)
#     (4) all            — runs all three steps in sequence
# USAGE:
#     python scripts/prepare_data.py add-scene-ids  --manifest <path>
#     python scripts/prepare_data.py scene-splits   --manifest <path> --seed 1337
#     python scripts/prepare_data.py random-splits  --manifest <path> --seed 42
#     python scripts/prepare_data.py all            --manifest <path>
# INPUTS       :
#     - data/manifests/df2023_v15_manifest.csv  (add-scene-ids, scene-splits, random-splits)
# OUTPUTS      :
#     - data/manifests/df2023_v15_manifest.csv  (add-scene-ids, in-place)
#     - data/manifests/splits/train_split.csv
#     - data/manifests/splits/val_split.csv
#     - data/manifests/splits/test_split.csv
#     - data/manifests/splits/train_split_random.csv
#     - data/manifests/splits/val_split_random.csv
#     - data/manifests/splits/test_split_random.csv
# AUTHOR       : Dr. Samer Aoudi
# AFFILIATION  : Higher Colleges of Technology (HCT), UAE
# ROLE         : Assistant Professor & Division Chair (CIS)
# EMAIL        : cybersecurity@sameraoudi.com
# ORCID        : 0000-0003-3887-0119
# CREATED      : 2026-03-20
# UPDATED      : 2026-03-20
# LICENSE      : MIT License
# CITATION     : If used in academic research, please cite:
#                Aoudi, S. et al. (2026). "Beyond Accuracy: Auditing Image
#                Forgery Localization via Scene-Disjoint Evaluation and
#                XAI Faithfulness." IEEE Access.
# DESIGN NOTES:
#     - Each subcommand delegates to an isolated private function so
#       the logic of each original script is preserved and testable.
#     - All seeds, paths, and split ratios are explicit CLI flags with
#       defaults matching the original scripts.
#     - The 'all' subcommand runs stages in dependency order:
#       add-scene-ids must complete before splits are generated.
# DEPENDENCIES:
#     - Python >= 3.10
#     - pandas
#     - scikit-learn
# ===============================================================================

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# Used by scene-splits only (has source_scene_id + scene_id + image_id fallback)
def _get_scene_id_for_splits(row):
    """
    Robustly extract scene ID from image_id.
    Format: COCO_DF_MethodID_SceneID -> returns SceneID

    Verbatim from generate_scene_splits.py::get_scene_id.
    """
    # 1. Trust existing columns if present
    if "source_scene_id" in row and pd.notna(row["source_scene_id"]):
        return row["source_scene_id"]
    if "scene_id" in row and pd.notna(row["scene_id"]):
        return row["scene_id"]

    # 2. Parse from image_id (e.g. COCO_DF_I000_00918198 -> 00918198)
    if "image_id" in row:
        return str(row["image_id"]).split("_")[-1]

    return "unknown"


# Used by add-scene-ids only (preserves existing source_scene_id, no scene_id fallback)
def _get_scene_id_for_manifest(row):
    """
    Parses scene ID from image_id string.
    Format: COCO_DF_MethodID_SceneID -> returns SceneID
    Example: COCO_DF_I000B00000_00918198 -> 00918198

    Verbatim from preprocess_add_scene_ids.py::get_scene_id.
    """
    # If already exists, keep it
    if "source_scene_id" in row and pd.notna(row["source_scene_id"]):
        return row["source_scene_id"]

    # Parse from image_id
    if "image_id" in row:
        val = str(row["image_id"])
        # Split by underscore and take the last segment
        parts = val.split("_")
        if len(parts) > 1:
            return parts[-1]
    return "unknown"


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _add_scene_ids(args):
    """
    Logic verbatim from preprocess_add_scene_ids.py::main().
    """
    input_path = Path(args.manifest)
    output_path = Path(args.out) if args.out else input_path

    if not input_path.exists():
        print(f"[ERROR] Manifest not found: {input_path}")
        sys.exit(1)

    print(f"[*] Loading: {input_path}")
    df = pd.read_csv(input_path)

    print("[*] Extracting Scene IDs from filenames...")
    df["source_scene_id"] = df.apply(_get_scene_id_for_manifest, axis=1)

    # Validation stats
    n_unique = df["source_scene_id"].nunique()
    print(f"[*] Stats: {len(df)} images derived from {n_unique} unique scenes.")

    print(f"[*] Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print("[OK] Manifest updated successfully.")


def _generate_scene_splits(args):
    """
    Logic verbatim from generate_scene_splits.py::main().
    Hardcoded MANIFEST_PATH, OUT_DIR, SEED are replaced by args.
    """
    manifest_path = args.manifest
    out_dir = Path(args.outdir)
    seed = args.seed

    # Derive the two test_size values from the ratio flags
    # train=0.8, val=0.1, test=0.1  →  first split test_size = val+test = 0.2
    #                                   second split test_size = test/(val+test) = 0.5
    val_plus_test = args.val + args.test
    second_test_size = args.test / val_plus_test if val_plus_test > 0 else 0.5

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading manifest: {manifest_path}")
    if not Path(manifest_path).exists():
        print(f"[ERROR] Manifest not found at {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path)

    # 1. Extract Scene IDs
    print("[*] Extracting Scene IDs...")
    df["scene_id"] = df.apply(_get_scene_id_for_splits, axis=1)

    # 2. Get Unique Scenes
    unique_scenes = df["scene_id"].unique()
    print(f"[*] Found {len(df)} images from {len(unique_scenes)} unique scenes.")

    # 3. Perform Disjoint Split on SCENES (Strict 80/10/10)
    # Step A: Split Train (80%) vs Temp (20%)
    train_scenes, temp_scenes = train_test_split(
        unique_scenes, test_size=val_plus_test, random_state=seed
    )

    # Step B: Split Temp into Val (10%) and Test (10%)
    # Note: 0.5 of 20% = 10%
    val_scenes, test_scenes = train_test_split(
        temp_scenes, test_size=second_test_size, random_state=seed
    )

    print("[*] Split Scenes Breakdown:")
    print(f"    Train Scenes: {len(train_scenes)}")
    print(f"    Val Scenes:   {len(val_scenes)}")
    print(f"    Test Scenes:  {len(test_scenes)}")

    # 4. Map Scenes back to Images
    train_df = df[df["scene_id"].isin(train_scenes)].copy()
    val_df = df[df["scene_id"].isin(val_scenes)].copy()
    test_df = df[df["scene_id"].isin(test_scenes)].copy()

    # 5. Save Splits
    print("[*] Saving CSVs...")
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    test_df.to_csv(out_dir / "test_split.csv", index=False)

    print(f"[OK] Done. Splits saved to {out_dir}")
    print(f"    Train Images: {len(train_df)} ({(len(train_df)/len(df))*100:.1f}%)")
    print(f"    Val Images:   {len(val_df)} ({(len(val_df)/len(df))*100:.1f}%)")
    print(f"    Test Images:  {len(test_df)} ({(len(test_df)/len(df))*100:.1f}%)")


def _generate_random_splits(args):
    """
    Logic verbatim from generate_random_splits.py (module-level code).
    Hardcoded paths and random_state=42 are replaced by args.
    """
    out_dir = Path(args.outdir)

    # Derive the two test_size values from the ratio flags (same logic as scene-splits)
    val_plus_test = args.val + args.test
    second_test_size = args.test / val_plus_test if val_plus_test > 0 else 0.5

    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.manifest).exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        sys.exit(1)

    # Load your full dataset
    df = pd.read_csv(args.manifest)

    # Randomly split 80/10/10 (ignoring scene IDs)
    train_df, temp_df = train_test_split(
        df, test_size=val_plus_test, random_state=args.seed, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=second_test_size, random_state=args.seed, shuffle=True
    )

    # Save them
    train_df.to_csv(out_dir / "train_split_random.csv", index=False)
    val_df.to_csv(out_dir / "val_split_random.csv", index=False)
    test_df.to_csv(out_dir / "test_split_random.csv", index=False)

    print(f"Random splits created in {out_dir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="prepare_data.py",
        description=(
            "Unified data preparation pipeline for DF2023-XAI.\n"
            "Subcommands: add-scene-ids, scene-splits, random-splits, all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------
    # Subcommand: add-scene-ids
    # ------------------------------------------------------------------
    p_add = subparsers.add_parser(
        "add-scene-ids",
        help="Add source_scene_id column to manifest (idempotent).",
        description=(
            "Reads the df2023 manifest CSV, derives the source scene ID for each "
            "image by parsing the trailing segment of the image_id field "
            "(format: COCO_DF_MethodID_SceneID → SceneID), and writes the "
            "augmented manifest back to disk."
        ),
    )
    p_add.add_argument(
        "--manifest",
        default="data/manifests/df2023_manifest.csv",
        help="Path to input manifest CSV (default: data/manifests/df2023_manifest.csv)",
    )
    p_add.add_argument(
        "--out",
        default=None,
        help="Path to output manifest (default: overwrite --manifest in-place)",
    )

    # ------------------------------------------------------------------
    # Subcommand: scene-splits
    # ------------------------------------------------------------------
    p_scene = subparsers.add_parser(
        "scene-splits",
        help="Generate scene-disjoint 80/10/10 train/val/test splits.",
        description=(
            "Loads the full df2023 manifest CSV, extracts source scene IDs, "
            "splits unique scenes 80/10/10, and maps scenes back to images — "
            "guaranteeing no scene appears in multiple partitions."
        ),
    )
    p_scene.add_argument(
        "--manifest",
        default="data/manifests/df2023_manifest.csv",
        help="Path to input manifest CSV (default: data/manifests/df2023_manifest.csv)",
    )
    p_scene.add_argument(
        "--outdir",
        default="data/manifests/splits",
        help="Output directory for split CSVs (default: data/manifests/splits)",
    )
    p_scene.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible scene assignment (default: 1337)",
    )
    p_scene.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Fraction of scenes for training split (default: 0.8)",
    )
    p_scene.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Fraction of scenes for validation split (default: 0.1)",
    )
    p_scene.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Fraction of scenes for test split (default: 0.1)",
    )

    # ------------------------------------------------------------------
    # Subcommand: random-splits
    # ------------------------------------------------------------------
    p_random = subparsers.add_parser(
        "random-splits",
        help="Generate random (non-scene-disjoint) 80/10/10 splits (ablation).",
        description=(
            "Loads the full df2023 manifest CSV and performs a purely random "
            "80/10/10 split (no scene-disjointness constraint). Used for "
            "ablation comparisons against the scene-disjoint methodology."
        ),
    )
    p_random.add_argument(
        "--manifest",
        default="data/manifests/df2023_manifest.csv",
        help="Path to input manifest CSV (default: data/manifests/df2023_manifest.csv)",
    )
    p_random.add_argument(
        "--outdir",
        default="data/manifests/splits",
        help="Output directory for split CSVs (default: data/manifests/splits)",
    )
    p_random.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    p_random.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Fraction of images for training split (default: 0.8)",
    )
    p_random.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Fraction of images for validation split (default: 0.1)",
    )
    p_random.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Fraction of images for test split (default: 0.1)",
    )

    # ------------------------------------------------------------------
    # Subcommand: all
    # ------------------------------------------------------------------
    p_all = subparsers.add_parser(
        "all",
        help="Run all three stages in sequence: add-scene-ids → scene-splits → random-splits.",
        description=(
            "Runs add-scene-ids, scene-splits, and random-splits in dependency "
            "order. Accepts the union of all flags from all three subcommands."
        ),
    )
    p_all.add_argument(
        "--manifest",
        default="data/manifests/df2023_manifest.csv",
        help="Path to manifest CSV used by all three stages "
        "(default: data/manifests/df2023_manifest.csv)",
    )
    p_all.add_argument(
        "--out",
        default=None,
        help="Output path for the scene-id-augmented manifest "
        "(default: overwrite --manifest in-place)",
    )
    p_all.add_argument(
        "--outdir",
        default="data/manifests/splits",
        help="Output directory for all split CSVs (default: data/manifests/splits)",
    )
    # scene-splits seed
    p_all.add_argument(
        "--scene-seed",
        type=int,
        default=1337,
        dest="scene_seed",
        help="Random seed for scene-disjoint splits (default: 1337)",
    )
    # random-splits seed
    p_all.add_argument(
        "--random-seed",
        type=int,
        default=42,
        dest="random_seed",
        help="Random seed for random splits (default: 42)",
    )
    p_all.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="Train fraction for both split types (default: 0.8)",
    )
    p_all.add_argument(
        "--val",
        type=float,
        default=0.1,
        help="Val fraction for both split types (default: 0.1)",
    )
    p_all.add_argument(
        "--test",
        type=float,
        default=0.1,
        help="Test fraction for both split types (default: 0.1)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand == "add-scene-ids":
        _add_scene_ids(args)

    elif args.subcommand == "scene-splits":
        _generate_scene_splits(args)

    elif args.subcommand == "random-splits":
        _generate_random_splits(args)

    elif args.subcommand == "all":
        # Stage 1
        print("[1/3] Adding scene IDs...")
        _add_scene_ids(args)
        print()

        # Stage 2 — build a namespace matching what _generate_scene_splits expects
        print("[2/3] Generating scene-disjoint splits...")
        scene_args = argparse.Namespace(
            manifest=args.manifest,
            outdir=args.outdir,
            seed=args.scene_seed,
            train=args.train,
            val=args.val,
            test=args.test,
        )
        _generate_scene_splits(scene_args)
        print()

        # Stage 3 — build a namespace matching what _generate_random_splits expects
        print("[3/3] Generating random splits...")
        random_args = argparse.Namespace(
            manifest=args.manifest,
            outdir=args.outdir,
            seed=args.random_seed,
            train=args.train,
            val=args.val,
            test=args.test,
        )
        _generate_random_splits(random_args)
        print()

        print("[OK] All stages complete.")


if __name__ == "__main__":
    main()
