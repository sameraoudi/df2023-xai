"""
===============================================================================
PROJECT      : DF2023-XAI
SCRIPT       : build_manifest.py
VERSION      : 1.0.0
DESCRIPTION  : CLI entry point for generating the master dataset manifest.
-------------------------------------------------------------------------------
FUNCTIONALITY:
    Scans raw image and mask directories (supporting symlinks), parses 
    filenames for manipulation metadata, and aggregates paths into a 
    "Master CSV". This serves as the single source of truth for the 
    training and validation pipelines.

USAGE:
    python -m df2023xai.cli.build_manifest run \
        --images /path/to/train/images \
        --masks /path/to/train/masks \
        --images-val /path/to/val/images \
        --masks-val /path/to/val/masks \
        --out data/manifests/df2023_manifest.csv \
        [--enforce-one-to-one | --no-enforce-one-to-one]

ARGUMENTS:
    --images      : (Dir) Path to training images (symlinks allowed).
    --masks       : (Dir) Path to training masks (symlinks allowed).
    --images-val  : (Dir) Path to validation images.
    --masks-val   : (Dir) Path to validation masks.
    --out         : (File) Target path for the generated Master CSV.
    --enforce-one-to-one : (Bool) If True (default), script exits on 
                           image/mask count mismatch.

AUTHOR       : Dr. Samer Aoudi
AFFILIATION  : Higher Colleges of Technology (HCT), UAE
ROLE         : Assistant Professor & Division Chair (CIS)
EMAIL        : cybersecurity@sameraoudi.com
ORCID        : 0000-0003-3887-0119
CREATED      : 2025-12-31
UPDATED      : 2026-02-01

LICENSE      : MIT License
CITATION     : If used in academic research, please cite:
               Aoudi, S. (2026). "Beyond Accuracy — A Risk-Centric 
               Comparative Evaluation of Deep Intrusion Detection Systems."

DESIGN NOTES:
    - Ground Truth Generator: Centralizes metadata prior to splitting.
    - Robustness: Uses absolute imports for cross-context execution.
    - Safety: Gracefully handles RuntimeErrors and fatal exceptions 
      with distinct exit codes (1 and 2).
    - Automation: Automatically handles directory creation for the output path.

DEPENDENCIES:
    - Python >= 3.10
    - typer
    - df2023xai.data.manifest (Internal)
===============================================================================
"""
from __future__ import annotations

import sys
import typer
from typer import Option

# Absolute import for robustness across execution contexts
from df2023xai.data.manifest import write_manifest

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def run(
    images: str = Option(..., "--images", help="Path to training images directory (symlinks allowed)"),
    masks: str = Option(..., "--masks", help="Path to training masks directory (symlinks allowed)"),
    images_val: str = Option(..., "--images-val", help="Path to validation images directory"),
    masks_val: str = Option(..., "--masks-val", help="Path to validation masks directory"),
    out: str = Option(..., "--out", help="Output path for the master manifest CSV"),
    enforce_one_to_one: bool = Option(True, "--enforce-one-to-one/--no-enforce-one-to-one", help="Crash if image/mask counts mismatch"),
):
    """
    Scan DF2023 directories (images/masks), parse filenames for manipulation
    metadata, and write the Master Manifest CSV.
    """
    try:
        # Note: write_manifest handles os.makedirs(..., exist_ok=True) internally
        write_manifest(
            train_images=images,
            train_masks=masks,
            val_images=images_val,
            val_masks=masks_val,
            out_csv=out,
            enforce_one_to_one=enforce_one_to_one
        )
        typer.echo(f"[OK] Master Manifest successfully written → {out}")
        
    except RuntimeError as e:
        typer.echo(f"[ERROR] Manifest generation failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"[FATAL] Unexpected error: {e}", err=True)
        sys.exit(2)


if __name__ == "__main__":
    app()
