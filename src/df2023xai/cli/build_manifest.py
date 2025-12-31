"""
===============================================================================
Script Name   : build_manifest.py
Description   : CLI entry point for generating the master dataset manifest.
                This script scans raw image and mask directories, enforces
                strictly one-to-one pairing based on filenames, and generates
                a "Master CSV" containing all metadata (manipulation types,
                stratification details) prior to splitting.

How to Run    :
                python -m df2023xai.cli.build_manifest run \
                    --images /path/to/train/images \
                    --masks /path/to/train/masks \
                    --images-val /path/to/val/images \
                    --masks-val /path/to/val/masks \
                    --out data/manifests/df2023_manifest.csv

Inputs        :
                --images      : Path to training images directory
                --masks       : Path to training masks directory
                --images-val  : Path to validation images directory
                --masks-val   : Path to validation masks directory

Outputs       :
                --out         : CSV file containing the aggregated manifest
                                (e.g., data/manifests/df2023_manifest.csv)

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
- This script serves as the "Ground Truth" generator. It does not perform
  splitting; it only aggregates file paths and metadata.
- Strict 1:1 Image-to-Mask mapping is enforced to prevent data alignment
  errors during training.
- Does not modify source files; safe to run on read-only mounts.
- Automatically creates the output directory if it does not exist.

Dependencies  :
- Python >= 3.10
- typer
- df2023xai.data.manifest (Internal Module)
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
