"""
===============================================================================
Script Name   : run_forensic_eval.py
Description   : CLI entry point for Forensic Evaluation (Phase 2).
                
                Functionality:
                - Loads trained models (SegFormer/U-Net) from checkpoints.
                - Runs inference on the Test Set (defined in manifest).
                - Computes standard forensic metrics:
                    * IoU (Intersection over Union)
                    * Dice Coefficient (F1 Score)
                    * Pixel-level Precision/Recall
                - Generates "metrics_summary.csv" for the paper.

How to Run    :
                # Full Evaluation (Recommended for Paper)
                python -m df2023xai.cli.run_forensic_eval --config configs/forensic_eval.yaml

Inputs        :
                --config : YAML config defining models and test data.
                           (Must contain 'models' list and 'data.manifest_csv')

Outputs       :
                - metrics_summary.csv : Table comparing all models.
                - detailed_report.csv : Per-image scores (optional).

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
- Benchmarking: Supports evaluating multiple models in one run to generate
  comparative tables directly.
- Sampling: Set 'sample_n: 0' in YAML to evaluate the ENTIRE test set.
  For debugging, use 'sample_n: 64'.
- Reproducibility: Enforces seeding even during inference to ensure
  consistent data sampling.

Dependencies  :
- Python >= 3.10
- typer, omegaconf
- df2023xai.eval.forensic (Internal)
===============================================================================
"""

from __future__ import annotations
import os
import sys
import typer
from omegaconf import OmegaConf
from pathlib import Path

# Internal Imports
from df2023xai.eval.forensic import write_forensic_reports
from df2023xai.utils.seed import set_global_seed

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _run_impl(cfg_path: str):
    """
    Execute the evaluation pipeline based on the provided YAML config.
    """
    if not os.path.exists(cfg_path):
        typer.echo(f"[ERROR] Config file not found: {cfg_path}", err=True)
        sys.exit(1)

    # 1. Load Configuration
    cfg = OmegaConf.load(cfg_path)
    
    # 2. Setup Output Directory
    out_dir = cfg.get("out", {}).get("dir", "outputs/eval_results")
    os.makedirs(out_dir, exist_ok=True)

    # 3. Reproducibility
    seed = cfg.get("seed", 1337)
    set_global_seed(seed)

    # 4. Parse Model List
    # Expecting config to look like:
    # models:
    #   - name: "SegFormer-B2"
    #     path: "outputs/segformer/best.pt"
    #   - name: "U-Net-R34"
    #     path: "outputs/unet/best.pt"
    models_list = cfg.get("models", [])
    if not models_list:
        typer.echo("[ERROR] No models found in config. Please specify 'models' list.", err=True)
        sys.exit(1)

    # 5. Determine Sample Size
    # sample_n <= 0 implies "Run on everything"
    sample_n = int(cfg.get("data", {}).get("sample_n", 0))
    sample_msg = f"{sample_n}" if sample_n > 0 else "FULL TEST SET"

    typer.echo(f"[*] Starting Forensic Evaluation")
    typer.echo(f"    - Output: {out_dir}")
    typer.echo(f"    - Models: {[m['name'] for m in models_list]}")
    typer.echo(f"    - Scope:  {sample_msg}")

    # 6. Run Evaluation Logic
    try:
        df = write_forensic_reports(
            models=models_list,
            manifest_csv=cfg.data.manifest_csv,
            img_size=cfg.data.get("img_size", 512),
            out_dir=out_dir,
            sample_n=sample_n,
            device="cuda"  # Explicitly prefer GPU
        )
        typer.echo(f"[OK] Evaluation Complete.")
        typer.echo(f"     Metrics Summary → {out_dir}/metrics_summary.csv")
    except Exception as e:
        typer.echo(f"[FATAL] Evaluation failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@app.command()
def run(config: str = typer.Option(..., "--config", help="Path to evaluation config (YAML)")):
    """Run forensic evaluation on the test set."""
    _run_impl(config)


# Backwards compatibility for running without sub-command
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, config: str = typer.Option(None, "--config")):
    if ctx.invoked_subcommand is None and config:
        _run_impl(config)


if __name__ == "__main__":
    app()
