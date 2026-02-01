"""
===============================================================================
PROJECT      : DF2023-XAI
SCRIPT       : run_forensic_eval.py
VERSION      : 1.0.0
DESCRIPTION  : Comparative forensic evaluation runner for Deepfake models.
-------------------------------------------------------------------------------
FUNCTIONALITY:
    Executes a multi-model evaluation pipeline based on a YAML configuration.
    It computes forensic metrics (e.g., pixel-level detection accuracy, 
    localization precision) across specified models, supports sampling for 
    quick audits, and generates aggregated CSV reports and visual summaries.

USAGE:
    python -m df2023xai.cli.run_forensic_eval --config configs/forensic_eval.yaml

CONFIG SCHEMA (YAML) REQUIREMENT:
    models:
      - {name: "ModelA", path: "path/to/weights.pt"}
    data:
      manifest_csv: "data/manifests/df2023_manifest.csv"
      sample_n: 500  # Set to 0 for full test set
    out:
      dir: "outputs/eval_results"

AUTHOR       : Dr. Samer Aoudi
AFFILIATION  : Higher Colleges of Technology (HCT), UAE
ROLE         : Assistant Professor & Division Chair (CIS)
EMAIL        : cybersecurity@sameraoudi.com
ORCID        : 0000-0003-3887-0119
CREATED      : 2026-01-05
UPDATED      : 2026-02-01

LICENSE      : MIT License
CITATION     : If used in academic research, please cite:
               Aoudi, S. (2026). "Beyond Accuracy — A Risk-Centric 
               Comparative Evaluation of Deep Intrusion Detection Systems."

DESIGN NOTES:
    - Configuration-Driven: Uses OmegaConf for flexible, nested parameters.
    - Reproducibility: Enforces a global seed (default: 1337) via internal 
      utility modules to ensure consistent sampling.
    - Hardware-Aware: Explicitly prefers CUDA for high-throughput evaluation.
    - Error Handling: Implements full traceback printing for fatal errors to 
      assist in debugging complex model loading issues.

DEPENDENCIES:
    - Python >= 3.10
    - typer, omegaconf, pathlib
    - df2023xai.eval.forensic (Internal)
    - df2023xai.utils.seed (Internal)
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
