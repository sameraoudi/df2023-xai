"""
===============================================================================
Script Name   : run_xai.py
Description   : CLI entry point for Phase 3: Explainable AI (XAI) Generation.
                
                Functionality:
                - Loads trained models and runs them on selected Test samples.
                - Generates Attribution Maps using:
                    * Integrated Gradients (IG) - Baseline faithfulness
                    * Grad-CAM++ - High-level localization
                    * Attention Rollout - Transformer-specific (SegFormer only)
                - Produces "Paper-Ready" composite panels:
                  [Original Image | Ground Truth Mask | Attribution Heatmap]

How to Run    :
                python -m df2023xai.cli.run_xai --config configs/xai_gen.yaml

Inputs        :
                --config : YAML config defining models, methods, and samples.

Outputs       :
                - outputs/xai/ panels (PNG images).
                - xai_summary.json (Runtime stats).

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
- Visualization: Uses 'Jet' colormap for heatmaps to maximize contrast for
  paper figures.
- Normalization: Heatmaps are min-max normalized per image to visualize
  relative importance within that specific sample.
- Architecture Matching: Reconstructs the exact model architecture from
  config before loading weights to prevent shape mismatches.

Dependencies  :
- Python >= 3.10
- torch, numpy, PIL, matplotlib
- df2023xai (Internal)
===============================================================================
"""

from __future__ import annotations
import os
import sys
import json
import typer
import torch
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from omegaconf import OmegaConf
from typing import Dict, Any

# Internal Imports
from df2023xai.data.dataset import ForgerySegDataset
import segmentation_models_pytorch as smp

# --- XAI Method Wrappers (Imports deferred to runtime to avoid overhead) ---
# form df2023xai.xai import gradcampp, ig, attention_rollout

app = typer.Typer(add_completion=False)


def _build_model_from_name(name: str, num_classes: int = 2) -> torch.nn.Module:
    """Reconstruct model architecture based on naming convention."""
    name = name.lower()
    if "segformer" in name:
        # e.g., segformer_b2 -> mit_b2
        encoder = name.replace("segformer_", "mit_")
        return smp.Segformer(encoder_name=encoder, classes=num_classes)
    elif "unet" in name:
        # e.g., unet_r34 -> resnet34
        encoder = "resnet34" 
        if "r50" in name: encoder = "resnet50"
        return smp.Unet(encoder_name=encoder, classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture in model name: {name}")


def _apply_colormap(heatmap: np.ndarray) -> Image.Image:
    """Convert 0..1 float heatmap to RGB image using Jet colormap."""
    # 1. Apply Colormap (returns RGBA [0..1])
    # 'jet' is standard for heatmaps; 'turbo' is clearer but less traditional.
    colored = cm.jet(heatmap)[:, :, :3] 
    
    # 2. Convert to Uint8 [0..255]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored)


def _create_panel(img_t: torch.Tensor, mask_t: torch.Tensor, heatmap_t: torch.Tensor) -> Image.Image:
    """
    Create a composite panel: [Original | Mask | Heatmap].
    Input Tensors: [C, H, W] or [H, W]
    """
    # 1. Prepare Original Image
    img_np = img_t.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    # 2. Prepare Ground Truth Mask (White on Black)
    mask_np = mask_t.cpu().numpy().astype(np.uint8) * 255
    pil_mask = Image.fromarray(mask_np, mode="L").convert("RGB")

    # 3. Prepare Heatmap
    # Normalize heatmap to 0..1 for visualization
    hm = heatmap_t.detach().cpu().numpy()
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    pil_heat = _apply_colormap(hm)

    # 4. Stitch Side-by-Side
    w, h = pil_img.size
    panel = Image.new("RGB", (w * 3, h))
    panel.paste(pil_img, (0, 0))
    panel.paste(pil_mask, (w, 0))
    panel.paste(pil_heat, (w * 2, 0))
    
    return panel


def _run_impl(cfg_path: str):
    if not os.path.exists(cfg_path):
        typer.echo(f"[ERROR] Config not found: {cfg_path}", err=True)
        sys.exit(1)

    cfg = OmegaConf.load(cfg_path)
    out_dir = cfg.get("out", {}).get("dir", "outputs/xai_panels")
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data (Validation Set)
    typer.echo(f"[*] Loading dataset: {cfg.data.manifest_csv}")
    ds = ForgerySegDataset(cfg.data.manifest_csv, split="val", img_size=cfg.data.img_size)
    
    # Select Samples (First N)
    n = min(len(ds), int(cfg.data.get("sample_n", 8)))
    indices = range(n)
    typer.echo(f"[*] Processing {n} samples...")

    # 2. Iterate Models
    for m_cfg in cfg.models:
        model_name = m_cfg["name"]
        ckpt_path = m_cfg["path"]
        
        typer.echo(f"[*] processing model: {model_name}")
        
        # Load Model
        try:
            model = _build_model_from_name(model_name).to(device)
            # Load weights (handle 'model' key if saved by loop.py)
            chk = torch.load(ckpt_path, map_location=device)
            state_dict = chk["model"] if "model" in chk else chk
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            typer.echo(f"[WARN] Failed to load {model_name}: {e}")
            continue

        # 3. Iterate XAI Methods
        methods = cfg.get("methods", {})
        
        # --- Grad-CAM++ ---
        if methods.get("gradcampp", {}).get("enabled", False):
            from df2023xai.xai.gradcampp import gradcampp_or_fallback
            for idx in indices:
                img, mask = ds[idx]
                heatmap = gradcampp_or_fallback(model, img.to(device))
                
                panel = _create_panel(img, mask, heatmap)
                panel.save(os.path.join(out_dir, f"{model_name}_sample{idx}_gradcampp.png"))

        # --- Integrated Gradients ---
        if methods.get("ig", {}).get("enabled", False):
            from df2023xai.xai.ig import integrated_gradients_minimal
            steps = methods.get("ig", {}).get("steps", 20)
            for idx in indices:
                img, mask = ds[idx]
                heatmap = integrated_gradients_minimal(model, img.to(device), steps=steps)
                
                panel = _create_panel(img, mask, heatmap)
                panel.save(os.path.join(out_dir, f"{model_name}_sample{idx}_ig.png"))

        # --- Attention Rollout (SegFormer Only) ---
        if methods.get("attention_rollout", {}).get("enabled", False) and "segformer" in model_name.lower():
            from df2023xai.xai.attention_rollout import attention_rollout_or_fallback
            for idx in indices:
                img, mask = ds[idx]
                heatmap = attention_rollout_or_fallback(model, img.to(device))
                
                panel = _create_panel(img, mask, heatmap)
                panel.save(os.path.join(out_dir, f"{model_name}_sample{idx}_attroll.png"))

    typer.echo(f"[OK] XAI Generation Complete → {out_dir}")


@app.command()
def run(config: str = typer.Option(..., "--config", help="Path to XAI config YAML")):
    _run_impl(config)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, config: str = typer.Option(None, "--config")):
    if ctx.invoked_subcommand is None and config:
        _run_impl(config)


if __name__ == "__main__":
    app()
