"""
===============================================================================
Script Name   : run_xai.py
Description   : CLI entry point for Phase 3: Explainable AI (XAI) Generation.
                
                Functionality:
                - Loads trained models (SegFormer/U-Net) and runs inference.
                - Generates Attribution Maps using 4 supported methods:
                    1. Grad-CAM++ (Localization)
                    2. Integrated Gradients (Axiomatic)
                    3. Attention Rollout / Saliency (Transformer Focus)
                    4. SHAP (Game Theoretic)
                - Produces composite panels: [Original | Mask | Heatmap].

How to Run    :
                python -m df2023xai.cli.run_xai --config configs/xai_gen.yaml

Inputs        :
                --config : YAML config defining models, methods, and samples.

Outputs       :
                - outputs/xai_panels/ (PNG images).

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
- Visualization: Uses 'Jet' colormap for high-contrast heatmaps.
- Normalization: Per-image min-max normalization.
- Lazy Imports: XAI methods are imported inside the loop to prevent
  circular dependencies and reduce startup time.

Dependencies  :
- Python >= 3.10
- torch, numpy, PIL, matplotlib
- df2023xai (Internal)
===============================================================================
"""

import os, json, sys, typer, torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from ..models.factory import load_model_from_dir
from ..data.dataset import ForgerySegDataset
# Import the fixed XAI function
from ..xai.shap import shap_or_fallback 

app = typer.Typer(add_completion=False)

def _to_pil_heatmap(img_t: torch.Tensor, heat: torch.Tensor):
    # Normalize Image for background
    rgb = (img_t.clone().clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    
    # Normalize Heatmap to Red Overlay
    h = (heat.clone().clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
    # Create Red channel overlay: (R=Heat, G=0, B=0)
    h_colored = np.stack([h, np.zeros_like(h), np.zeros_like(h)], axis=-1)
    
    # Blend: 60% Original + 40% Heatmap
    out = (0.6 * rgb + 0.4 * h_colored).clip(0,255).astype(np.uint8)
    return Image.fromarray(out)

def _run_impl(cfg_path: str):
    print(f"[XAI] Loading config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    os.makedirs(cfg.out.dir, exist_ok=True)

    # FIX 1: Correct Argument Order (Manifest, Size, Split)
    # dataset.py signature: (manifest_csv, img_size, split, aug_cfg)
    ds = ForgerySegDataset(
        manifest_csv=cfg.data.manifest_csv, 
        img_size=cfg.data.img_size,
        split="val" # Use validation set for audit
    )
    
    n = min(len(ds), int(cfg.data.get("sample_n", 8)))
    print(f"[XAI] Generating explanations for {n} samples...")
    
    # Pre-load samples to avoid reloading dataset
    samples = []
    for i in range(n):
        samples.append(ds[i]) # Returns (img, mask)

    results = {}
    
    for m in cfg.models:
        model_dir = m["path"]
        try:
            model = load_model_from_dir(model_dir).eval()
        except Exception as e:
            print(f"[Error] Failed to load {model_dir}: {e}")
            continue
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model_name = os.path.basename(os.path.dirname(model_dir))
        results[model_name] = {}
        print(f"[XAI] Processing Model: {model_name} on {device}")

        # --- Grad-CAM++ ---
        if cfg.methods.get("gradcampp", {}).get("enabled", False):
            from ..xai.gradcampp import gradcampp_or_fallback
            for k, (img, mask) in enumerate(samples): # FIX 2: Unpack 2 items
                heat = gradcampp_or_fallback(model, img)
                fname = f"{model_name}_gradcampp_{k}.png"
                _to_pil_heatmap(img, heat).save(os.path.join(cfg.out.dir, fname))

        # --- Integrated Gradients ---
        if cfg.methods.get("ig", {}).get("enabled", False):
            from ..xai.ig import integrated_gradients_minimal
            for k, (img, mask) in enumerate(samples):
                heat = integrated_gradients_minimal(model, img, steps=16)
                fname = f"{model_name}_ig_{k}.png"
                _to_pil_heatmap(img, heat).save(os.path.join(cfg.out.dir, fname))

        # --- Attention Rollout ---
        if cfg.methods.get("attention_rollout", {}).get("enabled", False):
            from ..xai.attention_rollout import attention_rollout_or_fallback
            for k, (img, mask) in enumerate(samples):
                heat = attention_rollout_or_fallback(model, img)
                fname = f"{model_name}_attroll_{k}.png"
                _to_pil_heatmap(img, heat).save(os.path.join(cfg.out.dir, fname))

        # --- SHAP (Now Enabled) ---
        if cfg.methods.get("shap", {}).get("enabled", False):
            print(f"  > Running SHAP (Warning: Slow)...")
            for k, (img, mask) in enumerate(samples):
                try:
                    # Uses the fixed shap.py with dynamic channel selection
                    heat = shap_or_fallback(model, img) 
                    fname = f"{model_name}_shap_{k}.png"
                    _to_pil_heatmap(img, heat).save(os.path.join(cfg.out.dir, fname))
                except Exception as e:
                    print(f"    [!] SHAP failed for sample {k}: {e}")

    with open(os.path.join(cfg.out.dir, "xai_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] XAI panels saved to {cfg.out.dir}")

@app.command()
def run(config: str = typer.Option(..., "--config", help="Path to XAI yaml")):
    _run_impl(config)

if __name__ == "__main__":
    app()
