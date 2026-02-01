"""
===============================================================================
PROJECT      : DF2023-XAI
SCRIPT       : run_xai.py
VERSION      : 1.0.0
DESCRIPTION  : XAI interpretation launcher for Deepfake segmentation models.
-------------------------------------------------------------------------------
FUNCTIONALITY:
    Generates visual explanations for model predictions using multiple XAI 
    techniques. It supports Grad-CAM++, Integrated Gradients (IG), Attention 
    Rollout, and SHAP. The script produces JET-colormapped heatmaps overlaid 
    on source images to audit where the model "looks" when identifying forgeries.

USAGE:
    python -m df2023xai.cli.run_xai --config configs/xai_gen.yaml

ARGUMENTS:
    --config      : (File) Path to the XAI configuration (YAML) specifying 
                    models, target samples, and enabled XAI methods.

AUTHOR       : Dr. Samer Aoudi
AFFILIATION  : Higher Colleges of Technology (HCT), UAE
ROLE         : Assistant Professor & Division Chair (CIS)
EMAIL        : cybersecurity@sameraoudi.com
ORCID        : 0000-0003-3887-0119
CREATED      : 2026-01-15
UPDATED      : 2026-02-01

LICENSE      : MIT License
CITATION     : If used in academic research, please cite:
               Aoudi, S. (2026). "Beyond Accuracy — A Risk-Centric 
               Comparative Evaluation of Deep Intrusion Detection Systems."

DESIGN NOTES:
    - Visualization: Uses a 60/40 alpha blend between source RGB and JET 
      heatmaps. Robust min-max normalization handles faint saliency signals.
    - Diversity: Supports both gradient-based (Grad-CAM++, IG) and 
      perturbation-based (SHAP) explanations.
    - Safety: Implements fallbacks for SHAP and Grad-CAM++ to prevent 
      crashes during batch processing of diverse model architectures.
    - Audit Trail: Outputs an 'xai_summary.json' alongside PNG panels.

DEPENDENCIES:
    - Python >= 3.10
    - torch, PIL, opencv-python, omegaconf
    - df2023xai.models.factory, df2023xai.xai.* (Internal)
===============================================================================
"""

import os, json, sys, typer, torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import cv2
import torch
from ..models.factory import load_model_from_dir
from ..data.dataset import ForgerySegDataset
# Import the fixed XAI function
from ..xai.shap import shap_or_fallback 

app = typer.Typer(add_completion=False)

def _to_pil_heatmap(img_t: torch.Tensor, heat: torch.Tensor):
    """
    Robust Visualization using JET Colormap.
    img_t: (3, H, W) tensor, [0, 1]
    heat: (H, W) tensor, [0, 1]
    """
    # 1. Prepare Background Image (RGB)
    # Convert Tensor [0,1] -> Numpy [0,255] uint8
    img_np = (img_t.clone().clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    
    # 2. Prepare Heatmap
    # Normalize heatmap strictly to [0, 255] for visualization
    h_np = heat.clone().detach().cpu().numpy()
    
    # Robust Min-Max normalization to handle faint signals
    if h_np.max() > h_np.min():
        h_np = (h_np - h_np.min()) / (h_np.max() - h_np.min())
    else:
        h_np = np.zeros_like(h_np) # Flat signal
        
    h_uint8 = (h_np * 255).astype(np.uint8)

    # 3. Apply JET Colormap (Blue=Low, Red=High)
    # OpenCV expects BGR for the underlying image, so we convert slightly for the op
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)

    # 4. Blend (60% Source + 40% Heatmap)
    overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heatmap_bgr, 0.4, 0)
    
    # 5. Convert back to RGB for PIL
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(overlay_rgb)

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
