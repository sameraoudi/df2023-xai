"""
===============================================================================
PROJECT      : DF2023-XAI
SCRIPT       : run_coherence_metric.py
VERSION      : 1.0.0
DESCRIPTION  : Quantitative XAI audit using Saliency Total Variation (TV).
-------------------------------------------------------------------------------
FUNCTIONALITY:
    Calculates the structural "coherence" of model explanations. By measuring 
    the Total Variation (TV) of input-gradient saliency maps, it quantifies 
    the smoothness of a model's focus. 
    - Lower TV: Suggests coherent, "blob-like" attention (SegFormer typical).
    - Higher TV: Suggests fragmented, "wireframe" attention (U-Net typical).

USAGE:
    python -m df2023xai.cli.run_coherence_metric --config configs/forensic_eval.yaml --limit 200

ARGUMENTS:
    --config      : (File) Path to the YAML configuration.
    --limit       : (Int) Maximum number of samples to audit (default: 200).

AUTHOR       : Dr. Samer Aoudi
AFFILIATION  : Higher Colleges of Technology (HCT), UAE
ROLE         : Assistant Professor & Division Chair (CIS)
EMAIL        : cybersecurity@sameraoudi.com
ORCID        : 0000-0003-3887-0119
CREATED      : 2026-01-28
UPDATED      : 2026-02-01

LICENSE      : MIT License
CITATION     : If used in academic research, please cite:
               Aoudi, S. (2026). "Beyond Accuracy — A Risk-Centric 
               Comparative Evaluation of Deep Intrusion Detection Systems."

DESIGN NOTES:
    - Architecture-Agnostic: Uses Input * Gradient instead of Layer-specific 
      methods (like Grad-CAM) to ensure fair comparison between CNNs and 
      Transformers (Vision Transformers/SegFormers).
    - TV Normalization: TV is normalized by pixel count, making the metric 
      resolution-invariant.
    - Resolution Support: Handles normalization of saliency maps to [0, 1] 
      to ensure a fair structured comparison across different model scales.

DEPENDENCIES:
    - Python >= 3.10
    - torch, pandas, omegaconf, tqdm
    - df2023xai.models.factory, df2023xai.data.dataset (Internal)
===============================================================================
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import os
from omegaconf import OmegaConf

from df2023xai.data.dataset import ForgerySegDataset
from df2023xai.models.factory import load_model_from_dir

def compute_saliency_tv(saliency_map):
    """
    Computes Total Variation (TV) of the saliency map.
    TV measures 'roughness'. 
    - Low TV = Coherent/Smooth (The "Blob") -> SegFormer Strength
    - High TV = Fragmented/Noisy (The "Wireframe") -> U-Net Weakness
    """
    # saliency_map: (H, W) tensor
    # Calculate vertical and horizontal gradients of the heatmap itself
    tv_h = torch.abs(saliency_map[1:, :] - saliency_map[:-1, :]).sum()
    tv_w = torch.abs(saliency_map[:, 1:] - saliency_map[:, :-1]).sum()
    
    # Normalize by number of pixels to be resolution-invariant
    return (tv_h + tv_w).item() / saliency_map.numel()

def get_input_gradient_saliency(model, img):
    """
    Robust, architecture-agnostic XAI.
    Calculates Input * Gradient (Saliency).
    Avoids 'target layer' issues common with Transformers.
    """
    img.requires_grad = True
    
    # Forward pass
    logits = model(img)
    
    # We want to explain the 'Forensic' class (Logit > 0)
    # Maximize the sum of logits (general sensitivity)
    score = logits.sum()
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Saliency = max over RGB channels of |Input * Gradient|
    # shape: (1, 3, H, W) -> (H, W)
    gradients = img.grad.data.abs()
    saliency = torch.max(gradients, dim=1)[0].squeeze()
    
    # Normalize to [0, 1] for fair TV comparison
    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
    return saliency.detach()

def evaluate_coherence(config_path, limit=200):
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    
    print(f"[*] Loading Test Set from: {cfg.data.manifest_csv}")
    ds = ForgerySegDataset(cfg.data.manifest_csv, cfg.data.img_size, split="test")
    
    # Subset for speed
    indices = torch.linspace(0, len(ds)-1, steps=limit).long()
    subset = Subset(ds, indices)
    dl = DataLoader(subset, batch_size=1, num_workers=2, shuffle=False)
    
    results = {'model': [], 'tv_score': []}
    
    for model_cfg in cfg.models:
        print(f"\n[*] Auditing Coherence (TV) for: {model_cfg.name}")
        try:
            model = load_model_from_dir(model_cfg.path).eval().cuda()
        except Exception as e:
            print(f"Failed to load {model_cfg.name}: {e}")
            continue
            
        total_tv = 0.0
        count = 0
        
        for img, mask in tqdm(dl):
            img = img.cuda()
            
            # Use Input Gradients (Robust) instead of Grad-CAM (Fragile)
            saliency = get_input_gradient_saliency(model, img)
            
            tv = compute_saliency_tv(saliency)
            
            total_tv += tv
            count += 1
            
        avg_tv = total_tv / count
        print(f"    -> Mean Total Variation (Lower is Better/Smoother): {avg_tv:.5f}")
        
        results['model'].append(model_cfg.name)
        results['tv_score'].append(avg_tv)

    df = pd.DataFrame(results)
    out_path = "xai_coherence_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[+] Saved coherence metrics to {out_path}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    
    evaluate_coherence(args.config, args.limit)
