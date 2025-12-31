"""
===============================================================================
Script Name   : attention_rollout.py
Description   : Robust Explainability Module for Transformer Architectures.
                
                Methodology:
                1. Primary Strategy: Gradient * Input (Saliency).
                   We use this as the robust "Universal Fallback" for SegFormer
                   because extracting raw attention weights from compiled 
                   backbones (like MiT-B2) often breaks across library versions.
                2. Scientific Validity: Grad*Input is a standard "White Box" 
                   method that highlights pixels which, if changed, would most 
                   affect the classification.

Inputs        :
                - model     : Trained SegFormer/U-Net (eval mode).
                - image     : Input tensor [C, H, W].
                - class_idx : Target class (default: Forgery).

Outputs       :
                - heatmap   : Tensor [H, W] normalized to [0, 1].

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
- Robustness: Unlike hook-based Attention Rollout which often fails on 
  different 'timm' versions, this Gradient-based approach guarantees a 
  valid heatmap for ANY differentiable model.
- Normalization: We normalize by the max value to highlight the "Peak" 
  regions of interest relative to that specific image.

Dependencies  :
- Python >= 3.10
- torch
===============================================================================
"""

from __future__ import annotations
import torch
from typing import Optional

def attention_rollout_or_fallback(
    model: torch.nn.Module,
    image: torch.Tensor,
    class_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Computes a 'Gradient * Input' Saliency Map.
    
    Why this instead of raw Rollout? 
    Raw Attention Rollout requires hooking specific internal layers of the 
    Vision Transformer encoder (MiT-B2). These layer names change frequently 
    between 'timm' versions, making the code brittle.
    
    Grad*Input is the mathematically stable equivalent for identifying 
    input sensitivity in Transformers.
    """
    device = next(model.parameters()).device
    model.eval()

    # Prepare Input
    x = image.unsqueeze(0).to(device)  # (1, C, H, W)
    x.requires_grad_(True)

    # Forward Pass
    logits = model(x)
    
    # Determine Target Logic
    if logits.ndim == 4:
        # Segmentation: [1, Classes, H, W]
        # Target the mean of the "Forgery" class (Index 1)
        if class_idx is None:
            target_idx = 1 if logits.shape[1] > 1 else 0
            target = logits[:, target_idx:target_idx+1].mean()
        else:
            target = logits[:, class_idx:class_idx+1].mean()
    else:
        # Classification Fallback
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        target = logits[0, class_idx]

    # Backward Pass
    model.zero_grad(set_to_none=True)
    target.backward(retain_graph=False)

    # Compute Saliency: |Gradient * Input|
    grad = x.grad.detach()
    sal = (grad * x.detach()).abs().sum(dim=1)[0] # Sum across RGB channels -> [H, W]
    
    # Normalize [0, 1]
    sal_min = sal.min()
    sal_max = sal.max()
    if sal_max > sal_min:
        sal = (sal - sal_min) / (sal_max - sal_min + 1e-8)
        
    return sal
