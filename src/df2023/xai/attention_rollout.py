"""
===============================================================================
Script Name   : attention_rollout.py
Description   : Robust Saliency & Attention approximation for Transformers.

                Methodology:
                - Primary: Computes 'Gradient * Input' (Input Saliency).
                - Rationale: Extracting raw Attention Matrices from compiled
                  backbones (like MiT-B2 in SegFormer) is brittle and often 
                  breaks between 'timm' versions. 
                - Validity: Grad*Input is a mathematically rigorous method 
                  to identify input sensitivity, serving as a reliable 
                  proxy for attention in "Black Box" scenarios.

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
- Robustness: Works for ANY differentiable model (CNN or Transformer).
- Normalization: We normalize by the max value to highlight the "Peak" 
  regions of interest relative to that specific image.
- Safety: Requires autograd to be enabled (even in eval mode) to compute
  the gradients w.r.t the input image.

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
    Acts as a universal fallback for Attention Rollout.
    """
    device = next(model.parameters()).device
    model.eval()

    # 1. Prepare Input (Requires Gradient)
    x = image.unsqueeze(0).to(device)  # (1, C, H, W)
    if not x.requires_grad:
        x.requires_grad_(True)

    # 2. Forward Pass
    logits = model(x)
    
    # 3. Determine Target Logic
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

    # 4. Backward Pass
    model.zero_grad(set_to_none=True)
    target.backward(retain_graph=False)

    # 5. Compute Saliency: |Gradient * Input|
    grad = x.grad.detach()
    sal = (grad * x.detach()).abs().sum(dim=1)[0] # Sum across RGB channels -> [H, W]
    
    # 6. Normalize [0, 1]
    sal_min = sal.min()
    sal_max = sal.max()
    if sal_max > sal_min:
        sal = (sal - sal_min) / (sal_max - sal_min + 1e-8)
        
    return sal
