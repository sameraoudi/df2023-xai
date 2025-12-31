"""
===============================================================================
Script Name   : gradcampp.py
Description   : Implementation of Grad-CAM++ (Gradient-weighted Class Activation 
                Mapping Plus Plus) for Semantic Segmentation.
                
                Methodology:
                - Computes weighted combination of positive partial derivatives
                  w.r.t. the last convolutional feature map.
                - Superior to vanilla Grad-CAM for localizing multiple instances
                  of the same class (crucial for multi-forgery images).
                
                Compatibility:
                - CNNs (U-Net, ResNet): Native support.
                - Transformers (SegFormer): Delegates to Attention Rollout.

Inputs        :
                - model     : Trained PyTorch model (eval mode).
                - image     : Input tensor [C, H, W].
                - class_idx : Target class index (1 = Forgery).

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
- Backward Compatibility: Uses 'register_full_backward_hook' for modern PyTorch.
- Fallback Mechanism: If no Convolutional layer is found (e.g. pure ViT),
  it attempts to load the Attention Rollout module instead.
- Optimization: Hooks are registered temporarily and removed immediately
  to prevent memory leaks during batch processing.

Dependencies  :
- Python >= 3.10
- torch, torch.nn.functional
===============================================================================
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

def _find_last_conv(m: nn.Module) -> Optional[nn.Conv2d]:
    """
    Recursively find the last Conv2d layer in a generic model.
    For U-Net, this usually identifies the final segmentation head.
    """
    last = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    return last


def gradcampp_or_fallback(
    model: nn.Module, 
    image: torch.Tensor, 
    class_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Generate Grad-CAM++ heatmap. 
    If model is a Transformer (no Conv2d), falls back to Attention Rollout.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 1. Identify Target Layer
    target_layer = _find_last_conv(model)
    
    # 2. Transformer Fallback
    if target_layer is None:
        # If no Conv2d found, assume it's a Transformer and switch strategies
        # Import inside function to avoid circular dependency at module level
        try:
            from df2023xai.xai.attention_rollout import attention_rollout_or_fallback
            return attention_rollout_or_fallback(model, image, class_idx)
        except ImportError:
            # If fallback unavailable, return empty heatmap
            return torch.zeros(image.shape[-2:], device=device)

    # 3. Register Hooks
    feats = {}
    grads = {}

    def fwd_hook(module, inputs, outputs):
        feats["y"] = outputs

    def bwd_hook(module, grad_in, grad_out):
        # grad_out is a tuple, take index 0
        grads["g"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # 4. Forward Pass
    # Prepare input [1, C, H, W] requiring gradient
    x = image.unsqueeze(0).to(device)
    if not x.requires_grad:
        x.requires_grad_(True)
        
    logits = model(x)
    
    # 5. Determine Target Score
    # For segmentation, we usually care about the mean score of the "Forgery" class (idx=1)
    # logits shape: [1, NumClasses, H, W]
    if class_idx is None:
        # Default: Focus on class 1 (Forgery) if binary, else max class
        target_class = 1 if logits.shape[1] > 1 else 0
    else:
        target_class = class_idx

    # Aggregate spatial logits to get a scalar for backprop
    score = logits[:, target_class, :, :].mean()

    # 6. Backward Pass
    model.zero_grad()
    score.backward()

    # 7. Compute Grad-CAM++ Weights
    # A: Activations [1, C, H, W]
    # G: Gradients [1, C, H, W]
    A = feats["y"].detach()
    G = grads["g"].detach()
    
    # Alpha calculation (simplified Grad-CAM++ closed form)
    # weights ~ (G^2) / (2*G^2 + (A*G).sum() + eps)
    g2 = G ** 2
    g3 = G ** 3
    alpha_numer = g2
    alpha_denom = 2*g2 + (A * g3).sum(dim=(2, 3), keepdim=True) + 1e-8
    alpha = alpha_numer / alpha_denom
    
    # Final Weights = sum(alpha * ReLU(G))
    w = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)
    
    # 8. Generate Heatmap
    # cam = sum(w * A)
    cam = (w * A).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam = F.relu(cam)
    
    # Normalize per image [0..1]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Upsample to match input image size
    cam = F.interpolate(
        cam, 
        size=image.shape[-2:], 
        mode="bilinear", 
        align_corners=False
    )
    
    # Cleanup
    h1.remove()
    h2.remove()
    
    # Return shape [H, W]
    return cam.squeeze()
