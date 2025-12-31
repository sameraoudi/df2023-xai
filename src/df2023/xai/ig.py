"""
===============================================================================
Script Name   : ig.py
Description   : Implementation of Integrated Gradients (IG) for Semantic 
                Segmentation.
                
                Methodology:
                - Approximates the path integral of gradients from a Baseline 
                  (Black Image) to the Input.
                - Formula: IG(x) = (x - x') * Integral(grad(F(alpha * x + (1-alpha)*x')))
                - Satisfies 'Completeness' and 'Sensitivity' axioms, making it 
                  more theoretically grounded than vanilla gradients.

Inputs        :
                - model     : Trained PyTorch model (eval mode).
                - image     : Input tensor [C, H, W].
                - steps     : Number of Riemann approximation steps (default 32).

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
- Memory Optimization: Computes gradients iteratively (step-by-step) rather 
  than stacking all inputs in memory. This allows running high-step counts 
  (e.g., 50-100) on 512x512 images without OOM.
- Aggregation: We sum the absolute attributions across color channels to 
  create a robust 2D saliency map.
- Normalization: Uses local min-max scaling to highlight relative importance
  within the specific image.

Dependencies  :
- Python >= 3.10
- torch
===============================================================================
"""

from __future__ import annotations
import torch
from typing import Optional

def integrated_gradients_minimal(
    model: torch.nn.Module, 
    image: torch.Tensor, 
    steps: int = 32, 
    class_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Compute Integrated Gradients attribution map.
    Optimized for memory efficiency (accumulates gradients iteratively).
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 1. Prepare Input and Baseline
    x = image.unsqueeze(0).to(device) # [1, C, H, W]
    baseline = torch.zeros_like(x)
    
    # 2. Iterative Integral Approximation (Riemann Sum)
    # We accumulate gradients over 'steps'
    accumulated_grads = torch.zeros_like(x)
    
    for i in range(1, steps + 1):
        alpha = i / steps
        
        # Interpolate
        # scale = baseline + alpha * (x - baseline)
        # Since baseline is zero, this simplifies to:
        scaled_input = x * alpha
        scaled_input.requires_grad_(True)
        
        # Forward Pass
        logits = model(scaled_input)
        
        # Determine Target Scalar for Backprop
        if logits.ndim == 4:
            # Segmentation Output: [1, Classes, H, W]
            # We target the mean score of the "Forgery" class (Index 1)
            # This asks: "Which pixels drive the global forgery prediction?"
            if class_idx is None:
                # Default: Class 1 if binary, else Class 0
                target_idx = 1 if logits.shape[1] > 1 else 0
                score = logits[:, target_idx:target_idx+1].mean()
            else:
                score = logits[:, class_idx:class_idx+1].mean()
        else:
            # Classification Output (Fallback)
            idx = int(logits.argmax(1).item()) if class_idx is None else class_idx
            score = logits[0, idx]
            
        # Backward Pass
        model.zero_grad(set_to_none=True)
        score.backward()
        
        # Accumulate
        accumulated_grads += scaled_input.grad.detach()
        
    # 3. Average Gradients
    avg_grads = accumulated_grads / steps
    
    # 4. Compute Attribution: (Input - Baseline) * AvgGrad
    # Since baseline is 0, this is: Input * AvgGrad
    # We take absolute value to measure "magnitude of influence"
    attr = (x * avg_grads).abs().sum(dim=1)[0] # [H, W]
    
    # 5. Normalize [0, 1] for visualization
    attr_min = attr.min()
    attr_max = attr.max()
    
    if attr_max > attr_min:
        attr = (attr - attr_min) / (attr_max - attr_min + 1e-8)
        
    return attr
