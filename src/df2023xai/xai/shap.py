"""
===============================================================================
Script Name   : shap.py
Description   : Wrapper for SHAP (Shapley Additive Explanations) using the
                Partition Explainer (Image).
                
                Methodology:
                - Uses 'PartitionExplainer' to approximate Shapley values by
                  masking out hierarchical superpixels.
                - This is significantly faster than KernelSHAP for 512x512 
                  images while retaining game-theoretic properties.

Inputs        :
                - model     : Trained PyTorch model.
                - image     : Input tensor [C, H, W].

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
- Lazy Import: The 'shap' library is heavy. We import it only when the 
  function is called to keep the rest of the pipeline lightweight.
- Masking: Uses 'inpaint_telea' masking to simulate missing features (background).
- Optimization: Limits 'max_evals' to prevent the script from hanging on 
  high-res images.

Dependencies  :
- shap (pip install shap)
- torch, numpy
===============================================================================
"""

from __future__ import annotations
import torch
import numpy as np
import warnings

# NOTE: Since this file is named 'shap.py', we must be careful with imports.
# We defer the library import to the function scope to avoid namespace collisions.

def shap_or_fallback(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """
    Compute SHAP attribution using PartitionExplainer.
    Falls back to a zero-map if 'shap' library is missing.
    """
    device = next(model.parameters()).device
    
    try:
        import shap
    except ImportError:
        warnings.warn("[SHAP] Library not installed. Returning empty map.")
        return torch.zeros(image.shape[-2:], device=device)

    model.eval()
    
    # 1. Prepare Function for SHAP
    # SHAP expects a function that takes numpy arrays (B,H,W,C) and returns scores
    def f(x_np):
        # Transpose NHWC -> NCHW
        x_tens = torch.from_numpy(x_np).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            logits = model(x_tens)
            # Target the "Forgery" class (Index 1) average score
            if logits.ndim == 4:
                return logits[:, 1, :, :].mean(dim=(1, 2)).cpu().numpy().reshape(-1, 1)
            else:
                return logits[:, 1].cpu().numpy().reshape(-1, 1)

    # 2. Prepare Input
    # Transpose [C,H,W] -> [H,W,C] for SHAP
    img_np = image.permute(1, 2, 0).cpu().numpy() # [H, W, 3]
    input_batch = np.expand_dims(img_np, axis=0)  # [1, H, W, 3]

    # 3. Create Explainer
    # Using a "Blur" or "Inpaint" masker is standard for image background removal simulation
    # We use a partition masker suitable for images
    masker = shap.maskers.Image("inpaint_telea", input_batch[0].shape)
    explainer = shap.Explainer(f, masker, output_names=["ForgeryScore"])

    # 4. Compute Shapley Values
    # max_evals limits runtime; higher is more accurate but slower
    # We select the top output class (Forgery)
    shap_values = explainer(
        input_batch, 
        max_evals=300, 
        batch_size=1, 
        outputs=shap.Explanation.argsort.flip[:1]
    )

    # 5. Extract Heatmap
    # SHAP returns (1, H, W, 3, OutputClasses). 
    values = shap_values.values[0] 
    
    # Sum absolute values across RGB to get a single saliency map
    heatmap_np = np.abs(values).sum(axis=-1) # [H, W]
    
    # Normalize
    heatmap_t = torch.from_numpy(heatmap_np).to(device).float()
    h_min, h_max = heatmap_t.min(), heatmap_t.max()
    if h_max > h_min:
        heatmap_t = (heatmap_t - h_min) / (h_max - h_min + 1e-8)
        
    return heatmap_t
