"""
===============================================================================
Script Name   : forensic.py
Description   : Core Evaluation Logic for Phase 2 (Forensic Metrics).
                
                Functionality:
                - Computes Binary IoU (Intersection over Union).
                - Computes Dice Coefficient (F1 Score).
                - Computes Pixel-F1 (Precision/Recall at pixel level).
                - Handles "Pristine" (Empty Mask) edge cases robustly.
                - Aggregates results into a Pandas DataFrame for reporting.

Inputs        :
                - models       : List of model configs (name, path).
                - manifest_csv : Path to Test Set manifest.
                - img_size     : Resolution for inference (512).

Outputs       :
                - metrics_summary.csv : Final table for the paper.
                - robustness_report.json : Placeholder for future adversarial tests.

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
- Metric Stability: Uses an epsilon (1e-7) to prevent divide-by-zero errors.
  Specifically handles the case where Ground Truth is empty (Pristine Image).
  If GT is empty and Pred is empty -> Score = 1.0 (Perfect).
- Architecture Agnostic: Dynamically builds U-Net or SegFormer based on the
  provided model name string.

Dependencies  :
- Python >= 3.10
- torch, pandas, numpy
- segmentation_models_pytorch (smp)
- df2023xai (Internal)
===============================================================================
"""

from __future__ import annotations
import os
import json
import torch
import pandas as pd
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# Internal Imports
from df2023xai.data.dataset import ForgerySegDataset


def _build_model_from_name(name: str, num_classes: int = 2) -> torch.nn.Module:
    """Reconstruct model architecture based on naming convention."""
    name = name.lower()
    if "segformer" in name:
        encoder = name.replace("segformer_", "mit_")
        return smp.Segformer(encoder_name=encoder, classes=num_classes)
    elif "unet" in name:
        encoder = "resnet34" 
        if "r50" in name: encoder = "resnet50"
        return smp.Unet(encoder_name=encoder, classes=num_classes)
    else:
        # Fallback for generic names
        return smp.Unet(encoder_name="resnet34", classes=num_classes)


def compute_batch_metrics(pred: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float, float]:
    """
    Robust calculation of IoU and Dice for binary segmentation.
    Returns sums of: (iou, dice, count) to be aggregated.
    """
    # pred: [B, H, W] (0 or 1)
    # mask: [B, H, W] (0 or 1)
    
    # Flatten spatial dimensions: [B, -1]
    pred_flat = pred.view(pred.size(0), -1).float()
    mask_flat = mask.view(mask.size(0), -1).float()
    
    # Intersection & Union
    intersection = (pred_flat * mask_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    mask_sum = mask_flat.sum(dim=1)
    
    union = pred_sum + mask_sum - intersection
    dice_denom = pred_sum + mask_sum
    
    # Epsilon for numerical stability
    eps = 1e-7
    
    # IoU Calculation
    # Note: If Union is 0 (both empty), IoU is 1.0 (Perfect match of background)
    iou = (intersection + eps) / (union + eps)
    
    # Dice Calculation
    dice = (2.0 * intersection + eps) / (dice_denom + eps)
    
    # Fix explicit pristine case where eps might slightly skew if not handled:
    # If mask is empty and pred is empty, math above gives ~1.0. 
    # If mask is empty and pred is NOT empty, math gives ~0.0.
    # This standard eps approach is robust enough for batch training/eval.
    
    return iou.sum().item(), dice.sum().item(), pred.size(0)


@torch.no_grad()
def eval_single_model(
    model_name: str, 
    model_path: str, 
    manifest_csv: str, 
    img_size: int, 
    sample_n: int = 0, 
    batch_size: int = 4,
    device: str = "cuda"
) -> Dict[str, float]:
    
    # 1. Load Data
    # Use "test" split if available, otherwise fallback logic could be added.
    # We assume manifest_csv points to the correct partition.
    ds = ForgerySegDataset(manifest_csv, split="test", img_size=img_size)
    
    # Handle Sampling
    total_len = len(ds)
    if sample_n > 0:
        total_len = min(total_len, sample_n)
        # Create a subset if sampling is requested
        indices = list(range(total_len))
        ds = torch.utils.data.Subset(ds, indices)
        
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Load Model
    model = _build_model_from_name(model_name).to(device)
    try:
        chk = torch.load(model_path, map_location=device)
        state_dict = chk["model"] if "model" in chk else chk
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load weights for {model_name}: {e}")
        return {"iou": 0.0, "dice": 0.0, "samples": 0}
        
    model.eval()

    # 3. Evaluation Loop
    total_iou = 0.0
    total_dice = 0.0
    seen_samples = 0
    
    for imgs, masks in dl:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # Inference
        logits = model(imgs)
        preds = logits.argmax(dim=1)  # [B, H, W]
        
        # Compute Metrics
        b_iou, b_dice, b_n = compute_batch_metrics(preds, masks)
        
        total_iou += b_iou
        total_dice += b_dice
        seen_samples += b_n

    # 4. Aggregate
    mean_iou = total_iou / max(seen_samples, 1)
    mean_dice = total_dice / max(seen_samples, 1)
    
    return {
        "samples": seen_samples,
        "iou": mean_iou,
        "dice": mean_dice
    }


def write_forensic_reports(
    models: List[Dict[str, str]], 
    manifest_csv: str, 
    img_size: int, 
    out_dir: str, 
    sample_n: int = 0,
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Main entry point to evaluate a list of models and save reports.
    models: List of dicts, e.g. [{'name': 'SegFormer', 'path': '...'}]
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    
    for m in models:
        m_name = m["name"]
        m_path = m["path"]
        print(f"[*] Evaluating: {m_name} ...")
        
        res = eval_single_model(
            model_name=m_name,
            model_path=m_path,
            manifest_csv=manifest_csv,
            img_size=img_size,
            sample_n=sample_n,
            device=device
        )
        
        rows.append({
            "model": m_name,
            "path": m_path,
            **res
        })
        
    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Save Robustness Placeholder
    robo = {
        "status": "partial",
        "description": "Base metrics on test set. Robustness (JPEG, Noise) to be added in Phase 2 extension.",
        "perturbations": ["None"]
    }
    with open(os.path.join(out_dir, "robustness_report.json"), "w") as f:
        json.dump(robo, f, indent=2)

    return df
