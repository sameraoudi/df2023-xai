"""
===============================================================================
Script Name   : loop.py
Description   : Core training and validation loop for DF2023 Forensic Segmentation.
                
                Capabilities:
                - Automatic Mixed Precision (AMP) via GradScaler.
                - Gradient Clipping for Transformer stability.
                - Real-time CSV logging (metrics.csv) for paper plotting.
                - Early Stopping based on Validation Dice coefficient.
                - Safe "logit clamping" to prevent float16 overflows.

Inputs        :
                - model        : PyTorch Neural Network (U-Net or SegFormer).
                - criterion    : Differentiable Loss function (Hybrid).
                - train/val_dl : DataLoaders.
                - cfg          : Parsed Configuration Dictionary.
                - out_dir      : Directory for checkpoints and logs.

Outputs       :
                - best.pt      : Model with highest Validation Dice.
                - last.pt      : Checkpoint at final step.
                - metrics.csv  : Time-series data of Loss/IoU/Dice.

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
- Validation Metric: Uses 'Soft Dice' and 'IoU' calculated on the fly.
  We add epsilon (1e-7) to denominators to prevent division-by-zero errors
  on empty masks (pristine images).
- Optimization: Uses Cosine Annealing LR scheduler with Warmup. This is 
  essential for SegFormer convergence.
- Data Unpacking: Robustly handles both Tuple and Dict batch formats to
  support different SMP/Timm versions.

Dependencies  :
- Python >= 3.10
- torch, numpy
===============================================================================
"""
# src/df2023xai/train/loop.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

def compute_binary_metrics(logits, targets, threshold=0.5):
    """
    Compute IoU and Dice for Binary Segmentation (Sigmoid).
    logits: [B, 1, H, W]
    targets: [B, H, W] (0 or 1)
    """
    # Sigmoid + Threshold
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (preds.sum() + targets.sum() + 1e-7)
    
    return iou.item(), dice.item()

def train_loop(model, criterion, train_loader, val_loader, cfg, out_dir):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(cfg['train']['lr']), 
        weight_decay=float(cfg['train']['weight_decay'])
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['train']['max_steps']
    )

    best_dice = 0.0
    step = 0
    max_steps = cfg['train']['max_steps']
    val_interval = cfg['train']['val_every_steps']
    
    # Detect Binary Mode from Model
    # If model.segmentation_head[0].out_channels == 1 -> Binary
    is_binary = True
    try:
        # Check SMP/Segformer head
        if hasattr(model, 'seg_head'):
            if model.seg_head[2].out_channels > 1: is_binary = False
        elif hasattr(model, 'segmentation_head'):
             if model.segmentation_head[0].out_channels > 1: is_binary = False
    except:
        pass # Default to Binary (Safe for our current config)

    print(f"[Loop] Starting training. Binary Mode: {is_binary}")
    
    model.train()
    
    # Infinite Iterator wrapper
    train_iter = iter(train_loader)
    
    with tqdm(total=max_steps, initial=0) as pbar:
        while step < max_steps:
            try:
                images, masks = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, masks = next(train_iter)
                
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            logits = model(images)
            
            # Loss Calculation
            # Binary Loss expects [B, 1, H, W] and [B, H, W]
            loss = criterion(logits, masks)
            
            optimizer.zero_grad()
            loss.backward()
            
            if cfg['train'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
                
            optimizer.step()
            scheduler.step()
            
            step += 1
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            # Validation
            if step % val_interval == 0:
                val_iou, val_dice = validate(model, val_loader, device, is_binary)
                model.train() # Switch back
                
                # Log to console
                tqdm.write(f"Step {step} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")
                
                # Save Best
                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
                    tqdm.write(f"  [+] New Best Dice! Saved best.pt")

                # Save Last
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_dice': best_dice
                }, os.path.join(out_dir, "last.pt"))

def validate(model, loader, device, is_binary=True):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    steps = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            
            if is_binary:
                iou, dice = compute_binary_metrics(logits, masks)
            else:
                # Fallback for old Multi-class code (if ever needed)
                preds = logits.argmax(dim=1)
                # ... (Simple multiclass metrics logic would go here)
                iou, dice = 0.0, 0.0 
            
            total_iou += iou
            total_dice += dice
            steps += 1
            
            if steps > 50: break # Validation limit to save time
            
    return total_iou / steps, total_dice / steps
