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

from __future__ import annotations
import math
import sys
import time
import logging
import csv
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class CsvLogger:
    """Helper to write training metrics to a CSV file for plotting."""
    def __init__(self, filepath: Path, fieldnames: list[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def _setup_logger():
    logger = logging.getLogger("train_loop")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(sh)
    return logger


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robustly unpack batch regardless of Dict or Tuple format."""
    if isinstance(batch, dict):
        # Common keys used by various dataloaders
        for a in ("image", "img", "x"):
            if a in batch:
                for b in ("mask", "y", "target"):
                    if b in batch:
                        return batch[a], batch[b]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Dataset must yield (image, mask) or dict with standard keys.")


def _amp_ctx():
    """Returns the correct AMP context manager for the current PyTorch version."""
    if torch.cuda.is_available():
        try:
            return torch.amp.autocast("cuda")
        except AttributeError:
            # Fallback for older Torch versions
            from torch.cuda.amp import autocast
            return autocast()
    from contextlib import nullcontext
    return nullcontext()


def _grad_scaler(enabled: bool):
    """Initializes Gradient Scaler for AMP."""
    if not enabled:
        class _Dummy:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
            def unscale_(self, opt): pass
        return _Dummy()
    try:
        return torch.amp.GradScaler("cuda")
    except AttributeError:
        from torch.cuda.amp import GradScaler
        return GradScaler(enabled=True)


def _val_pass(model: nn.Module, val_loader, device: torch.device) -> Tuple[float, float]:
    """
    Perform a rigorous validation pass.
    Returns: (Mean IoU, Mean Dice)
    """
    model.eval()
    tot_iou = 0.0
    tot_dice = 0.0
    n = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            img, mask = _unpack_batch(batch)
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Inference
            logits = model(img)
            pred = torch.argmax(logits, dim=1)
            
            # Binary masks (Foreground=1)
            pf = (pred == 1).float()
            tf = (mask == 1).float()

            # Intersection & Union (Batch-wise sum for speed, then mean)
            inter = (pf * tf).sum(dim=(1, 2))
            union = (pf + tf - pf * tf).sum(dim=(1, 2)) + 1e-7
            
            # Dice: 2*Inter / (Area_Pred + Area_GT)
            dice_denom = pf.sum(dim=(1, 2)) + tf.sum(dim=(1, 2)) + 1e-7
            dice_score = (2.0 * inter / dice_denom).mean()
            
            # IoU: Inter / Union
            iou_score = (inter / union).mean()

            tot_dice += dice_score.item()
            tot_iou += iou_score.item()
            n += 1.0

    model.train()
    n = max(n, 1.0)
    return tot_iou / n, tot_dice / n


def train_loop(
    model: nn.Module, 
    criterion: nn.Module, 
    train_loader, 
    val_loader, 
    cfg: Dict[str, Any], 
    out_dir: str
):
    logger = _setup_logger()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Initialize CSV Logger for Paper Plots
    csv_log = CsvLogger(out / "metrics.csv", ["step", "lr", "train_loss", "val_iou", "val_dice"])

    # Parse Config
    tc = cfg.get("train", {}) or {}
    lr = float(tc.get("lr", 4e-4))
    wd = float(tc.get("weight_decay", 0.0))
    max_steps = int(tc.get("max_steps", 160000))
    warmup = int(tc.get("warmup_steps", 3000))
    min_lr = float(tc.get("min_lr", 1e-7))
    val_every = int(tc.get("val_every_steps", 1000))
    patience = int(tc.get("early_stop_patience", 12))
    grad_clip = float(tc.get("grad_clip", 0.0))
    use_amp = bool(tc.get("amp", True))

    # Setup Optimizer & Scaler
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = _grad_scaler(enabled=use_amp)

    # Cosine Annealing Scheduler with Warmup
    def _lr_at(step):
        if step < warmup: 
            return lr * (step / float(max(1, warmup)))
        prog = (step - warmup) / float(max(1, max_steps - warmup))
        prog = max(0.0, min(1.0, prog))
        return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))

    best_dice = -1.0
    no_improve = 0
    step = 0
    it = iter(train_loader)
    
    logger.info(f"[*] Starting training loop: Steps={max_steps}, Device={device}, AMP={use_amp}")
    model.train()

    while step < max_steps:
        # Fetch Data
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        img, mask = _unpack_batch(batch)
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Update LR
        cur_lr = _lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        # Forward Pass
        opt.zero_grad(set_to_none=True)
        with _amp_ctx():
            # Clamp logits for float16 stability
            logits = torch.clamp(model(img), -10.0, 10.0)
            loss = criterion(logits, mask)

        # Backward Pass
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            logger.warning(f"[Guard] Non-finite loss {loss.item()} at step={step}. Skipping update.")

        # Validation & Logging
        if step > 0 and (step % val_every) == 0:
            vi, vd = _val_pass(model, val_loader, device)
            
            logger.info(f"Step={step}/{max_steps} | LR={cur_lr:.2e} | TrainLoss={loss.item():.4f} | ValIoU={vi:.4f} | ValDice={vd:.4f}")
            
            # Log to CSV
            csv_log.log({
                "step": step, 
                "lr": f"{cur_lr:.2e}", 
                "train_loss": f"{loss.item():.4f}", 
                "val_iou": f"{vi:.4f}", 
                "val_dice": f"{vd:.4f}"
            })

            # Checkpoint: Last
            torch.save({
                "model": model.state_dict(), 
                "step": step, 
                "best_dice": best_dice
            }, out / "last.pt")

            # Checkpoint: Best
            if vd > best_dice:
                logger.info(f"    --> New Best Dice! ({best_dice:.4f} -> {vd:.4f})")
                best_dice = vd
                no_improve = 0
                torch.save({
                    "model": model.state_dict(), 
                    "step": step, 
                    "best_dice": best_dice
                }, out / "best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"[EARLY-STOP] No improvement for {no_improve} checks. Best Dice={best_dice:.4f}")
                    break

        step += 1

    # Final Save
    torch.save({"model": model.state_dict(), "step": step, "best_dice": best_dice}, out / "last.pt")
    logger.info(f"[Done] Training finished. Best Dice={best_dice:.4f}. Outputs -> {out}")
