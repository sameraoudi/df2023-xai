"""
===============================================================================
Script Name   : run_train.py
Description   : Universal training launcher for DF2023 Forensic Segmentation.
                
                Features:
                - Dynamic Model Loading (SegFormer / U-Net via SMP)
                - Differentiable Hybrid Loss (CE + SoftDice)
                - Automatic Mixed Precision (AMP)
                - YAML-based Configuration with Environment Variable support
                - Augmentation Pipeline integration

How to Run    : 
                # Train SegFormer
                # Run for SEEDS 1337, 2027, and 3141
                SEED=1337 python -m df2023xai.cli.run_train \
                    --config configs/train_segformer_b2_full.yaml train

                # Train U-Net
                # Run for SEEDS 1337, 2027, and 3141
                SEED=1337 python -m df2023xai.cli.run_train \
                    --config configs/train_unet_r34_full.yaml train

Inputs        :
                --config : Path to YAML configuration file.

Outputs       :
                - Model Checkpoints (best.pt, last.pt) in outputs/ dir.
                - Training Logs (stdout.log).
                - Resolved Config JSON (config_train.json).

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
- Reproducibility: Enforces global seeding (Python, NumPy, Torch) before
  initializing data loaders.
- Optimization: Uses 'pin_memory' and 'persistent_workers' for high-throughput
  GPU feeding (verified 97% utilization on L40).
- Safety: Dumps the exact configuration used (with resolved paths) to JSON
  for audit trails.

Dependencies  :
- Python >= 3.10
- torch, segmentation_models_pytorch (smp)
- df2023xai (Internal package)
===============================================================================
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import yaml
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# --- Internal Imports ---
# Adjust relative path to locate project root regardless of execution dir
PROJECT_ROOT = Path(__file__).resolve().parents[3]

from df2023xai.utils.seed import seed_from_cfg, set_global_seed, seed_worker
from df2023xai.utils.logging import setup_logging, log_run_header
from df2023xai.data.dataset import ForgerySegDataset
from df2023xai.train.losses import CEDice
from df2023xai.train.loop import train_loop

# Suppress harmless CUDA warnings from older SMP versions if present
warnings.filterwarnings(
    "ignore",
    message=r"nll_loss2d_forward_out_cuda_template.*",
    category=UserWarning,
)


def _resolve_vars(text: str, project_root: Path) -> str:
    """
    Resolve placeholders in YAML strings.
    - ${project_root} -> Absolute path to repo
    - ${env:VAR,default} -> Environment variable or default
    """
    out = text.replace("${project_root}", str(project_root))
    pat = re.compile(r"\$\{env:([^,}]+),([^}]+)\}")
    
    def _rep(m):
        name, default = m.group(1), m.group(2)
        return os.environ.get(name, default)
    
    return pat.sub(_rep, out)


def _load_cfg(path: Path) -> Dict[str, Any]:
    """Load and parse YAML config with recursive variable resolution."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def _walk(v):
        if isinstance(v, str):
            return _resolve_vars(v, PROJECT_ROOT)
        if isinstance(v, dict):
            return {k: _walk(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_walk(x) for x in v]
        return v

    return _walk(cfg)


def _build_dataloader(csv_path: str, cfg: Dict[str, Any], split: str) -> Tuple[ForgerySegDataset, DataLoader]:
    """
    Construct Dataset and Dataloader.
    CRITICAL: Passes augmentation config only for 'train' split.
    """
    data = cfg.get("data", {}) or {}
    aug_cfg = cfg.get("aug", {}) or {}  # Load augmentation settings
    
    img_size = int(data.get("img_size", 512))
    batch_size = int(data.get("batch_size", 4))
    num_workers = int(data.get("num_workers", 2))
    pin_memory = bool(data.get("pin_memory", True))
    persistent_workers = bool(data.get("persistent_workers", True))
    
    is_train = (split == "train")

    # Initialize Dataset with Augmentation Config
    ds = ForgerySegDataset(
        manifest_csv=csv_path,
        img_size=img_size,
        split=split,
        aug_cfg=aug_cfg if is_train else None  # Only augment training data
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=is_train,
    )
    return ds, dl


def _build_model_from_cfg(cfg: Dict[str, Any]) -> nn.Module:
    """
    Factory function to instantiate models based on YAML config.
    Supports 'unet', 'segformer', etc. via segmentation-models-pytorch.
    """
    mc = cfg.get("model", {})
    name = mc.get("name", "unet").lower()
    num_classes = int(mc.get("num_classes", 2))
    
    # 1. SegFormer Logic
    if "segformer" in name:
        # e.g. name="segformer_b2" -> encoder_name="mit_b2"
        encoder = name.replace("segformer_", "mit_")
        return smp.Segformer(
            encoder_name=encoder,
            encoder_weights="imagenet" if mc.get("pretrained", True) else None,
            classes=num_classes,
        )
    
    # 2. U-Net Logic
    elif "unet" in name:
        # e.g. name="unet_r34" or just "unet" -> defaults to resnet34 if not parsed
        # Simple heuristic: if name ends with _r34, use resnet34
        encoder = "resnet34"
        if "r18" in name: encoder = "resnet18"
        if "r50" in name: encoder = "resnet50"
        
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if mc.get("pretrained", True) else None,
            classes=num_classes,
        )
    
    else:
        raise ValueError(f"Unknown model name in config: {name}")


def _build_loss(cfg: Dict[str, Any], device: torch.device) -> Tuple[nn.Module, Any]:
    """Initialize the Hybrid Loss (CE + Dice)."""
    loss_cfg = cfg.get("loss", {}) or {}
    ce_w = loss_cfg.get("ce_class_weights", [0.2, 0.8])
    ignore_bg = bool(loss_cfg.get("ignore_background_in_dice", True))
    use_dice = bool(loss_cfg.get("use_dice", True))

    criterion = CEDice(class_weights=ce_w, ignore_background_in_dice=ignore_bg, use_dice=use_dice)
    criterion = criterion.to(device)
    
    w = getattr(criterion, "class_weights", None)
    return criterion, w


def main():
    parser = argparse.ArgumentParser(description="DF2023-XAI Training Launcher")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    args = parser.parse_args()

    # 1. Load Configuration
    cfg = _load_cfg(Path(args.config))

    # 2. Set Defaults (Fail-safe)
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    cfg.setdefault("out", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("aug", {})

    # 3. Setup Reproducibility & Logging
    seed = seed_
