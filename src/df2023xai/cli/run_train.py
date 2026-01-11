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
# src/df2023xai/cli/run_train.py
# ruff: noqa
# mypy: ignore-errors
"""
Launcher for segmentation training.
Corrected to use dynamic model factory and respect config parameters.
Revised to support resuming from checkpoint.
"""

from __future__ import annotations
import argparse, json, os, sys, yaml
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --- local imports -----------------------
from df2023xai.utils.seed import seed_from_cfg, set_global_seed, seed_worker
from df2023xai.utils.logging import setup_logging, log_run_header
from df2023xai.data.dataset import ForgerySegDataset

# FIX: Import from factory, not specific model file
from df2023xai.models.factory import build_model 

from df2023xai.train.losses import HybridForensicLoss
from df2023xai.train.loop import train_loop

import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"nll_loss2d_forward_out_cuda_template.*",
    category=UserWarning,
)

def _resolve_vars(text: str, project_root: Path) -> str:
    out = text.replace("${project_root}", str(project_root))
    import re
    pat = re.compile(r"\$\{env:([^,}]+),([^}]+)\}")
    def _rep(m):
        name, default = m.group(1), m.group(2)
        return os.environ.get(name, default)
    return pat.sub(_rep, out)

def _load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    def _walk(v):
        if isinstance(v, str): return _resolve_vars(v, PROJECT_ROOT)
        if isinstance(v, dict): return {k: _walk(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_walk(x) for x in v]
        return v
    return _walk(cfg)

def _build_dataloader(csv_path: str, cfg: Dict[str, Any], split: str) -> Tuple[ForgerySegDataset, DataLoader]:
    data = cfg.get("data", {}) or {}
    aug = cfg.get("aug", {}) or {}
    
    img_size = int(data.get("img_size", 512))
    batch_size = int(data.get("batch_size", 4))
    num_workers = int(data.get("num_workers", 2))
    pin_memory = bool(data.get("pin_memory", True))
    persistent_workers = bool(data.get("persistent_workers", True))
    is_train = (split == "train")

    ds = ForgerySegDataset(
        manifest_csv=csv_path,
        img_size=img_size,
        split=split,
        aug_cfg=aug if is_train else None,
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

def _build_model_and_loss(cfg: Dict[str, Any], device: torch.device):
    # FIX: Check 'models' (plural) to support your YAML, fallback to 'model'
    model_cfg = cfg.get("models") or cfg.get("model") or {}
    model_cfg = model_cfg.copy()
    
    # Extract known args, pass rest as kwargs
    name = model_cfg.pop("name", "unet_r34") 
    
    # FIX: Check for 'classes' (YAML style) or 'num_classes' (Code style)
    # Defaults to 2 only if neither is found.
    if "num_classes" in model_cfg:
        num_classes = int(model_cfg.pop("num_classes"))
    elif "classes" in model_cfg:
        num_classes = int(model_cfg.pop("classes"))
    else:
        num_classes = 2

    pretrained = bool(model_cfg.pop("pretrained", True))
    
    print(f"[INFO] Building model: {name} | classes={num_classes} | kwargs={model_cfg}")
    
    # Factory build 
    model = build_model(
        name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        **model_cfg
    ).to(device)

    # --- UPDATED LOSS LOGIC START ---
    loss_cfg = cfg.get("loss", {}) or {}
    
    # We ignore 'ce_class_weights' because we are now using the Binary formulation
    # consistent with Paper Eq(2) and Eq(3).
    # Default weights from Paper Section 3.5: lambda=0.5
    w_bce = float(loss_cfg.get("weight_bce", 0.5))
    w_dice = float(loss_cfg.get("weight_dice", 0.5))

    print(f"[INFO] Initializing HybridForensicLoss (BCE={w_bce}, Dice={w_dice})")
    criterion = HybridForensicLoss(weight_bce=w_bce, weight_dice=w_dice)
    criterion = criterion.to(device)
    # --- UPDATED LOSS LOGIC END ---

    return model, criterion, None # No class weights needed for BCE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pt checkpoint to resume weights from") # <--- NEW ARGUMENT
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))

    # Defaults
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    cfg.setdefault("out", {})
    cfg.setdefault("loss", {})
    cfg["loss"].setdefault("ce_class_weights", [0.2, 0.8])
    cfg["loss"].setdefault("use_dice", True)
    cfg["loss"].setdefault("ignore_background_in_dice", True)

    # Seed
    seed = seed_from_cfg(cfg, fallback=1337)
    set_global_seed(seed)
    
    # Logging
    out_dir = cfg.get("out", {}).get("dir", str(PROJECT_ROOT / "outputs" / "seg_unet_r34_v2" / f"seed{seed}"))
    log_file = cfg.get("out", {}).get("log_file", "stdout.log")
    
    # If resuming, append to log instead of overwriting? 
    # Standard setup_logging usually appends if file exists.
    logger = setup_logging("train_launcher", log_file=str(Path(out_dir) / log_file), to_stdout=True)
    log_run_header(logger, cfg, out_dir, PROJECT_ROOT)

    # Paths
    train_csv = cfg["data"]["train_manifest_csv"]
    val_csv   = cfg["data"]["val_manifest_csv"]
    if not Path(train_csv).exists() or not Path(val_csv).exists():
        raise ValueError("Config must specify 'train_manifest_csv' and 'val_manifest_csv' to prevent data leakage.")

    # Data
    _, train_loader = _build_dataloader(train_csv, cfg, "train")
    _, val_loader   = _build_dataloader(val_csv,   cfg, "val")

    # Model Building (Dynamic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, ce_w = _build_model_and_loss(cfg, device)
    
    # --- RESUME LOGIC START ---
    if args.resume:
        chk_path = Path(args.resume)
        if not chk_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {chk_path}")
        
        logger.info(f"[RESUME] Loading weights from {chk_path} ...")
        # map_location ensures we can load even if saved on different device ID
        checkpoint = torch.load(chk_path, map_location=device)
        
        # Handle dictionary structure (some save whole dict, some just state_dict)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            step_info = checkpoint.get("step", "?")
            dice_info = checkpoint.get("best_dice", "?")
            logger.info(f"[RESUME] Checkpoint meta: step={step_info}, best_dice={dice_info}")
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        try:
            model.load_state_dict(state_dict)
            logger.info("[RESUME] Model weights loaded successfully.")
            logger.info("[RESUME] NOTE: Optimizer state is NOT restored. Training starts with fresh optimizer (step 0).")
        except Exception as e:
            logger.error(f"[RESUME] Failed to load state dict: {e}")
            raise e
    # --- RESUME LOGIC END ---
    
    if ce_w is not None:
        try:
            logger.info(f"[loss] CE class weights device={ce_w.device} values={ce_w.tolist()}")
        except Exception:
            logger.info("[loss] CE class weights=<unavailable>")

    # Save Config and Start
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "config_train.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    train_loop(model=model, criterion=criterion, train_loader=train_loader, val_loader=val_loader, cfg=cfg, out_dir=out_dir)

if __name__ == "__main__":
    main()
