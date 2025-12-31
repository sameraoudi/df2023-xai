"""
===============================================================================
Script Name   : dataset.py
Description   : PyTorch Dataset definition for the DF2023 Forensic Segmentation
                task. It handles:
                1. Loading images and binary masks from disk.
                2. Resizing to canonical resolution (512x512).
                3. Normalizing pixel values to [0, 1].
                4. Applying geometric augmentations (Flip, Rotate) strictly
                   synchronizing image and mask transformations.

Inputs        :
                - manifest_csv : Path to the split-specific CSV (Train/Val/Test).
                - img_size     : Target resolution (default 512).
                - aug_cfg      : Dictionary of augmentation flags (Train only).

Outputs       :
                - image        : Float tensor [3, H, W] normalized.
                - mask         : Long tensor [H, W] {0, 1}.

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
- Mask Resizing: Uses 'nearest-neighbor' interpolation. Bilinear interpolation
  would introduce artifacts (e.g., values like 0.3 or 0.7) which destroy the
  ground-truth binary nature of forensic masks.
- Augmentation: Transforms are applied jointly to image and mask to maintain
  spatial alignment.
- Efficiency: Loads images via PIL/Simd to minimize I/O overhead.

Dependencies  :
- Python >= 3.10
- torch, numpy, PIL
===============================================================================
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import csv
import random
import numpy as np
import torch
from PIL import Image


class ForgerySegDataset(torch.utils.data.Dataset):
    """
    Forensic Segmentation Dataset for DF2023.
    """

    def __init__(
        self, 
        manifest_csv: str, 
        img_size: int = 512, 
        split: str = "train",
        aug_cfg: Optional[Dict[str, Any]] = None
    ):
        self.manifest_csv = manifest_csv
        self.img_size = int(img_size)
        self.split = split
        # Only apply augmentations if we are in training mode and config is provided
        self.aug_cfg = aug_cfg if (split == "train" and aug_cfg) else None
        self.rows = self._read_csv(manifest_csv)

    def _read_csv(self, path: str) -> List[Tuple[str, str]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        rows = []
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                ip = r.get("image_path")
                mp = r.get("mask_path")
                if ip and mp:
                    rows.append((ip, mp))
        if not rows:
            raise RuntimeError(f"No valid rows found in manifest: {path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _load_img(self, p: str) -> Image.Image:
        """Load RGB image and resize using high-quality Bilinear interpolation."""
        return Image.open(p).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)

    def _load_mask(self, p: str) -> Image.Image:
        """
        Load Mask and resize using Nearest Neighbor.
        CRITICAL: Do not use Bilinear here, or you corrupt the binary labels.
        """
        return Image.open(p).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

    def _augment(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply synchronized geometric transforms to image and mask."""
        if not self.aug_cfg:
            return img, mask
        
        # 1. Random Horizontal Flip
        if self.aug_cfg.get("hflip", False) and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Random 90-degree Rotation
        if self.aug_cfg.get("rot90", False) and random.random() > 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
            
        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.rows[idx]
        
        # 1. Load Data
        img_pil = self._load_img(ip)
        mask_pil = self._load_mask(mp)
        
        # 2. Apply On-the-fly Augmentations (if Training)
        img_pil, mask_pil = self._augment(img_pil, mask_pil)
        
        # 3. Convert to Tensor & Normalize
        # Image: [0, 255] -> [0.0, 1.0]
        arr_img = np.asarray(img_pil, dtype=np.float32) / 255.0
        # Transpose from HWC (PIL) to CHW (PyTorch)
        arr_img = np.transpose(arr_img, (2, 0, 1)) 
        
        # Mask: Binarize thresholding (safe guard against compression artifacts)
        arr_mask = np.asarray(mask_pil, dtype=np.uint8)
        arr_mask = (arr_mask > 127).astype(np.int64) # Long Tensor required for Loss
        
        return torch.from_numpy(arr_img).float(), torch.from_numpy(arr_mask).long()
