"""
===============================================================================
PROJECT      : DF2023-XAI
SCRIPT       : run_stratified_eval.py
VERSION      : 1.0.0
DESCRIPTION  : Performance profiling stratified by manipulation type.
-------------------------------------------------------------------------------
FUNCTIONALITY:
    Evaluates a model's robustness across different forgery categories: 
    Splicing, Copy-Move, Inpainting, and Enhancement. It parses 
    metadata directly from the DF2023 file naming convention to generate 
    a comparative performance table (IoU and F1-Score) per category.

USAGE:
    Run stratified analysis on the Test Set
    (Uses the first model defined in the config)
    python -m df2023xai.cli.run_stratified_eval configs/xai_gen.yaml

ARGUMENTS:
    config_path   : (File) Path to the YAML configuration containing the 
                    manifest path and model directory.

AUTHOR       : Dr. Samer Aoudi
AFFILIATION  : Higher Colleges of Technology (HCT), UAE
ROLE         : Assistant Professor & Division Chair (CIS)
EMAIL        : cybersecurity@sameraoudi.com
ORCID        : 0000-0003-3887-0119
CREATED      : 2026-01-20
UPDATED      : 2026-02-01

LICENSE      : MIT License
CITATION     : If used in academic research, please cite:
               Aoudi, S. (2026). "Beyond Accuracy — A Risk-Centric 
               Comparative Evaluation of Deep Intrusion Detection Systems."

DESIGN NOTES:
    - Metadata Parsing: Implements a specific decoder for DF2023 filenames 
      (e.g., COCO_DF_[TYPE]...) to extract manipulation categories.
    - Data Alignment: Uses a non-shuffled DataLoader (batch size 1) mapped 
      against a direct CSV read to ensure strict 1:1 path-to-metric integrity.
    - Metrics: Calculates Intersection over Union (IoU) and Dice Coefficient 
      (F1-Score) with smoothing (1e-7) to prevent division by zero.
    - Hardware: Fully optimized for CUDA execution.

DEPENDENCIES:
    - Python >= 3.10
    - torch, pandas, numpy, tqdm, omegaconf
    - df2023xai.models.factory, df2023xai.data.dataset (Internal)
===============================================================================
"""
import torch
import numpy as np
import os
import sys
import pandas as pd  # Make sure pandas is installed
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from df2023xai.models.factory import load_model_from_dir
from df2023xai.data.dataset import ForgerySegDataset

def get_manipulation_type(filepath):
    """
    Parses DF2023 filenames.
    Format: COCO_DF_[TYPE]... 
    Pos 0 of 3rd block: C=Copy-Move, S=Splicing, R=Inpainting, E=Enhancement
    """
    try:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        
        # Logic: COCO_DF_[CODE]...
        if len(parts) >= 3:
            code_block = parts[2]
            type_char = code_block[0]
            
            if type_char == 'S': return "Splicing"
            if type_char == 'C': return "Copy-Move"
            if type_char == 'R': return "Inpainting"
            if type_char == 'E': return "Enhancement"
            
        return "Unknown"
    except Exception:
        return "Error"

def compute_metrics(logits, mask):
    preds = (torch.sigmoid(logits) > 0.5).long()
    targets = (mask > 0).long()
    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (preds.sum().item() + targets.sum().item() + 1e-7)
    return iou, dice

def run_stratified(config_path):
    cfg = OmegaConf.load(config_path)
    
    # 1. Load the Manifest CSV directly to get paths reliably
    print(f"[*] Loading manifest: {cfg.data.manifest_csv}")
    try:
        df = pd.read_csv(cfg.data.manifest_csv)
        # Assumes column name is 'image_path' or 'image'. Adjust if needed.
        if 'image_path' in df.columns:
            all_paths = df['image_path'].tolist()
        elif 'image' in df.columns:
            all_paths = df['image'].tolist()
        else:
            # Fallback: assume first column
            all_paths = df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"[!] Error loading CSV: {e}")
        return

    # 2. Load Dataset & Model
    ds = ForgerySegDataset(
        manifest_csv=cfg.data.manifest_csv,
        img_size=cfg.data.img_size,
        split="test"
    )
    
    model_config = cfg.models[0]
    model = load_model_from_dir(model_config.path).eval().cuda()
    
    print(f"[*] Profiling {model_config.name} on {len(ds)} samples...")
    
    # Storage
    metrics_by_type = {}
    
    # CRITICAL: shuffle=False ensures alignment with all_paths list
    dl = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    
    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(dl)):
            img = img.cuda()
            mask = mask.cuda()
            
            # 3. Retrieve path from our loaded list, NOT the dataset object
            try:
                file_path = all_paths[i]
                m_type = get_manipulation_type(file_path)
            except IndexError:
                m_type = "IndexError"
            except Exception:
                m_type = "Error"

            if m_type not in metrics_by_type:
                metrics_by_type[m_type] = {'iou': [], 'dice': []}
            
            logits = model(img)
            iou, dice = compute_metrics(logits, mask)
            
            metrics_by_type[m_type]['iou'].append(iou)
            metrics_by_type[m_type]['dice'].append(dice)

    print("\n" + "="*65)
    print(f"{'Manipulation Type':<20} | {'IoU':<10} | {'F1-Score':<10} | {'Samples':<10}")
    print("-" * 65)
    
    for m_type in sorted(metrics_by_type.keys()):
        scores = metrics_by_type[m_type]
        avg_iou = np.mean(scores['iou'])
        avg_dice = np.mean(scores['dice'])
        count = len(scores['iou'])
        print(f"{m_type:<20} | {avg_iou:.4f}     | {avg_dice:.4f}     | {count:<10}")
    print("="*65 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m df2023xai.cli.run_stratified <config_path>")
    else:
        run_stratified(sys.argv[1])
