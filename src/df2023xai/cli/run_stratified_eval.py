import torch
import numpy as np
import os
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from df2023xai.models.factory import load_model_from_dir
from df2023xai.data.dataset import ForgerySegDataset

def get_manipulation_type(filepath):
    """
    Parses DF2023 filenames based on the Semantics Doc.
    Format: COCO_DF_[TYPE]...
    Pos 0 of 3rd block: C=Copy-Move, S=Splicing, R=Removal(Inpainting), E=Enhancement
    """
    try:
        filename = os.path.basename(filepath)
        # Example: COCO_DF_C100B00000_00508013.jpg
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # Code block is usually the 3rd part (Index 2) -> "C100B00000"
            code_block = parts[2]
            type_char = code_block[0]
            
            if type_char == 'S':
                return "Splicing"
            elif type_char == 'C':
                return "Copy-Move"
            elif type_char == 'R':
                return "Inpainting" # Mapped from 'Removal'
            elif type_char == 'E':
                return "Enhancement"
            
        return "Unknown"
    except Exception as e:
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
    
    # Load Dataset (Using Test Split)
    ds = ForgerySegDataset(
        manifest_csv=cfg.data.manifest_csv,
        img_size=cfg.data.img_size,
        split="test"
    )
    
    # Load Model (First model in config)
    model_config = cfg.models[0]
    model_path = model_config.path 
    model = load_model_from_dir(model_path).eval().cuda()
    
    print(f"[*] Profiling {model_config.name} on {len(ds)} samples...")
    print(f"[*] Parsing Logic: COCO_DF_[Char]... (S=Splicing, C=Copy-Move, R=Inpainting)")

    metrics_by_type = {}
    
    # Use batch_size=1 for accurate per-file analysis
    dl = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    
    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(dl)):
            img = img.cuda()
            mask = mask.cuda()
            
            # Retrieve path from dataset
            # Note: Ensure your ForgerySegDataset.__getitem__ returns path or you access it via ds.samples
            # Assuming ds.samples is a list of [img_path, mask_path]
            try:
                file_path = ds.samples[i][0]
                m_type = get_manipulation_type(file_path)
            except:
                m_type = "Unknown"

            if m_type not in metrics_by_type:
                metrics_by_type[m_type] = {'iou': [], 'dice': []}
            
            logits = model(img)
            iou, dice = compute_metrics(logits, mask)
            
            metrics_by_type[m_type]['iou'].append(iou)
            metrics_by_type[m_type]['dice'].append(dice)

    print("\n" + "="*65)
    print(f"{'Manipulation Type':<20} | {'IoU':<10} | {'F1-Score':<10} | {'Samples':<10}")
    print("-" * 65)
    
    # Sort for consistent table output
    for m_type in sorted(metrics_by_type.keys()):
        scores = metrics_by_type[m_type]
        avg_iou = np.mean(scores['iou'])
        avg_dice = np.mean(scores['dice'])
        count = len(scores['iou'])
        print(f"{m_type:<20} | {avg_iou:.4f}     | {avg_dice:.4f}     | {count:<10}")
    print("="*65 + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m df2023xai.cli.run_stratified <config_path>")
    else:
        run_stratified(sys.argv[1])
