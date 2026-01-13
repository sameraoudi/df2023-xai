import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from df2023xai.models.factory import load_model_from_dir
from df2023xai.data.dataset import ForgerySegDataset

def get_manipulation_type(filepath):
    """
    Infers manipulation type from the file path.
    Adjust these keywords based on your actual DF2023 file naming convention.
    """
    path_str = str(filepath).lower()
    if "splic" in path_str:
        return "Splicing"
    elif "copy" in path_str or "move" in path_str:
        return "Copy-Move"
    elif "inpaint" in path_str:
        return "Inpainting"
    else:
        return "Unknown"

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
    
    # Load Dataset
    ds = ForgerySegDataset(
        manifest_csv=cfg.data.manifest_csv,
        img_size=cfg.data.img_size,
        split="test"
    )
    
    # Load Model (Using the best SegFormer)
    # Assuming the first model in config is the one we want to profile
    model_path = cfg.models[0].path 
    model = load_model_from_dir(model_path).eval().cuda()
    
    print(f"[*] Profiling {cfg.models[0].name} by Manipulation Type...")
    
    # Storage
    metrics_by_type = {} # {'Splicing': {'iou': [], 'dice': []}, ...}

    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4)
    
    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(dl)):
            img = img.cuda()
            mask = mask.cuda()
            
            # Identify Type (Need access to file path from dataset)
            # Assuming ds.samples[i] contains the path, or similar attribute
            # If ds[i] only returns tensors, we need to inspect the internal list
            try:
                # Adjust 'samples' to whatever your dataset class uses to store paths
                file_path = ds.samples[i][0] if hasattr(ds, 'samples') else "unknown_splicing" 
                m_type = get_manipulation_type(file_path)
            except:
                m_type = "Unknown"

            if m_type not in metrics_by_type:
                metrics_by_type[m_type] = {'iou': [], 'dice': []}
            
            logits = model(img)
            iou, dice = compute_metrics(logits, mask)
            
            metrics_by_type[m_type]['iou'].append(iou)
            metrics_by_type[m_type]['dice'].append(dice)

    # Print Report
    print(f"\n{'Manipulation Type':<20} | {'IoU':<10} | {'F1-Score':<10} | {'Samples':<10}")
    print("-" * 60)
    for m_type, scores in metrics_by_type.items():
        avg_iou = np.mean(scores['iou'])
        avg_dice = np.mean(scores['dice'])
        count = len(scores['iou'])
        print(f"{m_type:<20} | {avg_iou:.4f}     | {avg_dice:.4f}     | {count:<10}")

if __name__ == "__main__":
    import sys
    run_stratified(sys.argv[1])
