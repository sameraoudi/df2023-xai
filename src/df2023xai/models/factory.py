# src/df2023xai/models/factory.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import os
import json

def build_model(name: str, num_classes: int = 2, pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Standardized Model Factory for Q1 Reproducibility.
    Directly wraps segmentation_models_pytorch (SMP) to avoid custom implementation bugs.
    """
    
    # 1. Architecture Configuration
    # We map your config names (segformer_b2, unet_r34) to SMP parameters.
    if "segformer" in name:
        # Extract encoder version (b0-b5) or default to b2
        encoder = "mit_b2"
        if "b0" in name: encoder = "mit_b0"
        elif "b1" in name: encoder = "mit_b1"
        elif "b3" in name: encoder = "mit_b3"
        elif "b4" in name: encoder = "mit_b4"
        elif "b5" in name: encoder = "mit_b5"
        
        weights = "imagenet" if pretrained else None
        print(f"[Factory] Building SegFormer (Encoder: {encoder}, Weights: {weights})")
        
        # SegFormer in SMP
        return smp.Segformer(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
            **kwargs
        )

    elif "unet" in name:
        # Default to ResNet34 if not specified
        encoder = "resnet34"
        if "r18" in name: encoder = "resnet18"
        if "r50" in name: encoder = "resnet50"
        
        weights = "imagenet" if pretrained else None
        print(f"[Factory] Building U-Net (Encoder: {encoder}, Weights: {weights})")
        
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
            **kwargs
        )

    else:
        raise ValueError(f"Architecture '{name}' not supported. Use 'segformer_bX' or 'unet_rXX'.")

def load_model_from_dir(model_dir: str) -> nn.Module:
    """
    Robustly loads a trained model for XAI/Evaluation.
    Auto-detects architecture and num_classes from config_train.json.
    """
    # 1. Locate Config and Checkpoint
    cfg_path = os.path.join(model_dir, "config_train.json")
    # Supports best.pt (standard) or last.pt (fallback)
    ckpt_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "last.pt")
        
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find config.json or .pt checkpoint in {model_dir}")

    # 2. Load Metadata
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    model_name = cfg.get("model", {}).get("name", "unet_r34")
    # CRITICAL: Default to 1 (Binary) if not specified, matching our new protocol
    num_classes = int(cfg.get("model", {}).get("num_classes", 1))

    print(f"[Loader] Found {model_name} (classes={num_classes}) in {model_dir}")

    # 3. Build & Load
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    
    # Handle state_dict keys (some checkpoints save {'state_dict': ...} vs raw dict)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove "module." prefix if trained with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    return model
