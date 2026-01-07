# src/df2023xai/models/factory.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

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
