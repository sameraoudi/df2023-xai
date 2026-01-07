# src/df2023xai/train/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridForensicLoss(nn.Module):
    """
    Implements the Hybrid Loss (Equation 1) from the Paper:
    L_total = 0.5 * BCE (Eq 2) + 0.5 * SoftDice (Eq 3)
    
    EXPECTS: 
        logits: [B, 1, H, W] (Standard binary model output)
        target: [B, H, W]    (0 or 1)
    """
    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        super().__init__()
        self.w_bce = weight_bce
        self.w_dice = weight_dice
        # Replaces nn.CrossEntropyLoss with BCEWithLogits to match Paper Eq (2)
        # pos_weight can be tuned for class imbalance, but we use Dice for that.
        self.bce = nn.BCEWithLogitsLoss() 

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Align Dimensions
        # Target needs to be [B, 1, H, W] to match logits for BCE
        if target.dim() == 3:
            target = target.unsqueeze(1).float()
        
        # 2. Weighted Cross-Entropy (Paper Equation 2)
        loss_bce = self.bce(logits, target)
        
        # 3. Soft Dice Loss (Paper Equation 3)
        # Apply Sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)
        
        dims = (2, 3) # Calculate over H, W
        intersection = torch.sum(probs * target, dim=dims)
        cardinality = torch.sum(probs + target, dim=dims)
        
        # Smooth is 1e-7 in paper
        dice_score = (2. * intersection + 1e-7) / (cardinality + 1e-7)
        loss_dice = 1.0 - dice_score.mean()
        
        # 4. Total Hybrid Loss
        return (self.w_bce * loss_bce) + (self.w_dice * loss_dice)
