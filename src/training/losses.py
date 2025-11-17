"""
Loss functions for surface detection.

Recommended: Combined BCE + Dice Loss with label smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice coefficient: 2 * |X ∩ Y| / (|X| + |Y|)
    """

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        """
        Initialize Dice Loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.

        Args:
            pred: Predictions (logits) of shape (B, C, H, W) or (B, C, D, H, W)
            target: Ground truth of shape (B, C, H, W) or (B, C, D, H, W)

        Returns:
            Dice loss (scalar)
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # Dice coefficient
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth + self.eps)

        # Dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss.

    This is the recommended loss for surface detection.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.0,
        pos_weight: Optional[float] = None
    ):
        """
        Initialize Combined Loss.

        Args:
            bce_weight: Weight for BCE loss (default: 0.5)
            dice_weight: Weight for Dice loss (default: 0.5)
            label_smoothing: Label smoothing factor (default: 0.0, recommended: 0.1)
            pos_weight: Positive class weight for BCE (None = balanced)
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing

        # BCE Loss
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()

        # Dice Loss
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predictions (logits)
            target: Ground truth

        Returns:
            Combined loss
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            target_smooth = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            target_smooth = target

        # BCE Loss
        bce_loss = self.bce(pred, target_smooth)

        # Dice Loss
        dice_loss = self.dice(pred, target)

        # Combined
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal Loss = -α(1-p)^γ * log(p)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            pred: Predictions (logits)
            target: Ground truth

        Returns:
            Focal loss
        """
        # Convert to probabilities
        pred_prob = torch.sigmoid(pred)

        # Focal loss components
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Focal loss
        focal_loss = focal_weight * bce

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss.

    Good for handling false positives and false negatives differently.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        """
        Initialize Tversky Loss.

        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky Loss.

        Args:
            pred: Predictions (logits)
            target: Ground truth

        Returns:
            Tversky loss
        """
        # Apply sigmoid
        pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # True positives, false positives, false negatives
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Tversky loss
        return 1 - tversky


class IoULoss(nn.Module):
    """
    IoU (Jaccard) Loss.

    IoU = |X ∩ Y| / |X ∪ Y|
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU Loss.

        Args:
            pred: Predictions (logits)
            target: Ground truth

        Returns:
            IoU loss
        """
        # Apply sigmoid
        pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # Intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        # IoU
        iou = (intersection + self.smooth) / (union + self.smooth)

        # IoU loss
        return 1 - iou


def get_loss_function(
    loss_type: str = "combined",
    **kwargs
) -> nn.Module:
    """
    Factory function to get loss function.

    Args:
        loss_type: Type of loss ("bce", "dice", "combined", "focal", "tversky", "iou")
        **kwargs: Loss-specific arguments

    Returns:
        Loss function
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == "dice":
        return DiceLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    elif loss_type == "focal":
        return FocalLoss(**kwargs)
    elif loss_type == "tversky":
        return TverskyLoss(**kwargs)
    elif loss_type == "iou":
        return IoULoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Example usage and testing
    print("Loss Functions Test")
    print("="*50)

    # Create dummy data
    batch_size = 4
    pred = torch.randn(batch_size, 1, 256, 256)  # Logits
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()

    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {target.shape}")

    # Test each loss
    print("\n1. BCE Loss:")
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(pred, target)
    print(f"Loss value: {loss.item():.4f}")

    print("\n2. Dice Loss:")
    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"Loss value: {loss.item():.4f}")

    print("\n3. Combined Loss (BCE + Dice):")
    combined_loss = CombinedLoss(bce_weight=0.5, dice_weight=0.5, label_smoothing=0.1)
    loss = combined_loss(pred, target)
    print(f"Loss value: {loss.item():.4f}")

    print("\n4. Focal Loss:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(pred, target)
    print(f"Loss value: {loss.item():.4f}")

    print("\n5. Tversky Loss:")
    tversky_loss = TverskyLoss(alpha=0.5, beta=0.5)
    loss = tversky_loss(pred, target)
    print(f"Loss value: {loss.item():.4f}")

    print("\n6. IoU Loss:")
    iou_loss = IoULoss()
    loss = iou_loss(pred, target)
    print(f"Loss value: {loss.item():.4f}")

    print("\n" + "="*50)
    print("Loss functions ready!")
    print("\nRecommendation:")
    print("- Use CombinedLoss with label_smoothing=0.1 for best results")
    print("- BCE weight: 0.5, Dice weight: 0.5")
