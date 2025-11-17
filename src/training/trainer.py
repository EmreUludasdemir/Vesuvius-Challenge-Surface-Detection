"""
Training loop for surface detection models.

Includes:
- Training and validation loops
- Metric computation
- Checkpointing
- Learning rate scheduling
- Mixed precision training
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import wandb


class Trainer:
    """Trainer for surface detection models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        use_amp: bool = True,
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = False
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            loss_fn: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            use_amp: Use automatic mixed precision (default: True)
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Use Weights & Biases logging
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Create checkpoint directory
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                dice = compute_dice(outputs, masks)
                iou = compute_iou(outputs, masks)

            # Update tracking
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': dice,
                'iou': iou
            })

        # Average metrics
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_dice': total_dice / len(train_loader),
            'train_iou': total_iou / len(train_loader)
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0

        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")

        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)

            # Compute metrics
            dice = compute_dice(outputs, masks)
            iou = compute_iou(outputs, masks)

            # Update tracking
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': dice,
                'iou': iou
            })

        # Average metrics
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_dice': total_dice / len(val_loader),
            'val_iou': total_iou / len(val_loader)
        }

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_every: int = 5
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                metrics['lr'] = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.log_metrics(metrics)

            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint("best_model.pt")
                print(f"  -> Best model saved (val_loss: {self.best_val_loss:.4f})")

            print("-" * 50)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Checkpoint loaded from epoch {self.current_epoch}")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tracking systems."""
        # Update history
        if 'train_loss' in metrics:
            self.train_losses.append(metrics['train_loss'])
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'])

        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=self.current_epoch)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient.

    Args:
        pred: Predictions (logits)
        target: Ground truth
        threshold: Threshold for binarization

    Returns:
        Dice score
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute IoU (Jaccard Index).

    Args:
        pred: Predictions (logits)
        target: Ground truth
        threshold: Threshold for binarization

    Returns:
        IoU score
    """
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)

    return iou.item()


def compute_hausdorff_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Hausdorff distance (approximate).

    Args:
        pred: Predictions
        target: Ground truth

    Returns:
        Hausdorff distance
    """
    # This is a simplified version
    # For full implementation, use scipy.spatial.distance.directed_hausdorff
    # or medpy.metric.binary.hd

    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5

    # Convert to numpy
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # Find boundary pixels
    from scipy.ndimage import binary_erosion

    pred_boundary = pred_np ^ binary_erosion(pred_np)
    target_boundary = target_np ^ binary_erosion(target_np)

    # Get coordinates
    pred_points = np.argwhere(pred_boundary)
    target_points = np.argwhere(target_boundary)

    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0

    # Compute distances (simplified)
    from scipy.spatial.distance import cdist
    distances = cdist(pred_points, target_points, metric='euclidean')

    # Hausdorff distance
    hd = max(distances.min(axis=1).max(), distances.min(axis=0).max())

    return float(hd)


if __name__ == "__main__":
    # Example usage
    print("Trainer Example")
    print("="*50)

    from src.models.unet3d import UNet3DDepthInvariant
    from src.training.losses import CombinedLoss

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    model = UNet3DDepthInvariant(in_channels=65, out_channels=1, base_features=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        use_amp=True,
        checkpoint_dir=Path("checkpoints"),
        use_wandb=False
    )

    print("\nTrainer initialized successfully!")
    print("Ready for training with:")
    print("  - Mixed precision training")
    print("  - Combined BCE + Dice loss")
    print("  - AdamW optimizer")
    print("  - Automatic checkpointing")
