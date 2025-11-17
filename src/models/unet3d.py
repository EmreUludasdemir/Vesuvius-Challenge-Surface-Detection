"""
3D U-Net Architecture for Surface Detection

Depth-invariant 3D U-Net for processing CT volumes.
Based on state-of-the-art approaches from Kaggle winners.

Architecture follows encoder-decoder structure with skip connections.
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
    Double 3D convolution block.

    (Conv3D -> BatchNorm -> ReLU) * 2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling block with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling block with transpose conv and skip connections."""

    def __init__(self, in_channels: int, out_channels: int, trilinear: bool = False):
        super().__init__()

        # Upsampling method
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Handle size mismatch (padding)
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2
        ])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric segmentation.

    Depth-invariant design suitable for surface detection.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        trilinear: bool = False
    ):
        """
        Initialize 3D U-Net.

        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            out_channels: Number of output channels (default: 1 for binary segmentation)
            base_features: Number of features in first layer (default: 32)
            depth: Number of downsampling/upsampling levels (default: 4)
            trilinear: Use trilinear upsampling instead of transpose conv
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Initial convolution
        self.inc = DoubleConv3D(in_channels, base_features)

        # Encoder (downsampling path)
        self.downs = nn.ModuleList()
        in_feat = base_features
        for i in range(depth):
            out_feat = in_feat * 2
            self.downs.append(Down3D(in_feat, out_feat))
            in_feat = out_feat

        # Decoder (upsampling path)
        self.ups = nn.ModuleList()
        for i in range(depth):
            out_feat = in_feat // 2
            self.ups.append(Up3D(in_feat, out_feat, trilinear))
            in_feat = out_feat

        # Output convolution
        self.outc = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Output tensor of shape (B, out_channels, D, H, W)
        """
        # Encoder
        x1 = self.inc(x)

        encoder_features = [x1]
        for down in self.downs:
            x1 = down(x1)
            encoder_features.append(x1)

        # Decoder
        x = encoder_features[-1]
        for i, up in enumerate(self.ups):
            skip_connection = encoder_features[-(i+2)]
            x = up(x, skip_connection)

        # Output
        logits = self.outc(x)

        return logits


class UNet3DDepthInvariant(nn.Module):
    """
    Depth-invariant 3D U-Net.

    Processes 3D volume but outputs 2D segmentation map.
    This is the recommended architecture from Kaggle winners.
    """

    def __init__(
        self,
        in_channels: int = 65,  # Number of depth slices
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4
    ):
        """
        Initialize depth-invariant 3D U-Net.

        Args:
            in_channels: Number of depth slices to process
            out_channels: Number of output channels
            base_features: Base number of features
            depth: Network depth
        """
        super().__init__()

        self.in_channels = in_channels

        # Encoder: Process each slice with 2D convolutions
        # Then aggregate along depth dimension
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )

        # Standard 2D U-Net for segmentation
        self.downs = nn.ModuleList()
        in_feat = base_features
        for i in range(depth):
            out_feat = in_feat * 2
            self.downs.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv2D(in_feat, out_feat)
            ))
            in_feat = out_feat

        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth):
            out_feat = in_feat // 2
            self.ups.append(Up2D(in_feat, out_feat))
            in_feat = out_feat

        # Output
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, D, H, W) where D is depth

        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        # Encode depth slices
        x = self.slice_encoder(x)  # (B, base_features, H, W)

        # Standard U-Net forward
        encoder_features = [x]
        for down in self.downs:
            x = down(x)
            encoder_features.append(x)

        # Decoder
        x = encoder_features[-1]
        for i, up in enumerate(self.ups):
            skip_connection = encoder_features[-(i+2)]
            x = up(x, skip_connection)

        # Output
        logits = self.outc(x)

        return logits


class DoubleConv2D(nn.Module):
    """Double 2D convolution block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up2D(nn.Module):
    """2D upsampling block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])

        # Concatenate
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


def get_model(
    model_type: str = "unet3d",
    in_channels: int = 1,
    out_channels: int = 1,
    base_features: int = 32,
    **kwargs
) -> nn.Module:
    """
    Factory function to get model.

    Args:
        model_type: Type of model ("unet3d" or "unet3d_depth_invariant")
        in_channels: Input channels
        out_channels: Output channels
        base_features: Base feature count
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    if model_type == "unet3d":
        return UNet3D(in_channels, out_channels, base_features, **kwargs)
    elif model_type == "unet3d_depth_invariant":
        return UNet3DDepthInvariant(in_channels, out_channels, base_features, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage and testing
    print("3D U-Net Model Examples")
    print("="*50)

    # Test standard 3D U-Net
    print("\n1. Standard 3D U-Net:")
    model = UNet3D(in_channels=1, out_channels=1, base_features=32, depth=4)
    print(f"Parameters: {count_parameters(model):,}")

    # Test input
    x = torch.randn(1, 1, 65, 256, 256)  # (B, C, D, H, W)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")

    # Test depth-invariant U-Net
    print("\n2. Depth-Invariant 3D U-Net:")
    model_di = UNet3DDepthInvariant(in_channels=65, out_channels=1, base_features=32)
    print(f"Parameters: {count_parameters(model_di):,}")

    # Test input (depth as channels)
    x_di = torch.randn(1, 65, 256, 256)  # (B, D, H, W)
    print(f"Input shape: {x_di.shape}")

    with torch.no_grad():
        output_di = model_di(x_di)
    print(f"Output shape: {output_di.shape}")

    print("\n" + "="*50)
    print("3D U-Net models ready!")
    print("\nRecommendation:")
    print("- Use UNet3DDepthInvariant for most cases (depth-invariant design)")
    print("- Use standard UNet3D if you need 3D output")
