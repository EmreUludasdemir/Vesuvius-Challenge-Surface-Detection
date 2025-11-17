"""
Data augmentation utilities for Vesuvius Challenge Surface Detection.

CRITICAL: Z-translation augmentation is the single most important augmentation
for this task. It provides translation invariance along the depth axis.

This module includes:
- Z-translation (depth shift) - MOST IMPORTANT
- 2D augmentations per-slice
- Combined augmentation pipeline
"""

from typing import Optional, Tuple, Union, Callable
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class ZTranslationAugment:
    """
    Z-translation augmentation - shifts slices along depth axis.

    This is THE MOST CRITICAL augmentation for surface detection.
    Provides translation invariance along depth dimension.
    """

    def __init__(self, max_shift: int = 5, p: float = 0.5):
        """
        Initialize Z-translation augmentation.

        Args:
            max_shift: Maximum number of slices to shift (default: 5)
            p: Probability of applying augmentation (default: 0.5)
        """
        self.max_shift = max_shift
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply Z-translation to volume.

        Args:
            volume: 3D volume of shape (D, H, W)

        Returns:
            Augmented volume with shifted slices
        """
        if np.random.rand() > self.p:
            return volume

        # Random shift amount
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)

        if shift == 0:
            return volume

        # Circular shift along depth axis
        volume_shifted = np.roll(volume, shift, axis=0)

        return volume_shifted


class PerSliceAugment:
    """
    Apply 2D augmentations independently to each slice.

    Supports all standard albumentations transforms.
    """

    def __init__(self, transform: A.Compose):
        """
        Initialize per-slice augmentation.

        Args:
            transform: Albumentations Compose object
        """
        self.transform = transform

    def __call__(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply 2D augmentation to each slice independently.

        Args:
            volume: 3D volume of shape (D, H, W)
            mask: Optional 2D mask of shape (H, W)

        Returns:
            Tuple of (augmented_volume, augmented_mask)
        """
        D, H, W = volume.shape
        augmented_volume = np.zeros_like(volume)

        # Apply same transform to all slices
        # (mask is 2D and shared across slices)
        for d in range(D):
            if mask is not None:
                transformed = self.transform(image=volume[d], mask=mask)
                augmented_volume[d] = transformed['image']

                # Only update mask once (it's the same for all slices)
                if d == 0:
                    augmented_mask = transformed['mask']
            else:
                transformed = self.transform(image=volume[d])
                augmented_volume[d] = transformed['image']
                augmented_mask = None

        return augmented_volume, augmented_mask


def get_training_augmentations(
    image_size: int = 256,
    use_heavy_augs: bool = True
) -> A.Compose:
    """
    Get standard 2D training augmentations.

    Args:
        image_size: Size of input images
        use_heavy_augs: Whether to use heavy augmentations (recommended)

    Returns:
        Albumentations Compose object
    """
    if use_heavy_augs:
        transforms = [
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),

            # Intensity transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
            ], p=0.5),

            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),

            # Dropout augmentations
            A.CoarseDropout(
                max_holes=8,
                max_height=image_size // 8,
                max_width=image_size // 8,
                p=0.5
            ),

            # Grid distortion
            A.OneOf([
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=1.0
                ),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=0.5, p=1.0),
            ], p=0.3),
        ]
    else:
        # Minimal augmentations
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]

    return A.Compose(transforms)


def get_validation_transforms() -> A.Compose:
    """
    Get validation transforms (no augmentation, just normalization).

    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # No augmentations for validation
    ])


class VolumeAugmentationPipeline:
    """
    Complete augmentation pipeline for 3D volumes.

    Combines Z-translation with per-slice 2D augmentations.
    """

    def __init__(
        self,
        z_translation_prob: float = 0.5,
        max_z_shift: int = 5,
        image_size: int = 256,
        use_heavy_augs: bool = True,
        is_training: bool = True
    ):
        """
        Initialize augmentation pipeline.

        Args:
            z_translation_prob: Probability of Z-translation
            max_z_shift: Maximum Z-axis shift
            image_size: Image size for 2D augmentations
            use_heavy_augs: Whether to use heavy augmentations
            is_training: Training or validation mode
        """
        self.is_training = is_training

        if is_training:
            # Z-translation augmentation (CRITICAL)
            self.z_translation = ZTranslationAugment(
                max_shift=max_z_shift,
                p=z_translation_prob
            )

            # 2D augmentations
            self.slice_augment = PerSliceAugment(
                get_training_augmentations(image_size, use_heavy_augs)
            )
        else:
            self.z_translation = None
            self.slice_augment = PerSliceAugment(
                get_validation_transforms()
            )

    def __call__(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply full augmentation pipeline.

        Args:
            volume: 3D volume of shape (D, H, W)
            mask: Optional 2D mask

        Returns:
            Tuple of (augmented_volume, augmented_mask)
        """
        # Step 1: Z-translation (if training)
        if self.is_training and self.z_translation is not None:
            volume = self.z_translation(volume)

        # Step 2: Per-slice 2D augmentations
        volume, mask = self.slice_augment(volume, mask)

        return volume, mask


class MixupAugmentation:
    """
    Mixup augmentation for 3D volumes.

    Less common but can help with regularization.
    """

    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        """
        Initialize Mixup augmentation.

        Args:
            alpha: Beta distribution parameter
            p: Probability of applying mixup
        """
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        volume1: np.ndarray,
        volume2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup to two samples.

        Args:
            volume1: First volume
            volume2: Second volume
            label1: First label
            label2: Second label

        Returns:
            Mixed volume and label
        """
        if np.random.rand() > self.p:
            return volume1, label1

        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix volumes and labels
        mixed_volume = lam * volume1 + (1 - lam) * volume2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_volume, mixed_label


class CutMixAugmentation:
    """
    CutMix augmentation for 3D volumes.

    Replaces random regions with patches from another sample.
    """

    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Initialize CutMix augmentation.

        Args:
            alpha: Beta distribution parameter
            p: Probability of applying cutmix
        """
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        volume1: np.ndarray,
        volume2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply CutMix to two samples.

        Args:
            volume1: First volume (D, H, W)
            volume2: Second volume (D, H, W)
            label1: First label (H, W)
            label2: Second label (H, W)

        Returns:
            Mixed volume, mixed label, and lambda (mixing ratio)
        """
        if np.random.rand() > self.p:
            return volume1, label1, 1.0

        D, H, W = volume1.shape

        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # Random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        mixed_volume = volume1.copy()
        mixed_volume[:, y1:y2, x1:x2] = volume2[:, y1:y2, x1:x2]

        mixed_label = label1.copy()
        mixed_label[y1:y2, x1:x2] = label2[y1:y2, x1:x2]

        # Compute actual lambda
        lam_actual = 1 - ((x2 - x1) * (y2 - y1) / (H * W))

        return mixed_volume, mixed_label, lam_actual


def visualize_augmentations(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_examples: int = 4
):
    """
    Visualize augmentation examples (for debugging).

    Args:
        volume: Input volume
        mask: Optional mask
        n_examples: Number of augmented examples to show
    """
    import matplotlib.pyplot as plt

    pipeline = VolumeAugmentationPipeline(is_training=True)

    fig, axes = plt.subplots(n_examples + 1, 3, figsize=(12, 4 * (n_examples + 1)))

    # Original
    axes[0, 0].imshow(volume[32], cmap='gray')
    axes[0, 0].set_title('Original - Slice 32')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(volume.max(axis=0), cmap='gray')
    axes[0, 1].set_title('Original - Max Projection')
    axes[0, 1].axis('off')

    if mask is not None:
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title('Original - Mask')
        axes[0, 2].axis('off')

    # Augmented examples
    for i in range(n_examples):
        aug_vol, aug_mask = pipeline(volume.copy(), mask.copy() if mask is not None else None)

        axes[i+1, 0].imshow(aug_vol[32], cmap='gray')
        axes[i+1, 0].set_title(f'Augmented {i+1} - Slice 32')
        axes[i+1, 0].axis('off')

        axes[i+1, 1].imshow(aug_vol.max(axis=0), cmap='gray')
        axes[i+1, 1].set_title(f'Augmented {i+1} - Max Projection')
        axes[i+1, 1].axis('off')

        if aug_mask is not None:
            axes[i+1, 2].imshow(aug_mask, cmap='gray')
            axes[i+1, 2].set_title(f'Augmented {i+1} - Mask')
            axes[i+1, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Augmentation Examples:")
    print("="*50)

    # Create dummy volume
    volume = np.random.rand(65, 256, 256).astype(np.float32)
    mask = np.random.randint(0, 2, (256, 256)).astype(np.uint8)

    # Test Z-translation
    print("\n1. Z-Translation Augmentation:")
    z_aug = ZTranslationAugment(max_shift=5, p=1.0)
    volume_shifted = z_aug(volume)
    print(f"Original shape: {volume.shape}")
    print(f"Shifted shape: {volume_shifted.shape}")
    print(f"Shift applied: {not np.array_equal(volume, volume_shifted)}")

    # Test per-slice augmentation
    print("\n2. Per-Slice Augmentation:")
    transforms = get_training_augmentations(image_size=256)
    slice_aug = PerSliceAugment(transforms)
    aug_volume, aug_mask = slice_aug(volume, mask)
    print(f"Augmented volume shape: {aug_volume.shape}")
    print(f"Augmented mask shape: {aug_mask.shape}")

    # Test full pipeline
    print("\n3. Full Augmentation Pipeline:")
    pipeline = VolumeAugmentationPipeline(is_training=True)
    aug_volume, aug_mask = pipeline(volume, mask)
    print(f"Final augmented volume shape: {aug_volume.shape}")
    print(f"Final augmented mask shape: {aug_mask.shape}")

    print("\n" + "="*50)
    print("Augmentation pipeline ready!")
