"""
Data preprocessing utilities for Vesuvius Challenge Surface Detection.

This module handles:
- Loading 3D CT volumes from stacked .tif files
- Normalization and windowing
- Patch extraction for training
- Surface volume creation
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Union

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


class VolumeLoader:
    """Load and preprocess 3D CT volumes from fragment/scroll data."""

    def __init__(
        self,
        data_root: Union[str, Path],
        fragment_id: str,
        num_slices: int = 65,
        normalize: bool = True,
        windowing: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize volume loader.

        Args:
            data_root: Root directory containing fragment data
            fragment_id: Fragment/scroll identifier
            num_slices: Number of slices to load (default: 65)
            normalize: Whether to normalize to [0, 1] range
            windowing: Optional (min, max) values for intensity windowing
        """
        self.data_root = Path(data_root)
        self.fragment_id = fragment_id
        self.num_slices = num_slices
        self.normalize = normalize
        self.windowing = windowing

        self.volume_path = self.data_root / fragment_id / "surface_volume"
        if not self.volume_path.exists():
            raise ValueError(f"Volume path does not exist: {self.volume_path}")

    def load_volume(self, start_slice: int = 0) -> np.ndarray:
        """
        Load 3D volume from stacked .tif files.

        Args:
            start_slice: Starting slice index (default: 0)

        Returns:
            3D numpy array of shape (D, H, W) where D is depth
        """
        slices = []

        for i in range(start_slice, start_slice + self.num_slices):
            slice_path = self.volume_path / f"{i:02d}.tif"

            if not slice_path.exists():
                raise FileNotFoundError(f"Slice not found: {slice_path}")

            # Load as grayscale
            slice_img = np.array(Image.open(slice_path))
            slices.append(slice_img)

        volume = np.stack(slices, axis=0)  # Shape: (D, H, W)

        # Apply windowing if specified
        if self.windowing is not None:
            volume = self._apply_windowing(volume, self.windowing)

        # Normalize to [0, 1] range
        if self.normalize:
            volume = self._normalize(volume)

        return volume

    def load_mask(self) -> Optional[np.ndarray]:
        """
        Load mask.png if available.

        Returns:
            2D binary mask or None if not found
        """
        mask_path = self.data_root / self.fragment_id / "mask.png"

        if not mask_path.exists():
            return None

        mask = np.array(Image.open(mask_path).convert('L'))
        return (mask > 0).astype(np.uint8)

    def load_labels(self) -> Optional[np.ndarray]:
        """
        Load inklabels.png ground truth if available.

        Returns:
            2D binary label mask or None if not found
        """
        labels_path = self.data_root / self.fragment_id / "inklabels.png"

        if not labels_path.exists():
            return None

        labels = np.array(Image.open(labels_path).convert('L'))
        return (labels > 0).astype(np.uint8)

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range."""
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            return (volume - vmin) / (vmax - vmin)
        return volume

    @staticmethod
    def _apply_windowing(volume: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
        """Apply intensity windowing (clipping)."""
        wmin, wmax = window
        return np.clip(volume, wmin, wmax)


class PatchExtractor:
    """Extract training patches from 3D volumes."""

    def __init__(
        self,
        patch_size: int = 256,
        stride: Optional[int] = None,
        balanced_sampling: bool = True,
        surface_ratio: float = 0.5
    ):
        """
        Initialize patch extractor.

        Args:
            patch_size: Size of square patches to extract
            stride: Stride for sliding window (default: patch_size for non-overlapping)
            balanced_sampling: Whether to balance surface/non-surface samples
            surface_ratio: Ratio of surface patches when balanced_sampling=True
        """
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.balanced_sampling = balanced_sampling
        self.surface_ratio = surface_ratio

    def extract_patches(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract patches from volume using sliding window.

        Args:
            volume: 3D volume of shape (D, H, W)
            mask: Optional 2D mask for valid regions
            labels: Optional 2D labels for supervised training

        Returns:
            Tuple of (volume_patches, label_patches, coordinates)
        """
        D, H, W = volume.shape

        volume_patches = []
        label_patches = []
        coordinates = []

        # Generate patch coordinates
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                # Check if patch is valid (if mask provided)
                if mask is not None:
                    patch_mask = mask[y:y+self.patch_size, x:x+self.patch_size]
                    # Skip if less than 50% of patch is valid
                    if patch_mask.mean() < 0.5:
                        continue

                # Extract volume patch
                vol_patch = volume[:, y:y+self.patch_size, x:x+self.patch_size]

                # Extract label patch if available
                if labels is not None:
                    label_patch = labels[y:y+self.patch_size, x:x+self.patch_size]

                    # Balanced sampling filter
                    if self.balanced_sampling:
                        has_surface = label_patch.sum() > 0

                        # Random sampling based on surface_ratio
                        if has_surface:
                            if np.random.rand() > self.surface_ratio:
                                continue
                        else:
                            if np.random.rand() > (1 - self.surface_ratio):
                                continue

                    label_patches.append(label_patch)

                volume_patches.append(vol_patch)
                coordinates.append((y, x))

        return volume_patches, label_patches, coordinates

    def reconstruct_from_patches(
        self,
        patches: List[np.ndarray],
        coordinates: List[Tuple[int, int]],
        output_shape: Tuple[int, int],
        use_gaussian_weighting: bool = True
    ) -> np.ndarray:
        """
        Reconstruct full-size prediction from overlapping patches.

        Args:
            patches: List of prediction patches
            coordinates: List of (y, x) coordinates for each patch
            output_shape: Target output shape (H, W)
            use_gaussian_weighting: Whether to use gaussian weighting for overlaps

        Returns:
            Reconstructed 2D prediction map
        """
        H, W = output_shape
        output = np.zeros((H, W), dtype=np.float32)
        weights = np.zeros((H, W), dtype=np.float32)

        # Create gaussian weight kernel if needed
        if use_gaussian_weighting:
            weight_kernel = self._create_gaussian_kernel(self.patch_size)
        else:
            weight_kernel = np.ones((self.patch_size, self.patch_size))

        # Accumulate weighted patches
        for patch, (y, x) in zip(patches, coordinates):
            output[y:y+self.patch_size, x:x+self.patch_size] += patch * weight_kernel
            weights[y:y+self.patch_size, x:x+self.patch_size] += weight_kernel

        # Normalize by weights
        output = np.divide(output, weights, where=weights > 0)

        return output

    @staticmethod
    def _create_gaussian_kernel(size: int, sigma: Optional[float] = None) -> np.ndarray:
        """Create 2D gaussian kernel for patch weighting."""
        if sigma is None:
            sigma = size / 6  # ~99% of mass within patch

        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / kernel.max()


def create_surface_volume(
    volume_path: Path,
    surface_points: np.ndarray,
    output_path: Path,
    radius: int = 32
) -> None:
    """
    Create surface volume centered on detected surface.

    Args:
        volume_path: Path to input volume directory
        surface_points: Array of surface coordinates (N, 3) - (z, y, x)
        output_path: Path to save surface volume
        radius: Number of slices above/below surface (default: 32)
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Load reference slice to get dimensions
    ref_slice = np.array(Image.open(volume_path / "00.tif"))
    H, W = ref_slice.shape

    # Create empty surface volume
    surface_volume = np.zeros((2 * radius + 1, H, W), dtype=ref_slice.dtype)

    # For each pixel, extract slices centered on surface
    for z, y, x in tqdm(surface_points, desc="Creating surface volume"):
        z = int(z)
        for i, offset in enumerate(range(-radius, radius + 1)):
            slice_idx = z + offset

            # Load slice if it exists
            slice_path = volume_path / f"{slice_idx:02d}.tif"
            if slice_path.exists():
                slice_img = np.array(Image.open(slice_path))
                surface_volume[i, y, x] = slice_img[y, x]

    # Save surface volume slices
    for i in range(2 * radius + 1):
        Image.fromarray(surface_volume[i]).save(output_path / f"{i:02d}.tif")

    print(f"Surface volume saved to {output_path}")


def standardize_volume(volume: np.ndarray, mean: float = 0.5, std: float = 0.25) -> np.ndarray:
    """
    Standardize volume to zero mean and unit variance.

    Args:
        volume: Input volume
        mean: Target mean (default: 0.5)
        std: Target std (default: 0.25)

    Returns:
        Standardized volume
    """
    return (volume - volume.mean()) / (volume.std() + 1e-8) * std + mean


if __name__ == "__main__":
    # Example usage
    print("Volume Loader Example:")
    print("="*50)

    # Example configuration
    data_root = Path("data/raw")
    fragment_id = "fragment_1"

    # Check if example data exists
    if not (data_root / fragment_id).exists():
        print(f"Example data not found at {data_root / fragment_id}")
        print("Please download fragment data first.")
    else:
        # Load volume
        loader = VolumeLoader(data_root, fragment_id, num_slices=65)
        volume = loader.load_volume()
        print(f"Loaded volume shape: {volume.shape}")
        print(f"Volume dtype: {volume.dtype}")
        print(f"Value range: [{volume.min():.3f}, {volume.max():.3f}]")

        # Load mask and labels
        mask = loader.load_mask()
        labels = loader.load_labels()

        if mask is not None:
            print(f"Mask shape: {mask.shape}")
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
            print(f"Surface coverage: {labels.mean()*100:.2f}%")

        # Extract patches
        print("\n" + "="*50)
        print("Patch Extraction Example:")

        extractor = PatchExtractor(patch_size=256, stride=128, balanced_sampling=True)
        vol_patches, label_patches, coords = extractor.extract_patches(
            volume, mask, labels
        )

        print(f"Extracted {len(vol_patches)} patches")
        if vol_patches:
            print(f"Patch shape: {vol_patches[0].shape}")
