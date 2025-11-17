"""
Baseline 3D Sobel Surface Detection

Classical computer vision approach based on ThaumatoAnakalyptor method.
This serves as a fast baseline for comparison with deep learning approaches.

Steps:
1. 3D Sobel gradient computation
2. Gradient magnitude thresholding
3. Umbilicus direction filtering (optional)
4. Binary mask creation
5. Point cloud extraction
"""

from typing import Tuple, Optional
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology
from tqdm import tqdm


class SobelSurfaceDetector:
    """
    Baseline surface detector using 3D Sobel gradients.

    Based on ThaumatoAnakalyptor's classical CV approach.
    """

    def __init__(
        self,
        gradient_threshold: float = 0.1,
        use_second_derivative: bool = True,
        morphology_radius: int = 2,
        min_component_size: int = 100
    ):
        """
        Initialize Sobel surface detector.

        Args:
            gradient_threshold: Threshold for gradient magnitude (0-1)
            use_second_derivative: Whether to use second derivatives
            morphology_radius: Radius for morphological operations
            min_component_size: Minimum size of connected components
        """
        self.gradient_threshold = gradient_threshold
        self.use_second_derivative = use_second_derivative
        self.morphology_radius = morphology_radius
        self.min_component_size = min_component_size

    def compute_gradients_3d(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 3D gradients using Sobel operator.

        Args:
            volume: Input 3D volume (D, H, W)

        Returns:
            Tuple of (grad_z, grad_y, grad_x) gradient components
        """
        # Sobel filters for 3D
        # Axis 0 (Z/depth)
        grad_z = ndimage.sobel(volume, axis=0)

        # Axis 1 (Y/height)
        grad_y = ndimage.sobel(volume, axis=1)

        # Axis 2 (X/width)
        grad_x = ndimage.sobel(volume, axis=2)

        return grad_z, grad_y, grad_x

    def compute_gradient_magnitude(
        self,
        grad_z: np.ndarray,
        grad_y: np.ndarray,
        grad_x: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient magnitude from components.

        Args:
            grad_z: Z gradient
            grad_y: Y gradient
            grad_x: X gradient

        Returns:
            Gradient magnitude
        """
        magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
        return magnitude

    def apply_umbilicus_filter(
        self,
        grad_z: np.ndarray,
        grad_y: np.ndarray,
        grad_x: np.ndarray,
        umbilicus_center: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Filter gradients by direction toward umbilicus (scroll center).

        This helps select inner surface in scrolls.

        Args:
            grad_z, grad_y, grad_x: Gradient components
            umbilicus_center: (z, y, x) coordinates of scroll center

        Returns:
            Filtered gradient magnitude
        """
        D, H, W = grad_z.shape
        cz, cy, cx = umbilicus_center

        # Create coordinate grids
        z_grid, y_grid, x_grid = np.mgrid[0:D, 0:H, 0:W]

        # Vector pointing to umbilicus
        to_umbilicus_z = cz - z_grid
        to_umbilicus_y = cy - y_grid
        to_umbilicus_x = cx - x_grid

        # Normalize
        magnitude = np.sqrt(
            to_umbilicus_z**2 + to_umbilicus_y**2 + to_umbilicus_x**2
        ) + 1e-8

        to_umbilicus_z /= magnitude
        to_umbilicus_y /= magnitude
        to_umbilicus_x /= magnitude

        # Dot product with gradient
        dot_product = (
            grad_z * to_umbilicus_z +
            grad_y * to_umbilicus_y +
            grad_x * to_umbilicus_x
        )

        # Keep only gradients pointing toward umbilicus
        filtered = np.maximum(dot_product, 0)

        return filtered

    def threshold_gradients(
        self,
        gradient_magnitude: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Threshold gradient magnitude to create binary mask.

        Args:
            gradient_magnitude: Gradient magnitude array
            threshold: Threshold value (if None, uses self.gradient_threshold)

        Returns:
            Binary mask
        """
        if threshold is None:
            threshold = self.gradient_threshold

        # Normalize gradient magnitude to [0, 1]
        grad_normalized = (gradient_magnitude - gradient_magnitude.min()) / \
                         (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)

        # Threshold
        binary_mask = grad_normalized > threshold

        return binary_mask.astype(np.uint8)

    def apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up mask.

        Args:
            binary_mask: Binary surface mask

        Returns:
            Cleaned binary mask
        """
        # 3D morphological operations
        structuring_element = morphology.ball(self.morphology_radius)

        # Close small holes
        mask_closed = morphology.binary_closing(binary_mask, structuring_element)

        # Open to remove small objects
        mask_opened = morphology.binary_opening(mask_closed, structuring_element)

        return mask_opened.astype(np.uint8)

    def remove_small_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Remove small connected components.

        Args:
            binary_mask: Binary mask

        Returns:
            Filtered binary mask
        """
        # Label connected components
        labeled = morphology.label(binary_mask)

        # Remove small components
        cleaned = morphology.remove_small_objects(
            labeled,
            min_size=self.min_component_size
        )

        return (cleaned > 0).astype(np.uint8)

    def detect_surface(
        self,
        volume: np.ndarray,
        umbilicus_center: Optional[Tuple[int, int, int]] = None,
        smooth_volume: bool = True,
        sigma: float = 1.0
    ) -> Tuple[np.ndarray, dict]:
        """
        Detect surface in 3D volume.

        Args:
            volume: Input 3D volume (D, H, W)
            umbilicus_center: Optional scroll center for direction filtering
            smooth_volume: Whether to smooth volume before gradient computation
            sigma: Gaussian smoothing sigma

        Returns:
            Tuple of (binary_surface_mask, debug_info)
        """
        print("Starting Sobel surface detection...")

        # Step 1: Optional smoothing
        if smooth_volume:
            print("Smoothing volume...")
            volume = gaussian_filter(volume, sigma=sigma)

        # Step 2: Compute gradients
        print("Computing 3D gradients...")
        grad_z, grad_y, grad_x = self.compute_gradients_3d(volume)

        # Step 3: Gradient magnitude or umbilicus filtering
        if umbilicus_center is not None:
            print("Applying umbilicus direction filter...")
            gradient_magnitude = self.apply_umbilicus_filter(
                grad_z, grad_y, grad_x, umbilicus_center
            )
        else:
            print("Computing gradient magnitude...")
            gradient_magnitude = self.compute_gradient_magnitude(
                grad_z, grad_y, grad_x
            )

        # Step 4: Second derivative (optional)
        if self.use_second_derivative:
            print("Computing second derivatives...")
            grad2_z = ndimage.sobel(grad_z, axis=0)
            grad2_y = ndimage.sobel(grad_y, axis=1)
            grad2_x = ndimage.sobel(grad_x, axis=2)

            second_derivative_magnitude = self.compute_gradient_magnitude(
                grad2_z, grad2_y, grad2_x
            )

            # Combine first and second derivatives
            gradient_magnitude = gradient_magnitude + 0.5 * second_derivative_magnitude

        # Step 5: Thresholding
        print("Thresholding gradients...")
        binary_mask = self.threshold_gradients(gradient_magnitude)

        # Step 6: Morphological operations
        print("Applying morphological operations...")
        binary_mask = self.apply_morphology(binary_mask)

        # Step 7: Remove small components
        print("Removing small components...")
        binary_mask = self.remove_small_components(binary_mask)

        # Debug info
        debug_info = {
            'grad_z': grad_z,
            'grad_y': grad_y,
            'grad_x': grad_x,
            'gradient_magnitude': gradient_magnitude,
            'surface_coverage': binary_mask.sum() / binary_mask.size
        }

        print(f"Surface detection complete. Coverage: {debug_info['surface_coverage']*100:.2f}%")

        return binary_mask, debug_info

    def extract_point_cloud(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Extract point cloud from binary mask.

        Args:
            binary_mask: Binary surface mask

        Returns:
            Point cloud array of shape (N, 3) with (z, y, x) coordinates
        """
        # Get coordinates of surface voxels
        points = np.argwhere(binary_mask > 0)

        return points


def auto_threshold_otsu(volume: np.ndarray) -> float:
    """
    Automatically determine threshold using Otsu's method.

    Args:
        volume: Input volume

    Returns:
        Optimal threshold value
    """
    # Compute gradients
    grad_z = ndimage.sobel(volume, axis=0)
    grad_y = ndimage.sobel(volume, axis=1)
    grad_x = ndimage.sobel(volume, axis=2)

    gradient_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)

    # Normalize
    grad_normalized = (gradient_magnitude - gradient_magnitude.min()) / \
                     (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)

    # Otsu threshold
    threshold = filters.threshold_otsu(grad_normalized)

    return threshold


def visualize_surface_detection(
    volume: np.ndarray,
    binary_mask: np.ndarray,
    debug_info: dict,
    slice_idx: int = 32
):
    """
    Visualize surface detection results.

    Args:
        volume: Input volume
        binary_mask: Detected surface mask
        debug_info: Debug information dictionary
        slice_idx: Slice index to visualize
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original volume
    axes[0, 0].imshow(volume[slice_idx], cmap='gray')
    axes[0, 0].set_title(f'Original Volume - Slice {slice_idx}')
    axes[0, 0].axis('off')

    # Gradient magnitude
    axes[0, 1].imshow(debug_info['gradient_magnitude'][slice_idx], cmap='hot')
    axes[0, 1].set_title('Gradient Magnitude')
    axes[0, 1].axis('off')

    # Binary mask
    axes[0, 2].imshow(binary_mask[slice_idx], cmap='gray')
    axes[0, 2].set_title('Detected Surface')
    axes[0, 2].axis('off')

    # Max projections
    axes[1, 0].imshow(volume.max(axis=0), cmap='gray')
    axes[1, 0].set_title('Volume - Max Projection (Z)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(debug_info['gradient_magnitude'].max(axis=0), cmap='hot')
    axes[1, 1].set_title('Gradient - Max Projection (Z)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(binary_mask.max(axis=0), cmap='gray')
    axes[1, 2].set_title('Surface - Max Projection (Z)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Sobel Surface Detection Example")
    print("="*50)

    # Create synthetic test volume with a simple surface
    D, H, W = 65, 256, 256

    # Create a synthetic cylindrical surface
    print("Creating synthetic test volume...")
    volume = np.zeros((D, H, W))

    center_y, center_x = H // 2, W // 2
    radius = 80

    for z in range(D):
        for y in range(H):
            for x in range(W):
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)

                # Create shell with some thickness
                if abs(dist - radius) < 5:
                    volume[z, y, x] = 1.0 + 0.1 * np.random.randn()

    # Add noise
    volume += 0.05 * np.random.randn(D, H, W)
    volume = np.clip(volume, 0, 1)

    print(f"Test volume shape: {volume.shape}")
    print(f"Value range: [{volume.min():.3f}, {volume.max():.3f}]")

    # Initialize detector
    detector = SobelSurfaceDetector(
        gradient_threshold=0.1,
        use_second_derivative=True,
        morphology_radius=2,
        min_component_size=100
    )

    # Detect surface
    binary_mask, debug_info = detector.detect_surface(
        volume,
        umbilicus_center=(D//2, center_y, center_x),
        smooth_volume=True,
        sigma=1.0
    )

    print(f"\nResults:")
    print(f"Detected surface voxels: {binary_mask.sum()}")
    print(f"Surface coverage: {debug_info['surface_coverage']*100:.2f}%")

    # Extract point cloud
    points = detector.extract_point_cloud(binary_mask)
    print(f"Point cloud size: {points.shape}")

    # Auto threshold example
    print("\nAuto-thresholding with Otsu's method:")
    auto_threshold = auto_threshold_otsu(volume)
    print(f"Otsu threshold: {auto_threshold:.3f}")

    print("\n" + "="*50)
    print("Baseline Sobel detector ready!")
    print("Note: This is a fast baseline. Deep learning models will perform better.")
