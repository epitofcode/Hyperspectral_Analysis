"""
Image Utilities for Hyperspectral Data
Provides functions for loading, preprocessing, and visualizing hyperspectral images

Enhanced with Spectral Python (SPy) integration for advanced HSI visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.colors import ListedColormap
import cv2
from typing import Tuple, Optional, List
import warnings

# Spectral Python (SPy) - Optional import
try:
    import spectral as spy
    import spectral.io.envi as envi
    SPY_AVAILABLE = True
except ImportError:
    SPY_AVAILABLE = False
    warnings.warn("Spectral Python (SPy) not available. Install with: pip install spectral")



def load_hyperspectral_mat(filepath: str) -> np.ndarray:
    """
    Load hyperspectral image from .mat file

    Args:
        filepath: Path to the .mat file

    Returns:
        Hyperspectral image array with shape (height, width, bands)
    """
    data = sio.loadmat(filepath)

    # Find the data key (skip metadata keys starting with '__')
    data_key = [key for key in data.keys() if not key.startswith('__')][0]

    image = data[data_key]
    print(f"Loaded hyperspectral image: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Value range: [{image.min():.4f}, {image.max():.4f}]")

    return image


def load_ground_truth(filepath: str) -> np.ndarray:
    """
    Load ground truth labels from .mat file

    Args:
        filepath: Path to the ground truth .mat file

    Returns:
        Ground truth array with shape (height, width)
    """
    data = sio.loadmat(filepath)

    # Find the data key
    data_key = [key for key in data.keys() if not key.startswith('__')][0]

    gt = data[data_key]
    print(f"Loaded ground truth: {gt.shape}")
    print(f"Number of classes: {len(np.unique(gt))}")
    print(f"Class distribution: {np.bincount(gt.flatten())}")

    return gt


def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize hyperspectral image

    Args:
        image: Input image with shape (height, width, bands)
        method: Normalization method ('minmax', 'zscore', 'l2')

    Returns:
        Normalized image
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = image.min()
        max_val = image.max()
        normalized = (image - min_val) / (max_val - min_val + 1e-8)

    elif method == 'zscore':
        # Z-score normalization (standardization)
        mean = image.mean()
        std = image.std()
        normalized = (image - mean) / (std + 1e-8)

    elif method == 'l2':
        # L2 normalization per pixel
        normalized = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = image[i, j, :]
                norm = np.linalg.norm(pixel)
                normalized[i, j, :] = pixel / (norm + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def select_rgb_bands(image: np.ndarray,
                     red_band: Optional[int] = None,
                     green_band: Optional[int] = None,
                     blue_band: Optional[int] = None) -> np.ndarray:
    """
    Select RGB bands from hyperspectral image for visualization

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        red_band: Band index for red channel (default: band near 650nm)
        green_band: Band index for green channel (default: band near 550nm)
        blue_band: Band index for blue channel (default: band near 450nm)

    Returns:
        RGB image with shape (height, width, 3)
    """
    total_bands = image.shape[2]

    # Default band selection (approximate wavelengths)
    if red_band is None:
        red_band = int(total_bands * 0.6)  # ~60% through spectrum
    if green_band is None:
        green_band = int(total_bands * 0.4)  # ~40% through spectrum
    if blue_band is None:
        blue_band = int(total_bands * 0.2)  # ~20% through spectrum

    # Extract RGB bands
    rgb = np.stack([
        image[:, :, red_band],
        image[:, :, green_band],
        image[:, :, blue_band]
    ], axis=2)

    # Normalize to [0, 1]
    rgb = normalize_image(rgb, method='minmax')

    return rgb


def visualize_rgb(image: np.ndarray,
                  title: str = "RGB Composite",
                  figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize hyperspectral image as RGB composite

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        title: Plot title
        figsize: Figure size
    """
    rgb = select_rgb_bands(image)

    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def visualize_bands(image: np.ndarray,
                    band_indices: list = None,
                    figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Visualize multiple spectral bands

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        band_indices: List of band indices to visualize (default: evenly spaced)
        figsize: Figure size
    """
    if band_indices is None:
        # Select evenly spaced bands
        total_bands = image.shape[2]
        band_indices = np.linspace(0, total_bands - 1, 9, dtype=int)

    n_bands = len(band_indices)
    cols = 3
    rows = (n_bands + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_bands > 1 else [axes]

    for idx, band_idx in enumerate(band_indices):
        ax = axes[idx]
        band = image[:, :, band_idx]

        im = ax.imshow(band, cmap='viridis')
        ax.set_title(f"Band {band_idx}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_ground_truth(gt: np.ndarray,
                          title: str = "Ground Truth",
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize ground truth labels

    Args:
        gt: Ground truth array with shape (height, width)
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Create colormap
    n_classes = len(np.unique(gt))
    cmap = plt.cm.get_cmap('tab20', n_classes)

    plt.imshow(gt, cmap=cmap)
    plt.title(title)
    plt.colorbar(label='Class Label', ticks=range(n_classes))
    plt.tight_layout()
    plt.show()


def visualize_spectral_signature(image: np.ndarray,
                                 coordinates: list,
                                 labels: list = None,
                                 figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot spectral signatures for specific pixel locations

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        coordinates: List of (row, col) tuples
        labels: List of labels for each coordinate
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    bands = np.arange(image.shape[2])

    for idx, (row, col) in enumerate(coordinates):
        signature = image[row, col, :]
        label = labels[idx] if labels else f"Pixel ({row}, {col})"
        plt.plot(bands, signature, label=label, linewidth=2)

    plt.xlabel('Band Number')
    plt.ylabel('Reflectance')
    plt.title('Spectral Signatures')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_statistics(image: np.ndarray) -> dict:
    """
    Compute basic statistics for hyperspectral image

    Args:
        image: Hyperspectral image with shape (height, width, bands)

    Returns:
        Dictionary containing statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': image.dtype,
        'min': image.min(),
        'max': image.max(),
        'mean': image.mean(),
        'std': image.std(),
        'median': np.median(image),
        'band_means': image.mean(axis=(0, 1)),
        'band_stds': image.std(axis=(0, 1))
    }

    return stats


def print_statistics(stats: dict) -> None:
    """
    Print image statistics

    Args:
        stats: Dictionary containing statistics from compute_statistics()
    """
    print("=" * 60)
    print("Hyperspectral Image Statistics")
    print("=" * 60)
    print(f"Shape: {stats['shape']}")
    print(f"Data type: {stats['dtype']}")
    print(f"Value range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std: {stats['std']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print("=" * 60)


def view_hypercube_spy(image: np.ndarray, title: str = "Hyperspectral Cube") -> None:
    """
    Visualize hyperspectral cube using Spectral Python's interactive viewer.

    Requires SPy to be installed. Provides interactive N-D visualization
    with spectral profiles and image browsing.

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        title: Window title
    """
    if not SPY_AVAILABLE:
        print("Spectral Python (SPy) is not available. Using fallback visualization.")
        visualize_rgb(image, title=title)
        return

    # Configure SPy settings for better visualization
    spy.settings.envi_support_nonlowercase_params = True

    # Display the cube
    view = spy.imshow(image, title=title)
    return view


def view_rgb_spy(image: np.ndarray,
                 red_band: Optional[int] = None,
                 green_band: Optional[int] = None,
                 blue_band: Optional[int] = None,
                 title: str = "RGB Composite") -> None:
    """
    Create RGB composite using Spectral Python.

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        red_band: Band index for red channel
        green_band: Band index for green channel
        blue_band: Band index for blue channel
        title: Window title
    """
    if not SPY_AVAILABLE:
        print("Spectral Python (SPy) is not available. Using fallback visualization.")
        visualize_rgb(image, title=title)
        return

    total_bands = image.shape[2]

    # Default band selection
    if red_band is None:
        red_band = int(total_bands * 0.6)
    if green_band is None:
        green_band = int(total_bands * 0.4)
    if blue_band is None:
        blue_band = int(total_bands * 0.2)

    # Create RGB view
    bands = [red_band, green_band, blue_band]
    view = spy.imshow(image, bands=bands, title=title)
    return view


def get_spectral_library_spy(image: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Extract mean spectral signatures for each class using SPy.

    Creates a spectral library where each class has a representative signature
    computed as the mean of all pixels belonging to that class.

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        ground_truth: Ground truth labels (height, width)

    Returns:
        Dictionary mapping class IDs to mean spectral signatures
    """
    classes = np.unique(ground_truth)
    classes = classes[classes > 0]  # Exclude background

    spectral_library = {}

    for class_id in classes:
        class_mask = ground_truth == class_id
        class_pixels = image[class_mask]

        mean_signature = np.mean(class_pixels, axis=0)
        spectral_library[int(class_id)] = mean_signature

    print(f"Created spectral library with {len(spectral_library)} classes")

    return spectral_library


def plot_spectral_library(spectral_library: dict,
                         class_names: Optional[dict] = None,
                         wavelengths: Optional[np.ndarray] = None,
                         figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot spectral signatures from a spectral library.

    Args:
        spectral_library: Dictionary mapping class IDs to spectral signatures
        class_names: Optional dictionary mapping class IDs to names
        wavelengths: Optional wavelength array for x-axis
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    if wavelengths is None:
        # Use band numbers if wavelengths not provided
        n_bands = len(list(spectral_library.values())[0])
        wavelengths = np.arange(n_bands)
        xlabel = 'Band Number'
    else:
        xlabel = 'Wavelength (nm)'

    for class_id, signature in spectral_library.items():
        if class_names and class_id in class_names:
            label = f"Class {class_id}: {class_names[class_id]}"
        else:
            label = f"Class {class_id}"

        plt.plot(wavelengths, signature, label=label, linewidth=2, alpha=0.8)

    plt.xlabel(xlabel)
    plt.ylabel('Reflectance')
    plt.title('Spectral Library - Mean Signatures per Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_as_envi(image: np.ndarray, filepath: str,
                 wavelengths: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None) -> None:
    """
    Save hyperspectral image in ENVI format using SPy.

    ENVI format is widely supported by remote sensing software.

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        filepath: Output filepath (without extension)
        wavelengths: Optional wavelength information
        metadata: Optional metadata dictionary
    """
    if not SPY_AVAILABLE:
        print("Spectral Python (SPy) is not available. Cannot save as ENVI format.")
        print("Saving as NumPy array instead.")
        np.save(f"{filepath}.npy", image)
        return

    # Prepare metadata
    if metadata is None:
        metadata = {}

    if wavelengths is not None:
        metadata['wavelength'] = wavelengths

    # Save in ENVI format
    envi.save_image(f"{filepath}.hdr", image, metadata=metadata, force=True)
    print(f"Saved hyperspectral image in ENVI format: {filepath}.hdr")


def load_from_envi(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load hyperspectral image from ENVI format using SPy.

    Args:
        filepath: Path to ENVI header file (.hdr)

    Returns:
        Tuple of (image_array, metadata)
    """
    if not SPY_AVAILABLE:
        raise ImportError("Spectral Python (SPy) is required to load ENVI files. "
                         "Install with: pip install spectral")

    # Load ENVI file
    img = envi.open(filepath)

    # Get image data
    image_array = img.load()

    # Get metadata
    metadata = img.metadata

    print(f"Loaded ENVI image: {image_array.shape}")
    if 'wavelength' in metadata:
        print(f"Wavelength range: {metadata['wavelength'][0]} - {metadata['wavelength'][-1]} nm")

    return image_array, metadata


def create_classification_overlay(image: np.ndarray,
                                  classification_map: np.ndarray,
                                  alpha: float = 0.5,
                                  figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Create overlay visualization of classification results on RGB image.

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        classification_map: Classification results (height, width)
        alpha: Transparency of overlay (0-1)
        figsize: Figure size
    """
    # Create RGB composite
    rgb = select_rgb_bands(image)

    # Create classification colormap
    n_classes = len(np.unique(classification_map))
    cmap = plt.cm.get_cmap('tab20', n_classes)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # RGB image
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Composite')
    axes[0].axis('off')

    # Classification overlay
    axes[1].imshow(rgb)
    # Mask background (class 0)
    classification_masked = np.ma.masked_where(classification_map == 0, classification_map)
    im = axes[1].imshow(classification_masked, cmap=cmap, alpha=alpha)
    axes[1].set_title('Classification Overlay')
    axes[1].axis('off')

    # Colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Class Label')

    plt.tight_layout()
    plt.show()


def visualize_3d_scatter(image: np.ndarray, ground_truth: np.ndarray,
                        n_samples: int = 1000, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Create 3D scatter plot of first 3 principal components colored by class.

    Useful for visualizing class separability in feature space.

    Args:
        image: Hyperspectral image with shape (height, width, bands)
        ground_truth: Ground truth labels (height, width)
        n_samples: Number of samples to plot (for performance)
        figsize: Figure size
    """
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    # Get labeled pixels
    labeled_mask = ground_truth > 0
    labeled_pixels = image[labeled_mask]
    labeled_classes = ground_truth[labeled_mask]

    # Subsample if too many points
    if len(labeled_pixels) > n_samples:
        indices = np.random.choice(len(labeled_pixels), n_samples, replace=False)
        labeled_pixels = labeled_pixels[indices]
        labeled_classes = labeled_classes[indices]

    # Apply PCA to reduce to 3D
    pca = PCA(n_components=3)
    pixels_3d = pca.fit_transform(labeled_pixels)

    # Create 3D scatter plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Color by class
    classes = np.unique(labeled_classes)
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))

    for i, class_id in enumerate(classes):
        class_mask = labeled_classes == class_id
        ax.scatter(pixels_3d[class_mask, 0],
                  pixels_3d[class_mask, 1],
                  pixels_3d[class_mask, 2],
                  c=[colors[i]],
                  label=f'Class {class_id}',
                  alpha=0.6,
                  s=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title('3D PCA Scatter Plot by Class')
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Image utilities module")
    print("Import this module to use image loading and visualization functions")
    print(f"Spectral Python (SPy) available: {SPY_AVAILABLE}")
