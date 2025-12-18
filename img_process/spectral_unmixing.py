"""
Spectral Unmixing - Hyperspectral Preprocessing (Simplified)

Decomposes mixed pixels into pure endmember spectra and their abundances.

Methods:
1. Vertex Component Analysis (VCA) - endmember extraction
2. Non-negative Least Squares (NNLS) - abundance estimation
3. Linear Spectral Unmixing
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import nnls

sys.path.append(str(Path(__file__).parent.parent / "code"))
from image_utils import load_hyperspectral_mat, load_ground_truth


def vertex_component_analysis(image, n_endmembers=5, snr_input=15):
    """
    Vertex Component Analysis (VCA) for endmember extraction.

    Simplified implementation based on:
    Nascimento & Dias, "Vertex Component Analysis: A Fast Algorithm to Unmix
    Hyperspectral Data", IEEE TGRS, 2005

    Args:
        image: Hyperspectral image (H, W, B)
        n_endmembers: Number of endmembers to extract
        snr_input: SNR estimate (higher = less noise reduction)

    Returns:
        Endmember spectra (n_endmembers, B), indices
    """
    height, width, bands = image.shape
    n_pixels = height * width

    # Reshape
    image_2d = image.reshape(n_pixels, bands).T  # (bands, pixels)

    print(f"Vertex Component Analysis:")
    print(f"  Extracting {n_endmembers} endmembers")
    print(f"  Image size: {height}x{width}")
    print(f"  Spectral bands: {bands}")

    # Mean centering
    mean_spectrum = np.mean(image_2d, axis=1, keepdims=True)
    image_centered = image_2d - mean_spectrum

    # SVD for dimensionality reduction (noise reduction step)
    U, S, Vt = np.linalg.svd(image_centered, full_matrices=False)

    # Project to endmember subspace
    k = min(n_endmembers, bands - 1)
    projected = U[:, :k].T @ image_2d  # (k, n_pixels)

    # Initialize with first endmember (maximum projection)
    endmember_indices = []
    remaining = np.ones(n_pixels, dtype=bool)

    # Find maximum projection
    max_idx = np.argmax(np.sum(projected**2, axis=0))
    endmember_indices.append(max_idx)

    # Iteratively find remaining endmembers
    for i in range(1, n_endmembers):
        # Project onto orthogonal subspace of current endmembers
        endmembers_so_far = image_2d[:, endmember_indices]

        # Orthogonal projection
        for j in range(n_pixels):
            if remaining[j]:
                pixel = image_2d[:, j:j+1]
                # Project out existing endmembers
                for endmember in endmembers_so_far.T:
                    projection = np.dot(pixel.T, endmember) / np.dot(endmember, endmember)
                    pixel = pixel - projection * endmember[:, np.newaxis]
                projected[:, j] = (U[:, :k].T @ pixel).flatten()

        # Find maximum remaining projection
        projected_norm = np.sum(projected**2, axis=0)
        projected_norm[~remaining] = -np.inf
        max_idx = np.argmax(projected_norm)

        endmember_indices.append(max_idx)
        remaining[max_idx] = False

    # Extract endmember spectra
    endmembers = image_2d[:, endmember_indices].T  # (n_endmembers, bands)

    print(f"  Extracted {len(endmember_indices)} endmember spectra")

    return endmembers, endmember_indices


def unmix_pixel(pixel_spectrum, endmembers):
    """
    Unmix a single pixel using non-negative least squares.

    Args:
        pixel_spectrum: Spectrum of pixel (B,)
        endmembers: Endmember matrix (n_endmembers, B)

    Returns:
        Abundance vector (n_endmembers,)
    """
    # Solve: pixel = endmembers^T * abundances
    # Subject to: abundances >= 0, sum(abundances) = 1

    abundances, _ = nnls(endmembers.T, pixel_spectrum)

    # Normalize to sum to 1
    if np.sum(abundances) > 0:
        abundances = abundances / np.sum(abundances)

    return abundances


def linear_spectral_unmixing(image, endmembers):
    """
    Perform linear spectral unmixing on entire image.

    Args:
        image: Hyperspectral image (H, W, B)
        endmembers: Endmember spectra (n_endmembers, B)

    Returns:
        Abundance maps (H, W, n_endmembers)
    """
    height, width, bands = image.shape
    n_endmembers = endmembers.shape[0]

    print(f"Performing linear spectral unmixing:")
    print(f"  Endmembers: {n_endmembers}")
    print(f"  Processing {height*width} pixels...")

    # Reshape image
    image_2d = image.reshape(-1, bands)

    # Initialize abundance map
    abundances = np.zeros((height * width, n_endmembers))

    # Unmix each pixel
    for i in range(height * width):
        abundances[i, :] = unmix_pixel(image_2d[i, :], endmembers)

        if (i+1) % 10000 == 0:
            print(f"    Processed {i+1}/{height*width} pixels...")

    # Reshape to image
    abundance_maps = abundances.reshape(height, width, n_endmembers)

    print(f"  Unmixing complete!")

    return abundance_maps


def visualize_endmembers(endmembers, save_path=None):
    """
    Visualize extracted endmember spectra.

    Args:
        endmembers: Endmember matrix (n_endmembers, B)
        save_path: Path to save figure
    """
    n_endmembers, bands = endmembers.shape
    band_indices = np.arange(bands)

    plt.figure(figsize=(12, 6))

    for i in range(n_endmembers):
        plt.plot(band_indices, endmembers[i, :], linewidth=2, label=f'Endmember {i+1}')

    plt.xlabel('Band Index')
    plt.ylabel('Reflectance')
    plt.title(f'Extracted Endmember Spectra ({n_endmembers} endmembers)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved endmember plot: {save_path}")

    plt.show()


def main():
    """Test spectral unmixing on Indian Pines."""
    print("="*80)
    print("SPECTRAL UNMIXING - HYPERSPECTRAL PREPROCESSING")
    print("="*80)

    # Load data
    IMAGE_PATH = "../data/indian_pines/indian_pines_image.mat"
    GT_PATH = "../data/indian_pines/indian_pines_gt.mat"

    image = load_hyperspectral_mat(IMAGE_PATH)
    gt = load_ground_truth(GT_PATH)

    print(f"\nOriginal image shape: {image.shape}")

    # Extract endmembers
    print("\n" + "-"*80)
    print("STEP 1: ENDMEMBER EXTRACTION (VCA)")
    print("-"*80)
    n_endmembers = 6  # Try 6 endmembers for 16 classes
    endmembers, endmember_indices = vertex_component_analysis(image, n_endmembers=n_endmembers)

    # Visualize endmembers
    visualize_endmembers(endmembers, save_path="extracted_endmembers.png")

    # Unmix image (on a small subset for speed)
    print("\n" + "-"*80)
    print("STEP 2: LINEAR SPECTRAL UNMIXING (Small subset)")
    print("-"*80)

    # Use small region for demonstration (full image takes very long)
    subset_size = 50
    image_subset = image[:subset_size, :subset_size, :]

    abundance_maps = linear_spectral_unmixing(image_subset, endmembers)

    # Visualize abundance maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Abundance Maps (Subset)', fontsize=16, fontweight='bold')

    for i in range(min(6, n_endmembers)):
        ax = axes[i//3, i%3]
        im = ax.imshow(abundance_maps[:, :, i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Endmember {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("abundance_maps.png", dpi=150, bbox_inches='tight')
    print(f"Saved abundance maps: abundance_maps.png")
    plt.show()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Spectral Unmixing:")
    print("  - Decomposes mixed pixels into pure materials (endmembers)")
    print("  - Each pixel = weighted combination of endmembers")
    print("  - Useful for sub-pixel classification")
    print("\nLimitations:")
    print("  - Assumes linear mixing")
    print("  - Computationally expensive (NNLS for each pixel)")
    print("  - Number of endmembers must be specified")
    print("\nFor classification:")
    print("  - Can use abundance maps as features")
    print("  - Expected gain: +1-2% for mixed-pixel scenarios")
    print("  - Better for coarse spatial resolution images")


if __name__ == "__main__":
    main()
