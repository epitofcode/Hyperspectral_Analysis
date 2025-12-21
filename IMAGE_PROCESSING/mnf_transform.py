"""
Minimum Noise Fraction (MNF) Transform - Hyperspectral Preprocessing

MNF is similar to PCA but designed specifically for hyperspectral images.
It separates noise from signal by ordering components by SNR instead of variance.

MNF is a two-step process:
1. Decorrelate and rescale noise using noise covariance
2. Apply PCA to noise-whitened data
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).parent.parent / "code"))
from image_utils import load_hyperspectral_mat


def estimate_noise_covariance(image, method='diff'):
    """
    Estimate noise covariance matrix.

    Args:
        image: Hyperspectral image (H, W, B)
        method: 'diff' for difference method, 'homogeneous' for homogeneous region

    Returns:
        Noise covariance matrix (B, B)
    """
    height, width, bands = image.shape

    if method == 'diff':
        # Difference method: assumes noise is high-frequency
        # Take differences along spatial dimensions
        diff_h = image[1:, :, :] - image[:-1, :, :]  # Vertical differences
        diff_w = image[:, 1:, :] - image[:, :-1, :]  # Horizontal differences

        # Flatten and stack
        diff_h_flat = diff_h.reshape(-1, bands)
        diff_w_flat = diff_w.reshape(-1, bands)
        diffs = np.vstack([diff_h_flat, diff_w_flat])

        # Covariance of differences estimates noise
        noise_cov = np.cov(diffs, rowvar=False) / 2.0

    elif method == 'homogeneous':
        # Use a homogeneous region (e.g., center crop)
        center_size = min(20, height//4, width//4)
        h_start = (height - center_size) // 2
        w_start = (width - center_size) // 2

        region = image[h_start:h_start+center_size, w_start:w_start+center_size, :]
        region_flat = region.reshape(-1, bands)

        # Assume variation in homogeneous region is mostly noise
        noise_cov = np.cov(region_flat, rowvar=False)

    else:
        raise ValueError(f"Unknown noise estimation method: {method}")

    return noise_cov


def apply_mnf(image, n_components=50, noise_method='diff'):
    """
    Apply Minimum Noise Fraction transform.

    Args:
        image: Hyperspectral image (H, W, B)
        n_components: Number of components to keep
        noise_method: Method for noise estimation

    Returns:
        MNF-transformed image, transformation matrix
    """
    height, width, bands = image.shape

    print(f"Applying MNF transform:")
    print(f"  Components: {n_components}")
    print(f"  Noise estimation: {noise_method}")

    # Reshape to (pixels, bands)
    image_2d = image.reshape(-1, bands)
    image_2d_centered = image_2d - np.mean(image_2d, axis=0)

    # Step 1: Estimate noise covariance
    print("  Estimating noise covariance...")
    noise_cov = estimate_noise_covariance(image, method=noise_method)

    # Step 2: Eigendecomposition of noise covariance
    noise_eigenvalues, noise_eigenvectors = np.linalg.eigh(noise_cov)

    # Ensure positive eigenvalues
    noise_eigenvalues = np.maximum(noise_eigenvalues, 1e-10)

    # Step 3: Noise whitening transformation
    # D^(-1/2) * V^T
    noise_whitening = noise_eigenvectors @ np.diag(1.0 / np.sqrt(noise_eigenvalues)) @ noise_eigenvectors.T

    # Apply noise whitening
    whitened_data = image_2d_centered @ noise_whitening

    # Step 4: PCA on whitened data
    print("  Applying PCA to noise-whitened data...")
    pca = PCA(n_components=n_components)
    mnf_data = pca.fit_transform(whitened_data)

    # Reshape back
    mnf_image = mnf_data.reshape(height, width, n_components)

    # Calculate signal-to-noise ratio for each component
    snr = pca.explained_variance_ / (1.0 / noise_eigenvalues[-n_components:])

    print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"  Mean SNR of components: {np.mean(snr):.2f}")

    return mnf_image, pca, snr


def compare_pca_vs_mnf(image, n_components=50):
    """
    Compare PCA and MNF side by side.

    Args:
        image: Hyperspectral image
        n_components: Number of components

    Returns:
        PCA result, MNF result, statistics
    """
    height, width, bands = image.shape
    image_2d = image.reshape(-1, bands)

    # Standard PCA
    print("\n" + "-"*80)
    print("STANDARD PCA")
    print("-"*80)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(image_2d)
    pca_image = pca_data.reshape(height, width, n_components)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # MNF
    print("\n" + "-"*80)
    print("MINIMUM NOISE FRACTION (MNF)")
    print("-"*80)
    mnf_image, mnf_model, snr = apply_mnf(image, n_components=n_components)

    return pca_image, mnf_image, {
        'pca_variance': pca.explained_variance_ratio_.sum(),
        'mnf_variance': mnf_model.explained_variance_ratio_.sum(),
        'mnf_snr': snr
    }


def main():
    """Test MNF transform on Indian Pines."""
    print("="*80)
    print("MINIMUM NOISE FRACTION (MNF) TRANSFORM")
    print("="*80)

    # Load data
    IMAGE_PATH = "../data/indian_pines/indian_pines_image.mat"
    image = load_hyperspectral_mat(IMAGE_PATH)

    print(f"\nOriginal image shape: {image.shape}")

    # Compare PCA vs MNF
    pca_result, mnf_result, stats = compare_pca_vs_mnf(image, n_components=50)

    print("\n" + "="*80)
    print("COMPARISON: PCA vs MNF")
    print("="*80)
    print(f"PCA shape: {pca_result.shape}")
    print(f"MNF shape: {mnf_result.shape}")
    print(f"\nPCA variance retained: {stats['pca_variance']*100:.2f}%")
    print(f"MNF variance retained: {stats['mnf_variance']*100:.2f}%")
    print(f"MNF mean SNR: {np.mean(stats['mnf_snr']):.2f}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("MNF vs PCA:")
    print("  - MNF: Orders components by signal-to-noise ratio")
    print("  - PCA: Orders components by variance")
    print("  - MNF: Better for noisy hyperspectral data")
    print("  - PCA: Simpler, computationally faster")
    print("\nExpected accuracy gain: +1-2% typically")
    print("But increased computational complexity")


if __name__ == "__main__":
    main()
