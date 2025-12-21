"""
Spectral Smoothing - Hyperspectral Preprocessing

Applies Savitzky-Golay filter to smooth spectral curves and reduce noise.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

sys.path.append(str(Path(__file__).parent.parent / "code"))
from image_utils import load_hyperspectral_mat, load_ground_truth


def apply_spectral_smoothing(image, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth spectral signatures.

    Args:
        image: Hyperspectral image (H, W, B)
        window_length: Length of filter window (must be odd)
        polyorder: Polynomial order for fitting

    Returns:
        Smoothed image
    """
    height, width, bands = image.shape

    # Ensure window_length is odd and valid
    if window_length % 2 == 0:
        window_length += 1
    if window_length > bands:
        window_length = bands if bands % 2 == 1 else bands - 1

    print(f"Applying Savitzky-Golay filter:")
    print(f"  Window length: {window_length}")
    print(f"  Polynomial order: {polyorder}")

    # Reshape to (pixels, bands)
    image_2d = image.reshape(-1, bands)

    # Apply filter to each pixel's spectrum
    smoothed_2d = np.zeros_like(image_2d)

    for i in range(image_2d.shape[0]):
        spectrum = image_2d[i, :]
        smoothed_spectrum = savgol_filter(spectrum, window_length, polyorder)
        smoothed_2d[i, :] = smoothed_spectrum

    # Reshape back
    smoothed_image = smoothed_2d.reshape(height, width, bands)

    return smoothed_image


def visualize_smoothing(original, smoothed, pixel_row, pixel_col, save_path=None):
    """
    Visualize the effect of smoothing on a sample pixel's spectrum.

    Args:
        original: Original image
        smoothed: Smoothed image
        pixel_row: Row index of sample pixel
        pixel_col: Column index of sample pixel
        save_path: Path to save figure
    """
    original_spectrum = original[pixel_row, pixel_col, :]
    smoothed_spectrum = smoothed[pixel_row, pixel_col, :]

    bands = len(original_spectrum)
    band_indices = np.arange(bands)

    plt.figure(figsize=(12, 5))

    # Original vs Smoothed
    plt.subplot(1, 2, 1)
    plt.plot(band_indices, original_spectrum, 'b-', alpha=0.5, label='Original', linewidth=1)
    plt.plot(band_indices, smoothed_spectrum, 'r-', label='Smoothed', linewidth=2)
    plt.xlabel('Band Index')
    plt.ylabel('Reflectance')
    plt.title(f'Spectral Smoothing (Pixel [{pixel_row}, {pixel_col}])')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Difference
    plt.subplot(1, 2, 2)
    difference = original_spectrum - smoothed_spectrum
    plt.plot(band_indices, difference, 'g-', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Band Index')
    plt.ylabel('Difference (Original - Smoothed)')
    plt.title('Noise Removed by Smoothing')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")

    plt.show()


def main():
    """Test spectral smoothing on Indian Pines."""
    print("="*80)
    print("SPECTRAL SMOOTHING - SAVITZKY-GOLAY FILTER")
    print("="*80)

    # Load data
    IMAGE_PATH = "../data/indian_pines/indian_pines_image.mat"
    GT_PATH = "../data/indian_pines/indian_pines_gt.mat"

    image = load_hyperspectral_mat(IMAGE_PATH)
    gt = load_ground_truth(GT_PATH)

    print(f"\nOriginal image shape: {image.shape}")

    # Test different window lengths
    window_lengths = [5, 11, 21]

    for window_length in window_lengths:
        print("\n" + "-"*80)
        print(f"Window Length: {window_length}")
        print("-"*80)

        smoothed = apply_spectral_smoothing(image, window_length=window_length, polyorder=2)

        # Calculate noise reduction
        difference = np.abs(image - smoothed)
        mean_diff = np.mean(difference)
        max_diff = np.max(difference)

        print(f"Mean absolute difference: {mean_diff:.4f}")
        print(f"Max absolute difference: {max_diff:.4f}")

        # Visualize for one sample pixel
        if window_length == 11:  # Only for default
            # Find a labeled pixel from Class 2 (large class)
            class2_pixels = np.argwhere(gt == 2)
            if len(class2_pixels) > 0:
                sample_pixel = class2_pixels[len(class2_pixels)//2]
                pixel_row, pixel_col = sample_pixel

                visualize_smoothing(
                    image, smoothed,
                    pixel_row, pixel_col,
                    save_path=f"spectral_smoothing_w{window_length}.png"
                )

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Savitzky-Golay filter smooths spectral curves while preserving shape")
    print("Larger window = more smoothing, but may remove subtle features")
    print("Recommended: window_length=11, polyorder=2")


if __name__ == "__main__":
    main()
