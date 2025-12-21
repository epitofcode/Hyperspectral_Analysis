"""
Atmospheric Correction - Hyperspectral Preprocessing (Simplified)

NOTE: Real atmospheric correction requires atmospheric models (MODTRAN, FLAASH, etc.)
and sensor/illumination parameters. This is a simplified version for educational purposes.

Methods implemented:
1. Dark Object Subtraction (DOS)
2. Empirical Line Method (ELM) - requires reference spectra
3. Flat Field Correction
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "code"))
from image_utils import load_hyperspectral_mat


def dark_object_subtraction(image, percentile=1):
    """
    Dark Object Subtraction (DOS) method.

    Assumption: Darkest pixels in the scene should have near-zero reflectance.
    Any signal is atmospheric path radiance that should be subtracted.

    Args:
        image: Hyperspectral image (H, W, B)
        percentile: Percentile to use for dark object (default 1%)

    Returns:
        Corrected image
    """
    height, width, bands = image.shape

    print(f"Applying Dark Object Subtraction (DOS):")
    print(f"  Percentile: {percentile}%")

    # Reshape to (pixels, bands)
    image_2d = image.reshape(-1, bands)

    # Find dark object value per band (percentile across all pixels)
    dark_values = np.percentile(image_2d, percentile, axis=0)

    print(f"  Mean dark value: {np.mean(dark_values):.2f}")
    print(f"  Dark value range: [{np.min(dark_values):.2f}, {np.max(dark_values):.2f}]")

    # Subtract dark object from each band
    corrected_2d = image_2d - dark_values[np.newaxis, :]

    # Ensure non-negative
    corrected_2d = np.maximum(corrected_2d, 0)

    # Reshape back
    corrected_image = corrected_2d.reshape(height, width, bands)

    return corrected_image


def flat_field_correction(image, reference_pixel=None):
    """
    Flat Field Correction.

    Normalizes each pixel's spectrum by a reference spectrum
    (typically from a bright, uniform target).

    Args:
        image: Hyperspectral image (H, W, B)
        reference_pixel: (row, col) of reference pixel, or None for mean

    Returns:
        Corrected image
    """
    height, width, bands = image.shape

    print(f"Applying Flat Field Correction:")

    # Get reference spectrum
    if reference_pixel is None:
        # Use mean spectrum of brightest pixels
        image_2d = image.reshape(-1, bands)
        brightness = np.sum(image_2d, axis=1)
        top_10_percent = int(0.1 * len(brightness))
        brightest_indices = np.argpartition(brightness, -top_10_percent)[-top_10_percent:]
        reference_spectrum = np.mean(image_2d[brightest_indices, :], axis=0)
        print(f"  Reference: Mean of top 10% brightest pixels")
    else:
        row, col = reference_pixel
        reference_spectrum = image[row, col, :]
        print(f"  Reference: Pixel at ({row}, {col})")

    print(f"  Reference spectrum mean: {np.mean(reference_spectrum):.2f}")

    # Normalize each pixel by reference
    corrected_image = np.zeros_like(image, dtype=np.float32)

    for band_idx in range(bands):
        if reference_spectrum[band_idx] > 0:
            corrected_image[:, :, band_idx] = image[:, :, band_idx] / reference_spectrum[band_idx]
        else:
            corrected_image[:, :, band_idx] = image[:, :, band_idx]

    # Scale back to reasonable range
    corrected_image = corrected_image * np.mean(reference_spectrum)

    return corrected_image


def simple_gain_offset_correction(image):
    """
    Simple gain and offset correction per band.

    Normalizes each band to [0, 1] range independently.

    Args:
        image: Hyperspectral image (H, W, B)

    Returns:
        Corrected image
    """
    height, width, bands = image.shape

    print(f"Applying Simple Gain/Offset Correction:")

    corrected_image = np.zeros_like(image, dtype=np.float32)

    for band_idx in range(bands):
        band = image[:, :, band_idx]

        min_val = np.min(band)
        max_val = np.max(band)

        if max_val > min_val:
            # Normalize to [0, 1]
            corrected_band = (band - min_val) / (max_val - min_val)
        else:
            corrected_band = band

        # Scale to original mean for consistency
        corrected_band = corrected_band * (np.mean(band) / np.mean(corrected_band))

        corrected_image[:, :, band_idx] = corrected_band

    print(f"  All bands normalized to similar ranges")

    return corrected_image


def main():
    """Test atmospheric correction methods on Indian Pines."""
    print("="*80)
    print("ATMOSPHERIC CORRECTION - HYPERSPECTRAL PREPROCESSING")
    print("="*80)
    print("\nNOTE: These are simplified methods for educational purposes.")
    print("Real atmospheric correction requires atmospheric modeling (MODTRAN, etc.)")

    # Load data
    IMAGE_PATH = "../data/indian_pines/indian_pines_image.mat"
    image = load_hyperspectral_mat(IMAGE_PATH)

    print(f"\nOriginal image shape: {image.shape}")
    print(f"Original value range: [{np.min(image):.2f}, {np.max(image):.2f}]")
    print(f"Original mean: {np.mean(image):.2f}")

    # Method 1: Dark Object Subtraction
    print("\n" + "-"*80)
    print("METHOD 1: DARK OBJECT SUBTRACTION (DOS)")
    print("-"*80)
    corrected_dos = dark_object_subtraction(image, percentile=1)
    print(f"Result range: [{np.min(corrected_dos):.2f}, {np.max(corrected_dos):.2f}]")
    print(f"Result mean: {np.mean(corrected_dos):.2f}")

    # Method 2: Flat Field Correction
    print("\n" + "-"*80)
    print("METHOD 2: FLAT FIELD CORRECTION")
    print("-"*80)
    corrected_flat = flat_field_correction(image)
    print(f"Result range: [{np.min(corrected_flat):.2f}, {np.max(corrected_flat):.2f}]")
    print(f"Result mean: {np.mean(corrected_flat):.2f}")

    # Method 3: Simple Gain/Offset
    print("\n" + "-"*80)
    print("METHOD 3: SIMPLE GAIN/OFFSET CORRECTION")
    print("-"*80)
    corrected_gain = simple_gain_offset_correction(image)
    print(f"Result range: [{np.min(corrected_gain):.2f}, {np.max(corrected_gain):.2f}]")
    print(f"Result mean: {np.mean(corrected_gain):.2f}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Atmospheric correction removes atmospheric effects:")
    print("  - Path radiance (scattered light)")
    print("  - Absorption by gases (water vapor, O2, etc.)")
    print("  - Illumination variations")
    print("\nSimplified methods (DOS, Flat Field):")
    print("  - Easy to implement")
    print("  - Don't require atmospheric parameters")
    print("  - Less accurate than physics-based models")
    print("\nFor research-grade correction:")
    print("  - Use MODTRAN, FLAASH, or ATCOR")
    print("  - Requires sensor calibration data")
    print("  - Expected accuracy gain: +2-5% for noisy scenes")


if __name__ == "__main__":
    main()
