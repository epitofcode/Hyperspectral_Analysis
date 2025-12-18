"""
Bad Band Removal - Hyperspectral Preprocessing

Removes bands with low SNR or water absorption regions.
Common water absorption bands: 1400nm, 1900nm regions.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "code"))
from image_utils import load_hyperspectral_mat


def identify_bad_bands_snr(image, snr_threshold=10):
    """
    Identify bad bands based on Signal-to-Noise Ratio.

    Args:
        image: Hyperspectral image (H, W, B)
        snr_threshold: Minimum SNR threshold

    Returns:
        List of band indices to keep
    """
    height, width, bands = image.shape

    good_bands = []
    snr_values = []

    for band_idx in range(bands):
        band = image[:, :, band_idx]

        # Calculate SNR: mean / std
        mean_val = np.mean(band)
        std_val = np.std(band)

        if std_val > 0:
            snr = mean_val / std_val
        else:
            snr = 0

        snr_values.append(snr)

        if snr >= snr_threshold:
            good_bands.append(band_idx)

    return good_bands, snr_values


def remove_water_absorption_bands(bands, wavelengths=None):
    """
    Remove water absorption bands if wavelength information is available.

    Water absorption regions:
    - 1350-1450 nm
    - 1800-1950 nm
    - 2400-2500 nm

    Args:
        bands: Total number of bands
        wavelengths: Array of wavelength values (nm)

    Returns:
        List of band indices to keep
    """
    if wavelengths is None:
        # Assume linear spacing from 400nm to 2500nm for Indian Pines
        wavelengths = np.linspace(400, 2500, bands)

    water_regions = [
        (1350, 1450),
        (1800, 1950),
        (2400, 2500)
    ]

    good_bands = []
    for band_idx in range(bands):
        wl = wavelengths[band_idx]

        # Check if wavelength is outside water absorption regions
        in_water_region = False
        for start, end in water_regions:
            if start <= wl <= end:
                in_water_region = True
                break

        if not in_water_region:
            good_bands.append(band_idx)

    return good_bands


def apply_bad_band_removal(image, method='snr', snr_threshold=10):
    """
    Apply bad band removal to hyperspectral image.

    Args:
        image: Hyperspectral image (H, W, B)
        method: 'snr' or 'water' or 'both'
        snr_threshold: SNR threshold for SNR method

    Returns:
        Cleaned image, kept band indices
    """
    height, width, bands = image.shape

    if method == 'snr':
        good_bands, snr_values = identify_bad_bands_snr(image, snr_threshold)
        print(f"SNR-based removal: Kept {len(good_bands)}/{bands} bands")
        print(f"Mean SNR: {np.mean(snr_values):.2f}")

    elif method == 'water':
        good_bands = remove_water_absorption_bands(bands)
        print(f"Water absorption removal: Kept {len(good_bands)}/{bands} bands")

    elif method == 'both':
        # First remove water bands
        water_bands = remove_water_absorption_bands(bands)

        # Then check SNR on remaining bands
        temp_image = image[:, :, water_bands]
        good_indices, snr_values = identify_bad_bands_snr(temp_image, snr_threshold)

        # Map back to original indices
        good_bands = [water_bands[i] for i in good_indices]
        print(f"Combined removal: Kept {len(good_bands)}/{bands} bands")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract good bands
    cleaned_image = image[:, :, good_bands]

    return cleaned_image, good_bands


def main():
    """Test bad band removal on Indian Pines."""
    print("="*80)
    print("BAD BAND REMOVAL - HYPERSPECTRAL PREPROCESSING")
    print("="*80)

    # Load data
    IMAGE_PATH = "../data/indian_pines/indian_pines_image.mat"
    image = load_hyperspectral_mat(IMAGE_PATH)

    print(f"\nOriginal image shape: {image.shape}")

    # Method 1: SNR-based
    print("\n" + "-"*80)
    print("METHOD 1: SNR-Based Bad Band Removal")
    print("-"*80)
    cleaned_snr, bands_snr = apply_bad_band_removal(image, method='snr', snr_threshold=5)
    print(f"Result shape: {cleaned_snr.shape}")

    # Method 2: Water absorption
    print("\n" + "-"*80)
    print("METHOD 2: Water Absorption Band Removal")
    print("-"*80)
    cleaned_water, bands_water = apply_bad_band_removal(image, method='water')
    print(f"Result shape: {cleaned_water.shape}")

    # Method 3: Both
    print("\n" + "-"*80)
    print("METHOD 3: Combined Approach")
    print("-"*80)
    cleaned_both, bands_both = apply_bad_band_removal(image, method='both', snr_threshold=5)
    print(f"Result shape: {cleaned_both.shape}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original bands: {image.shape[2]}")
    print(f"After SNR removal: {cleaned_snr.shape[2]} ({cleaned_snr.shape[2]/image.shape[2]*100:.1f}%)")
    print(f"After water removal: {cleaned_water.shape[2]} ({cleaned_water.shape[2]/image.shape[2]*100:.1f}%)")
    print(f"After combined: {cleaned_both.shape[2]} ({cleaned_both.shape[2]/image.shape[2]*100:.1f}%)")


if __name__ == "__main__":
    main()
