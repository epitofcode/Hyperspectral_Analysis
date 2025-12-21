"""
Inspect KSC dataset to understand the data structure and values
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("INSPECTING KSC DATASET")
print("="*80)

# Load image
print("\n1. Loading KSC image...")
img_data = sio.loadmat('../data/ksc/ksc_image.mat')
print(f"Keys in .mat file: {list(img_data.keys())}")

# Find data key
data_key = [k for k in img_data.keys() if not k.startswith('__')][0]
print(f"Data key: '{data_key}'")

image = img_data[data_key]
print(f"\nImage shape: {image.shape}")
print(f"Data type: {image.dtype}")
print(f"Value range: [{image.min()}, {image.max()}]")
print(f"Mean: {image.mean():.4f}")
print(f"Std: {image.std():.4f}")

# Check if there are any non-zero pixels
non_zero_count = np.count_nonzero(image)
total_pixels = image.size
print(f"\nNon-zero values: {non_zero_count:,} / {total_pixels:,} ({non_zero_count/total_pixels*100:.2f}%)")

# Check individual bands
print("\n2. Band statistics (first 10 bands):")
for i in range(min(10, image.shape[2])):
    band = image[:, :, i]
    print(f"Band {i:3d}: min={band.min():8.2f}, max={band.max():8.2f}, mean={band.mean():8.2f}, std={band.std():8.2f}, non-zero={np.count_nonzero(band):,}")

# Check specific bands for RGB
print("\n3. Checking RGB band candidates:")
n_bands = image.shape[2]
red_band = int(n_bands * 0.6)
green_band = int(n_bands * 0.4)
blue_band = int(n_bands * 0.2)

print(f"Red band (60%): {red_band}")
print(f"  Range: [{image[:,:,red_band].min()}, {image[:,:,red_band].max()}]")
print(f"  Mean: {image[:,:,red_band].mean():.4f}")
print(f"  Non-zero: {np.count_nonzero(image[:,:,red_band]):,}")

print(f"\nGreen band (40%): {green_band}")
print(f"  Range: [{image[:,:,green_band].min()}, {image[:,:,green_band].max()}]")
print(f"  Mean: {image[:,:,green_band].mean():.4f}")
print(f"  Non-zero: {np.count_nonzero(image[:,:,green_band]):,}")

print(f"\nBlue band (20%): {blue_band}")
print(f"  Range: [{image[:,:,blue_band].min()}, {image[:,:,blue_band].max()}]")
print(f"  Mean: {image[:,:,blue_band].mean():.4f}")
print(f"  Non-zero: {np.count_nonzero(image[:,:,blue_band]):,}")

# Try different band combinations
print("\n4. Trying better band combinations:")
# For AVIRIS data, typical good bands are:
# Red: ~650nm (band ~50-60)
# Green: ~550nm (band ~30-40)
# Blue: ~450nm (band ~10-20)

better_red = 50
better_green = 30
better_blue = 10

if better_red < n_bands and better_green < n_bands and better_blue < n_bands:
    print(f"\nBetter RGB candidates:")
    print(f"Red band {better_red}: mean={image[:,:,better_red].mean():.4f}, max={image[:,:,better_red].max():.2f}")
    print(f"Green band {better_green}: mean={image[:,:,better_green].mean():.4f}, max={image[:,:,better_green].max():.2f}")
    print(f"Blue band {better_blue}: mean={image[:,:,better_blue].mean():.4f}, max={image[:,:,better_blue].max():.2f}")

# Check ground truth
print("\n5. Loading ground truth...")
gt_data = sio.loadmat('../data/ksc/ksc_gt.mat')
gt_key = [k for k in gt_data.keys() if not k.startswith('__')][0]
print(f"GT key: '{gt_key}'")

gt = gt_data[gt_key]
print(f"GT shape: {gt.shape}")
print(f"Unique classes: {np.unique(gt)}")
print(f"Class distribution: {np.bincount(gt.flatten())}")

# Visualize a few bands to see what's happening
print("\n6. Creating visualization...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('KSC Dataset Inspection', fontsize=16, fontweight='bold')

# Show several individual bands
for idx, band_idx in enumerate([0, 10, 20, 30, 50, 70, 100, 120]):
    if band_idx < n_bands:
        ax = axes[idx // 4, idx % 4]
        band = image[:, :, band_idx]
        # Normalize for visualization
        band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
        im = ax.imshow(band_norm, cmap='gray')
        ax.set_title(f'Band {band_idx}\nRange:[{band.min():.0f}, {band.max():.0f}]', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('../ksc_inspection.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: ksc_inspection.png")
plt.show()

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
