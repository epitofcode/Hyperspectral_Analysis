"""
Test fixed KSC visualization
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from image_utils import load_hyperspectral_mat, load_ground_truth, select_rgb_bands

print("="*80)
print("TESTING FIXED KSC VISUALIZATION")
print("="*80)

# Load data
print("\n1. Loading KSC dataset...")
image = load_hyperspectral_mat('../data/ksc/ksc_image.mat')
gt = load_ground_truth('../data/ksc/ksc_gt.mat')

# Create RGB with fixed function
print("\n2. Creating RGB composite with fixed function...")
rgb = select_rgb_bands(image)
print(f"RGB shape: {rgb.shape}")
print(f"RGB range: [{rgb.min():.4f}, {rgb.max():.4f}]")
print(f"RGB mean: {rgb.mean():.4f}")

# Visualize
print("\n3. Creating visualization...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('KSC Dataset - FIXED Visualization', fontsize=16, fontweight='bold')

# RGB composite
axes[0].imshow(rgb)
axes[0].set_title('RGB Composite\n(Bands 50-R, 30-G, 10-B)', fontsize=12)
axes[0].axis('off')

# Ground truth
gt_masked = np.ma.masked_where(gt == 0, gt)
im1 = axes[1].imshow(gt_masked, cmap='tab20')
axes[1].set_title('Ground Truth Labels', fontsize=12)
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Overlay
axes[2].imshow(rgb)
axes[2].imshow(gt_masked, cmap='tab20', alpha=0.5)
axes[2].set_title('Classification Overlay', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('../ksc_visualization_fixed.png', dpi=200, bbox_inches='tight')
print(f"\nVisualization saved to: ksc_visualization_fixed.png")
plt.show()

print("\n" + "="*80)
print("VISUALIZATION TEST COMPLETE")
print("="*80)
print("\nIf the RGB looks good (shows actual wetland scene), the fix worked!")
print("If it still looks wrong, the dataset might be corrupted.")
