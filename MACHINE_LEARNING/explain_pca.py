"""
Visualize HOW PCA works on KSC dataset
Shows what PCA actually computes and why we keep 50 components
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from image_utils import load_hyperspectral_mat

print("="*80)
print("EXPLAINING PCA: What Does It Actually Compute?")
print("="*80)

# Load KSC data
print("\n1. Loading KSC hyperspectral image...")
image = load_hyperspectral_mat('../data/ksc/ksc_image.mat')
h, w, bands = image.shape
print(f"Shape: {image.shape}")

# Reshape for PCA
X = image.reshape(-1, bands)
print(f"Reshaped to: {X.shape} (pixels × bands)")

# Apply PCA
print("\n2. Computing PCA...")
pca = PCA(n_components=176)  # All components to see full spectrum
X_pca = pca.fit_transform(X)

# Get eigenvalues (variance)
explained_variance = pca.explained_variance_
variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)

print(f"\nPCA Complete!")
print(f"Total components: {len(explained_variance)}")

# Show top components
print("\n3. Top 10 Principal Components:")
print(f"{'PC':<5} {'Variance':>12} {'% of Total':>12} {'Cumulative %':>15}")
print("-" * 50)
for i in range(10):
    print(f"PC{i+1:<3} {explained_variance[i]:>12.1f} {variance_ratio[i]*100:>11.2f}% {cumulative_variance[i]*100:>14.2f}%")

print("\n4. Key milestones:")
pc_for_90 = np.argmax(cumulative_variance >= 0.90) + 1
pc_for_95 = np.argmax(cumulative_variance >= 0.95) + 1
pc_for_99 = np.argmax(cumulative_variance >= 0.99) + 1

print(f"  First {pc_for_90} PCs capture 90% variance")
print(f"  First {pc_for_95} PCs capture 95% variance")
print(f"  First {pc_for_99} PCs capture 99% variance")
print(f"  We chose: 50 PCs -> {cumulative_variance[49]*100:.2f}% variance")

# Show first eigenvector (what PC1 actually is)
print("\n5. What is PC1 (the first principal component)?")
print("   PC1 = weighted combination of all bands:")
pc1_weights = pca.components_[0]
print(f"   PC1 = {pc1_weights[0]:.3f}*Band1 + {pc1_weights[1]:.3f}*Band2 + ... + {pc1_weights[-1]:.3f}*Band176")
print(f"   Strongest contributions from bands: {np.argsort(np.abs(pc1_weights))[-5:][::-1]}")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
fig.suptitle('How PCA Works on KSC Hyperspectral Data', fontsize=16, fontweight='bold')

# 1. Variance per component (log scale)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(range(1, 177), explained_variance, 'b-', linewidth=2)
ax1.axvline(x=50, color='r', linestyle='--', linewidth=2, label='Our choice (50 PCs)')
ax1.set_xlabel('Principal Component Number')
ax1.set_ylabel('Variance (Eigenvalue)')
ax1.set_title('Variance Captured by Each PC')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Cumulative variance
ax2 = plt.subplot(2, 3, 2)
ax2.plot(range(1, 177), cumulative_variance * 100, 'g-', linewidth=2)
ax2.axhline(y=90, color='orange', linestyle=':', label='90% variance')
ax2.axhline(y=95, color='red', linestyle=':', label='95% variance')
ax2.axvline(x=50, color='r', linestyle='--', linewidth=2, label=f'50 PCs ({cumulative_variance[49]*100:.1f}%)')
ax2.fill_between(range(1, 51), 0, 100, alpha=0.2, color='green', label='Components we keep')
ax2.set_xlabel('Number of Principal Components')
ax2.set_ylabel('Cumulative Variance Explained (%)')
ax2.set_title('Cumulative Variance: Why 50 PCs?')
ax2.set_xlim([0, 176])
ax2.set_ylim([0, 105])
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Variance ratio (first 50)
ax3 = plt.subplot(2, 3, 3)
ax3.bar(range(1, 51), variance_ratio[:50] * 100, color='steelblue', alpha=0.7)
ax3.set_xlabel('Principal Component Number')
ax3.set_ylabel('Variance Explained (%)')
ax3.set_title('Variance Contribution (First 50 PCs)')
ax3.grid(True, alpha=0.3, axis='y')

# 4. First 3 eigenvectors (PC loadings)
ax4 = plt.subplot(2, 3, 4)
ax4.plot(range(1, 177), pca.components_[0], 'r-', label='PC1', linewidth=2, alpha=0.7)
ax4.plot(range(1, 177), pca.components_[1], 'g-', label='PC2', linewidth=2, alpha=0.7)
ax4.plot(range(1, 177), pca.components_[2], 'b-', label='PC3', linewidth=2, alpha=0.7)
ax4.set_xlabel('Original Band Number')
ax4.set_ylabel('Loading (Weight)')
ax4.set_title('PC Loadings: How Bands Combine')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.text(0.02, 0.98, 'Each PC is a weighted\ncombination of ALL bands',
         transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 5. Signal vs Noise cutoff
ax5 = plt.subplot(2, 3, 5)
signal_pcs = variance_ratio[:50]
noise_pcs = variance_ratio[50:]
ax5.hist(signal_pcs * 100, bins=20, alpha=0.7, color='green', label='Signal PCs (1-50)')
ax5.hist(noise_pcs * 100, bins=20, alpha=0.7, color='red', label='Noise PCs (51-176)')
ax5.set_xlabel('Variance Explained (%)')
ax5.set_ylabel('Number of Components')
ax5.set_title('Signal vs Noise Distribution')
ax5.legend()
ax5.set_yscale('log')

# 6. Information retention table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
table_data = [
    ['PCs Kept', 'Variance', 'Info Loss', 'Benefit'],
    ['10', f'{cumulative_variance[9]*100:.1f}%', f'{(1-cumulative_variance[9])*100:.1f}%', 'Very fast'],
    ['30', f'{cumulative_variance[29]*100:.1f}%', f'{(1-cumulative_variance[29])*100:.1f}%', 'Fast'],
    ['50 (ours)', f'{cumulative_variance[49]*100:.1f}%', f'{(1-cumulative_variance[49])*100:.1f}%', 'Good balance'],
    ['100', f'{cumulative_variance[99]*100:.1f}%', f'{(1-cumulative_variance[99])*100:.1f}%', 'Slower'],
    ['176 (all)', '100.0%', '0.0%', 'Very slow + overfitting'],
]

table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight our choice
for i in range(4):
    table[(3, i)].set_facecolor('#90EE90')
    table[(3, i)].set_text_props(weight='bold')

ax6.set_title('PCA Dimensionality Reduction Trade-offs', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('../pca_explanation.png', dpi=200, bbox_inches='tight')
print(f"\nVisualization saved to: pca_explanation.png")
plt.show()

# Show actual PC images
print("\n6. Visualizing first 3 principal component images...")
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('First 3 Principal Components (What PCA Creates)', fontsize=14, fontweight='bold')

for i in range(3):
    pc_image = X_pca[:, i].reshape(h, w)
    # Normalize for display
    pc_norm = (pc_image - pc_image.min()) / (pc_image.max() - pc_image.min())

    axes[i].imshow(pc_norm, cmap='viridis')
    axes[i].set_title(f'PC{i+1}\n({variance_ratio[i]*100:.1f}% of variance)')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('../pca_images.png', dpi=150, bbox_inches='tight')
print(f"PC images saved to: pca_images.png")
plt.show()

print("\n" + "="*80)
print("EXPLANATION COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. PCA finds DIRECTIONS of maximum variance in 176D space")
print("2. These directions are COMBINATIONS of all original bands")
print("3. Eigenvalues tell us how much variance each PC captures")
print("4. We keep top 50 PCs that capture 93.83% of variance")
print("5. Bottom 126 PCs are mostly noise (< 7% variance)")
print("\nPCA doesn't 'remove bands' - it TRANSFORMS them into better features!")
