"""
KSC (Kennedy Space Center) - Spatial-Spectral Classification
==============================================================

Simplified baseline approach following CLAUDE.md pragmatism principle:
"Avoid adding preprocessing steps or algorithmic complexity unless they
provide a statistically significant and justifiable improvement"

Dataset: Kennedy Space Center (KSC) - 13 wetland vegetation classes
Sensor: AVIRIS (512 x 614 x 176 bands)

Method (MINIMAL):
- PCA dimensionality reduction (50 components, ~94% variance)
- Single-scale spatial patches (11x11)
- Single SVM classifier with class balancing
- No augmentation, no Gabor, no ensemble complexity
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import warnings
import time

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from image_utils import load_hyperspectral_mat, load_ground_truth, select_rgb_bands

# ==================================================================
# CONFIGURATION - SIMPLIFIED
# ==================================================================
CONFIG = {
    'n_pca': 50,
    'patch_size': 11,       # Single patch size
    'train_ratio': 0.5,
    'random_state': 42
}

print("="*80)
print("KSC - SIMPLIFIED BASELINE CLASSIFICATION")
print("="*80)
print("\nSimplified approach (following CLAUDE.md pragmatism):")
print(f"  PCA: {CONFIG['n_pca']} components")
print(f"  Patch size: {CONFIG['patch_size']}x{CONFIG['patch_size']}")
print(f"  Classifier: Single SVM with class balancing")
print(f"  No Gabor, No augmentation, No ensemble")
print()

# Class names
class_names = [
    'Scrub', 'Willow swamp', 'CP hammock', 'CP/Oak',
    'Slash pine', 'Oak/Broadleaf', 'Hardwood swamp',
    'Graminoid marsh', 'Spartina marsh', 'Cattail marsh',
    'Salt marsh', 'Mud flats', 'Water'
]

# ==================================================================
# FUNCTIONS
# ==================================================================

def extract_patches(image, positions, patch_size):
    """Extract spatial patches at given positions."""
    h, w, bands = image.shape
    radius = patch_size // 2

    # Pad image
    padded_image = np.pad(image,
                         ((radius, radius), (radius, radius), (0, 0)),
                         mode='reflect')

    patches = []
    for row, col in positions:
        adj_row = row + radius
        adj_col = col + radius

        patch = padded_image[
            adj_row - radius:adj_row + radius + 1,
            adj_col - radius:adj_col + radius + 1,
            :
        ]
        patches.append(patch.flatten())

    return np.array(patches)

# ==================================================================
# STEP 1: LOAD DATA
# ==================================================================
print("="*80)
print("STEP 1: LOADING KSC DATASET")
print("="*80)

data_dir = Path(__file__).parent.parent / 'data' / 'ksc'
image = load_hyperspectral_mat(data_dir / 'ksc_image.mat')
gt = load_ground_truth(data_dir / 'ksc_gt.mat')

h, w, bands = image.shape
print(f"\nImage shape: {h} x {w} x {bands} bands")
print(f"Ground truth shape: {gt.shape}")

unique_classes = np.unique(gt[gt > 0])
n_classes = len(unique_classes)
print(f"Number of classes: {n_classes}")

print("\nClass distribution:")
for idx, class_id in enumerate(unique_classes):
    count = np.sum(gt == class_id)
    print(f"  Class {class_id:2d} ({class_names[idx]:20s}): {count:5d} samples")

# ==================================================================
# STEP 2: PCA DIMENSIONALITY REDUCTION
# ==================================================================
print("\n" + "="*80)
print("STEP 2: PCA DIMENSIONALITY REDUCTION")
print("="*80)

print(f"\nReducing {bands} bands -> {CONFIG['n_pca']} PCA components...")
start_time = time.time()

image_2d = image.reshape(-1, bands)
pca = PCA(n_components=CONFIG['n_pca'], random_state=CONFIG['random_state'])
pca_data = pca.fit_transform(image_2d)
pca_image = pca_data.reshape(h, w, CONFIG['n_pca'])
variance = np.sum(pca.explained_variance_ratio_) * 100

print(f"  Completed in {time.time() - start_time:.2f}s")
print(f"  Variance preserved: {variance:.2f}%")

# ==================================================================
# STEP 3: SPATIAL-SPECTRAL FEATURE EXTRACTION
# ==================================================================
print("\n" + "="*80)
print("STEP 3: SPATIAL-SPECTRAL FEATURE EXTRACTION")
print("="*80)

labeled_mask = gt > 0
labeled_positions = np.argwhere(labeled_mask)
y_labels = gt[labeled_mask]

print(f"\nTotal labeled pixels: {len(labeled_positions):,}")
print(f"Extracting {CONFIG['patch_size']}x{CONFIG['patch_size']} patches...")

start_time = time.time()
X = extract_patches(pca_image, labeled_positions, CONFIG['patch_size'])

print(f"  Completed in {time.time() - start_time:.2f}s")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Features per pixel: {X.shape[1]:,}")

# ==================================================================
# STEP 4: TRAIN/TEST SPLIT
# ==================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN/TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels,
    train_size=CONFIG['train_ratio'],
    random_state=CONFIG['random_state'],
    stratify=y_labels
)

print(f"\nDataset split:")
print(f"  Training: {len(X_train):,} samples")
print(f"  Testing:  {len(X_test):,} samples")

# Per-class distribution
print(f"\nPer-class training samples:")
for class_id in unique_classes:
    train_count = np.sum(y_train == class_id)
    test_count = np.sum(y_test == class_id)
    print(f"  Class {class_id:2d} ({class_names[class_id-1]:20s}): {train_count:4d} train, {test_count:4d} test")

# ==================================================================
# STEP 5: FEATURE STANDARDIZATION
# ==================================================================
print("\n" + "="*80)
print("STEP 5: FEATURE STANDARDIZATION")
print("="*80)

print("\nStandardizing features (zero mean, unit variance)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  Standardization complete")

# ==================================================================
# STEP 6: SVM CLASSIFIER TRAINING
# ==================================================================
print("\n" + "="*80)
print("STEP 6: SVM CLASSIFIER TRAINING")
print("="*80)

print("\nTraining SVM with RBF kernel...")
print("  Parameters: C=100, gamma='scale', class_weight='balanced'")

start_time = time.time()
svm = SVC(
    kernel='rbf',
    C=100,
    gamma='scale',
    class_weight='balanced',
    random_state=CONFIG['random_state']
)
svm.fit(X_train_scaled, y_train)

print(f"  Training completed in {time.time() - start_time:.2f}s")

# ==================================================================
# STEP 7: PREDICTION AND EVALUATION
# ==================================================================
print("\n" + "="*80)
print("STEP 7: PREDICTION AND EVALUATION")
print("="*80)

print("\nPredicting on test set...")
y_pred = svm.predict(X_test_scaled)

# Calculate metrics
oa = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Per-class accuracy
per_class_acc = []
for class_id in unique_classes:
    class_mask = y_test == class_id
    if np.sum(class_mask) > 0:
        class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
        per_class_acc.append(class_acc * 100)
    else:
        per_class_acc.append(0.0)

aa = np.mean(per_class_acc)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nOverall Accuracy (OA): {oa*100:.2f}%")
print(f"Average Accuracy (AA): {aa:.2f}%")
print(f"Kappa Coefficient:     {kappa:.4f}")

print("\nPer-Class Accuracy:")
print("-" * 60)
for idx, class_id in enumerate(unique_classes):
    print(f"  {class_id:2d}. {class_names[idx]:20s}: {per_class_acc[idx]:6.2f}%")
print("-" * 60)

# ==================================================================
# SAVE RESULTS
# ==================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_dir = Path(__file__).parent / 'RESULTS' / 'ksc'
output_dir.mkdir(parents=True, exist_ok=True)

# Save text results
results_file = output_dir / 'ksc_results.txt'
with open(results_file, 'w') as f:
    f.write("KSC (KENNEDY SPACE CENTER) - SIMPLIFIED BASELINE RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("CONFIGURATION:\n")
    f.write("-" * 40 + "\n")
    f.write("Method: Simplified baseline (CLAUDE.md pragmatism principle)\n")
    f.write(f"PCA components: {CONFIG['n_pca']}\n")
    f.write(f"Patch size: {CONFIG['patch_size']}x{CONFIG['patch_size']}\n")
    f.write(f"Classifier: Single SVM (RBF, C=100, class_weight='balanced')\n")
    f.write(f"No Gabor, No augmentation, No ensemble\n")

    f.write("\n\nCLASS DISTRIBUTION:\n")
    f.write("-" * 40 + "\n")
    for idx, class_id in enumerate(unique_classes):
        count = np.sum(gt == class_id)
        f.write(f"  Class {class_id:2d} ({class_names[idx]:20s}): {count:5d} samples\n")

    f.write("\n\nOVERALL RESULTS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Overall Accuracy (OA): {oa*100:.2f}%\n")
    f.write(f"Average Accuracy (AA): {aa:.2f}%\n")
    f.write(f"Kappa Coefficient:     {kappa:.4f}\n")

    f.write("\n\nPER-CLASS ACCURACY:\n")
    f.write("-" * 40 + "\n")
    for idx, class_id in enumerate(unique_classes):
        f.write(f"  {class_id:2d}. {class_names[idx]:20s}: {per_class_acc[idx]:6.2f}%\n")

    f.write("\n\nCONFUSION MATRIX:\n")
    f.write("-" * 40 + "\n")
    f.write(str(conf_matrix))

print(f"\nResults saved to: {results_file}")

# ==================================================================
# STEP 8: FULL IMAGE CLASSIFICATION
# ==================================================================
print("\n" + "="*80)
print("STEP 8: FULL IMAGE CLASSIFICATION")
print("="*80)

print("\nClassifying all labeled pixels...")
# Predict on ALL labeled pixels to create full classification map
X_all_scaled = scaler.transform(X)
y_pred_all = svm.predict(X_all_scaled)

# Create classification map
prediction_map = np.zeros((h, w), dtype=np.uint8)
for idx, pos in enumerate(labeled_positions):
    prediction_map[pos[0], pos[1]] = y_pred_all[idx]

print(f"  Classification map created: {prediction_map.shape}")

# ==================================================================
# VISUALIZATION
# ==================================================================
print("\n" + "="*80)
print("STEP 9: GENERATING VISUALIZATIONS")
print("="*80)

print("\nCreating comprehensive visualization with 6 panels...")

# Create RGB composite
rgb_image = select_rgb_bands(image, red_band=50, green_band=30, blue_band=10)

# Create figure with 3x2 layout
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Define colors for classes
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

# Plot 1: RGB composite
axes[0, 0].imshow(rgb_image)
axes[0, 0].set_title('RGB Composite\n(Bands 50-30-10)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Plot 2: Ground truth
gt_colored = np.zeros((h, w, 3))
for idx, class_id in enumerate(unique_classes):
    mask = gt == class_id
    gt_colored[mask] = colors[idx][:3]

axes[0, 1].imshow(gt_colored)
axes[0, 1].set_title('Ground Truth\n(13 wetland classes)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Plot 3: Predicted classification
pred_colored = np.zeros((h, w, 3))
for idx, class_id in enumerate(unique_classes):
    mask = prediction_map == class_id
    pred_colored[mask] = colors[idx][:3]

axes[1, 0].imshow(pred_colored)
axes[1, 0].set_title(f'Predicted Classification\nOA: {oa*100:.2f}%, Kappa: {kappa:.4f}',
                    fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Plot 4: Classification overlay on RGB
rgb_with_overlay = rgb_image.copy()
overlay_alpha = 0.5
for idx, class_id in enumerate(unique_classes):
    mask = prediction_map == class_id
    if np.any(mask):
        rgb_with_overlay[mask] = (rgb_with_overlay[mask] * (1 - overlay_alpha) +
                                  np.array(colors[idx][:3]) * overlay_alpha)

axes[1, 1].imshow(rgb_with_overlay)
axes[1, 1].set_title('Classification Overlay on RGB\n(50% transparency)',
                    fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# Plot 5: Per-class accuracy
axes[2, 0].barh(range(n_classes), per_class_acc, color=colors[:n_classes])
axes[2, 0].set_yticks(range(n_classes))
axes[2, 0].set_yticklabels([f"{cid}. {class_names[cid-1]}" for cid in unique_classes], fontsize=9)
axes[2, 0].set_xlabel('Accuracy (%)', fontsize=10, fontweight='bold')
axes[2, 0].set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
axes[2, 0].grid(axis='x', alpha=0.3)
axes[2, 0].set_xlim([0, 100])

# Plot 6: Overall metrics
axes[2, 1].axis('off')
metrics_text = f"""
SIMPLIFIED BASELINE

Overall Accuracy (OA)
{oa*100:.2f}%

Average Accuracy (AA)
{aa:.2f}%

Kappa Coefficient
{kappa:.4f}

Configuration
PCA: {CONFIG['n_pca']} components
Patch: {CONFIG['patch_size']}x{CONFIG['patch_size']}
Classifier: Single SVM

Training: {len(X_train):,} samples
Testing: {len(X_test):,} samples
"""
axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=13, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.suptitle('KSC Classification - Simplified Baseline (CLAUDE.md Pragmatism)',
             fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save figure
fig_file = output_dir / 'KSC_RESULTS.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {fig_file}")

print("\n" + "="*80)
print("CLASSIFICATION COMPLETE!")
print("="*80)
print(f"\nSimplified Baseline - Overall Accuracy: {oa*100:.2f}%")
print(f"Results saved to: {output_dir}")
print("="*80 + "\n")

plt.show()
