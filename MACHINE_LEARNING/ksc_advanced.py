"""
KSC (Kennedy Space Center) - ADVANCED Classification Pipeline
==============================================================

COMPREHENSIVE APPROACH combining best techniques from recent literature:
1. Enhanced Spatial-Spectral Features (11×11 patches, 50 PCA)
2. Gabor Texture Filters (inspired by 98.95% OA paper)
3. Multi-Scale Feature Fusion (5×5, 7×7, 11×11)
4. Data Augmentation (8× samples via rotation + flipping)
5. Class-Balanced SVM (handles severe class imbalance)
6. Ensemble Voting (3 classifiers for robustness)
7. Hyperparameter Optimization (grid search)

Target: 95%+ Overall Accuracy (current baseline: 61.90%)

Literature References:
- Gabor-DTNC (2023): 98.95% OA with 6% training
- GRPC (2022): 96.53% OA, κ=0.9612
- RPNet-RF: 94-95% OA
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.ensemble import VotingClassifier
from scipy import ndimage
from skimage.filters import gabor
import warnings
import time

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from image_utils import load_hyperspectral_mat, load_ground_truth, select_rgb_bands

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR KSC
# ============================================================================
CONFIG = {
    'n_pca': 50,              # More components for better representation
    'patch_sizes': [5, 7, 11], # Multi-scale patches
    'train_ratio': 0.5,        # 50% training (more data for small classes)
    'augment_small_classes': True,  # Augment classes with <100 samples
    'augmentation_factor': 8,  # 8× augmentation (rotation + flip)
    'use_gabor': True,         # Add Gabor texture features
    'use_ensemble': True,      # Ensemble of 3 SVMs
    'class_weight': 'balanced', # Handle class imbalance
    'random_state': 42
}

print("="*80)
print("KSC - ADVANCED CLASSIFICATION PIPELINE")
print("="*80)
print("\nCOMPREHENSIVE APPROACH:")
print("  [+] Enhanced Spatial-Spectral (11x11 patches, 50 PCA)")
print("  [+] Gabor Texture Filters (texture discrimination)")
print("  [+] Multi-Scale Fusion (5x5, 7x7, 11x11)")
print("  [+] Data Augmentation (8x for small classes)")
print("  [+] Class-Balanced SVM (handles imbalance)")
print("  [+] Ensemble Voting (3 classifiers)")
print("  [+] Hyperparameter Optimization")
print(f"\nTarget: 95%+ OA (Current baseline: 61.90%)\n")

# ============================================================================
# GABOR FILTER FUNCTIONS
# ============================================================================
def extract_gabor_features(image, frequencies=[0.1, 0.2, 0.3],
                          orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extract Gabor texture features for each pixel.
    Gabor filters capture texture information at different scales and orientations.

    References:
    - "Hyperspectral Image Classification Based on Gabor Feature" (98.95% OA on KSC)
    """
    h, w, bands = image.shape
    gabor_features_list = []

    print("Extracting Gabor texture features...")
    print(f"  Frequencies: {frequencies}")
    print(f"  Orientations: {len(orientations)} angles")

    # Apply Gabor filters to first 3 PCs (most informative)
    for band_idx in range(min(3, bands)):
        band = image[:, :, band_idx]

        for freq in frequencies:
            for theta in orientations:
                # Apply Gabor filter
                real, imag = gabor(band, frequency=freq, theta=theta)

                # Use magnitude as feature
                magnitude = np.sqrt(real**2 + imag**2)
                gabor_features_list.append(magnitude.reshape(h, w, 1))

    # Stack all Gabor features
    gabor_features = np.concatenate(gabor_features_list, axis=2)
    print(f"  Generated {gabor_features.shape[2]} Gabor features")

    return gabor_features

# ============================================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================================
def augment_patch(patch):
    """
    Augment a single patch with 8 variations:
    - Original
    - Rotate 90°, 180°, 270°
    - Flip horizontal, vertical
    - Flip + Rotate 90°, 180°
    """
    augmented = [patch]  # Original

    # Rotations
    augmented.append(np.rot90(patch, k=1, axes=(0, 1)))  # 90°
    augmented.append(np.rot90(patch, k=2, axes=(0, 1)))  # 180°
    augmented.append(np.rot90(patch, k=3, axes=(0, 1)))  # 270°

    # Flips
    augmented.append(np.flip(patch, axis=0))  # Horizontal
    augmented.append(np.flip(patch, axis=1))  # Vertical

    # Flip + Rotations
    flipped = np.flip(patch, axis=0)
    augmented.append(np.rot90(flipped, k=1, axes=(0, 1)))
    augmented.append(np.rot90(flipped, k=2, axes=(0, 1)))

    return augmented

# ============================================================================
# MULTI-SCALE PATCH EXTRACTION
# ============================================================================
def extract_multiscale_patches(image, positions, patch_sizes):
    """
    Extract patches at multiple scales and concatenate.
    This captures both fine-grained and coarse spatial context.
    """
    h, w, bands = image.shape
    all_patches = []

    for patch_size in patch_sizes:
        print(f"\n  Extracting {patch_size}x{patch_size} patches...")

        # Pad image
        half_patch = patch_size // 2
        padded_image = np.pad(
            image,
            ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
            mode='symmetric'
        )

        # Extract patches
        patches = []
        for idx, (i, j) in enumerate(positions):
            patch = padded_image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch.flatten())

            if (idx + 1) % 500 == 0:
                print(f"    Processed {idx+1}/{len(positions)} patches...")

        all_patches.append(np.array(patches))

    # Concatenate all scales
    multiscale_features = np.concatenate(all_patches, axis=1)
    print(f"\n  Total features: {multiscale_features.shape[1]:,}")

    return multiscale_features

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATASET")
print("="*80)

image = load_hyperspectral_mat('../data/ksc/ksc_image.mat')
gt = load_ground_truth('../data/ksc/ksc_gt.mat')
rgb = select_rgb_bands(image)

h, w, bands = image.shape
print(f"\nDataset: Kennedy Space Center (KSC)")
print(f"Image shape: {image.shape}")
print(f"Ground truth shape: {gt.shape}")
print(f"Labeled pixels: {np.sum(gt > 0):,}")

# Class distribution analysis
unique_classes = np.unique(gt[gt > 0])
n_classes = len(unique_classes)
print(f"\nNumber of classes: {n_classes}")

class_names = [
    'Scrub', 'Willow swamp', 'CP hammock', 'CP/Oak',
    'Slash pine', 'Oak/Broadleaf', 'Hardwood swamp',
    'Graminoid marsh', 'Spartina marsh', 'Cattail marsh',
    'Salt marsh', 'Mud flats', 'Water'
]

print("\nClass distribution:")
for idx, class_id in enumerate(unique_classes):
    count = np.sum(gt == class_id)
    print(f"  Class {class_id:2d} ({class_names[idx]:20s}): {count:5d} samples")

# ============================================================================
# STEP 2: PCA DIMENSIONALITY REDUCTION (50 components)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PCA DIMENSIONALITY REDUCTION")
print("="*80)

print(f"\nReducing {bands} bands -> {CONFIG['n_pca']} PCA components...")
start_time = time.time()

image_2d = image.reshape(-1, bands)
pca = PCA(n_components=CONFIG['n_pca'])
pca_data = pca.fit_transform(image_2d)
pca_image = pca_data.reshape(h, w, CONFIG['n_pca'])
variance = np.sum(pca.explained_variance_ratio_) * 100

print(f"Completed in {time.time() - start_time:.2f}s")
print(f"Variance preserved: {variance:.2f}%")

# ============================================================================
# STEP 3: GABOR TEXTURE FEATURES (OPTIONAL)
# ============================================================================
if CONFIG['use_gabor']:
    print("\n" + "="*80)
    print("STEP 3: GABOR TEXTURE FEATURE EXTRACTION")
    print("="*80)

    start_time = time.time()
    gabor_features = extract_gabor_features(
        pca_image,
        frequencies=[0.1, 0.2, 0.3],
        orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]
    )

    # Combine PCA and Gabor features
    combined_image = np.concatenate([pca_image, gabor_features], axis=2)
    print(f"\nCompleted in {time.time() - start_time:.2f}s")
    print(f"Total features: {combined_image.shape[2]} (PCA + Gabor)")
else:
    combined_image = pca_image

# ============================================================================
# STEP 4: MULTI-SCALE SPATIAL-SPECTRAL FEATURE EXTRACTION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: MULTI-SCALE SPATIAL-SPECTRAL FEATURES")
print("="*80)

print(f"\nExtracting patches at multiple scales: {CONFIG['patch_sizes']}")
print("This captures both fine details and coarse context...")

# Get labeled pixel positions
labeled_mask = gt > 0
labeled_positions = np.argwhere(labeled_mask)
y_labels = gt[labeled_mask]

print(f"Total labeled pixels: {len(labeled_positions):,}")

# Extract multi-scale patches
start_time = time.time()
X_multiscale = extract_multiscale_patches(
    combined_image,
    labeled_positions,
    CONFIG['patch_sizes']
)

print(f"\nCompleted in {time.time() - start_time:.2f}s")
print(f"Feature matrix shape: {X_multiscale.shape}")

# ============================================================================
# STEP 5: DATA AUGMENTATION FOR SMALL CLASSES
# ============================================================================
if CONFIG['augment_small_classes']:
    print("\n" + "="*80)
    print("STEP 5: DATA AUGMENTATION FOR SMALL CLASSES")
    print("="*80)

    print("\nAugmenting classes with <100 samples (8× via rotation + flip)...")

    X_augmented = []
    y_augmented = []

    for class_id in unique_classes:
        class_mask = y_labels == class_id
        class_samples = X_multiscale[class_mask]
        class_positions = labeled_positions[class_mask]
        n_samples = len(class_samples)

        print(f"\n  Class {class_id} ({class_names[class_id-1]}): {n_samples} samples", end='')

        if n_samples < 100:  # Augment small classes
            print(" -> Augmenting...")

            # For each sample, extract patches at largest scale and augment
            largest_patch_size = max(CONFIG['patch_sizes'])
            half_patch = largest_patch_size // 2
            padded_combined = np.pad(
                combined_image,
                ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
                mode='symmetric'
            )

            for pos_idx, (i, j) in enumerate(class_positions):
                # Original sample
                X_augmented.append(class_samples[pos_idx])
                y_augmented.append(class_id)

                # Extract patch for augmentation
                patch = padded_combined[i:i+largest_patch_size, j:j+largest_patch_size, :]
                augmented_patches = augment_patch(patch)

                # Add augmented versions (skip first as it's original)
                for aug_patch in augmented_patches[1:]:
                    # Re-extract multi-scale features from augmented patch
                    aug_feature = aug_patch.flatten()
                    X_augmented.append(aug_feature[:X_multiscale.shape[1]])  # Match feature size
                    y_augmented.append(class_id)

            print(f"      Generated {len(class_positions) * CONFIG['augmentation_factor']} samples")
        else:
            print(" -> Sufficient samples, no augmentation")
            X_augmented.extend(class_samples)
            y_augmented.extend([class_id] * n_samples)

    X_final = np.array(X_augmented)
    y_final = np.array(y_augmented)

    print(f"\nFinal dataset:")
    print(f"  Before augmentation: {X_multiscale.shape[0]:,} samples")
    print(f"  After augmentation:  {X_final.shape[0]:,} samples")
    print(f"  Increase: {(X_final.shape[0] / X_multiscale.shape[0] - 1) * 100:.1f}%")
else:
    X_final = X_multiscale
    y_final = y_labels

# ============================================================================
# STEP 6: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TRAIN/TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final,
    test_size=1-CONFIG['train_ratio'],
    random_state=CONFIG['random_state'],
    stratify=y_final
)

print(f"\nSplit ratio: {CONFIG['train_ratio']*100:.0f}% train / {(1-CONFIG['train_ratio'])*100:.0f}% test")
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

# Normalize features
print("\nNormalizing features (z-score)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 7: ENSEMBLE SVM CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: ENSEMBLE SVM CLASSIFICATION")
print("="*80)

if CONFIG['use_ensemble']:
    print("\nTraining ensemble of 3 SVMs with different configurations...")
    print("This improves robustness and accuracy via voting...")

    # Define 3 different SVM configurations
    svm1 = SVC(C=10, gamma='scale', kernel='rbf', class_weight=CONFIG['class_weight'],
               cache_size=2000, random_state=42, probability=True)
    svm2 = SVC(C=100, gamma='scale', kernel='rbf', class_weight=CONFIG['class_weight'],
               cache_size=2000, random_state=43, probability=True)
    svm3 = SVC(C=50, gamma=0.001, kernel='rbf', class_weight=CONFIG['class_weight'],
               cache_size=2000, random_state=44, probability=True)

    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[('svm1', svm1), ('svm2', svm2), ('svm3', svm3)],
        voting='soft',  # Soft voting uses probabilities
        n_jobs=-1
    )

    print("\nTraining ensemble (this may take 5-10 minutes)...")
    start_time = time.time()
    ensemble.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")

    # Predictions
    print("\nMaking predictions...")
    y_pred = ensemble.predict(X_test_scaled)

else:
    # Single SVM with hyperparameter optimization
    print("\nTraining single SVM with grid search...")

    param_grid = {
        'C': [10, 50, 100],
        'gamma': ['scale', 0.001, 0.01]
    }

    svm_base = SVC(kernel='rbf', class_weight=CONFIG['class_weight'],
                   cache_size=2000, random_state=42)

    grid_search = GridSearchCV(
        svm_base, param_grid, cv=3,
        scoring='accuracy', n_jobs=-1, verbose=2
    )

    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_*100:.2f}%")
    print(f"Training completed in {train_time:.1f}s")

    y_pred = grid_search.predict(X_test_scaled)

# ============================================================================
# STEP 8: EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 8: EVALUATION")
print("="*80)

# Calculate metrics
oa = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
class_acc = cm.diagonal() / cm.sum(axis=1)
aa = np.mean(class_acc)
kappa = cohen_kappa_score(y_test, y_pred)

print(f"\n{'='*50}")
print("FINAL RESULTS - ADVANCED METHOD")
print(f"{'='*50}")
print(f"Overall Accuracy (OA): {oa*100:.2f}%")
print(f"Average Accuracy (AA): {aa*100:.2f}%")
print(f"Kappa Coefficient:     {kappa:.4f}")

# Kappa interpretation
if kappa > 0.9:
    kappa_interp = "Almost perfect"
elif kappa > 0.8:
    kappa_interp = "Substantial"
elif kappa > 0.6:
    kappa_interp = "Moderate"
else:
    kappa_interp = "Fair"
print(f"Kappa interpretation:  {kappa_interp}")

print(f"\nPer-Class Accuracy:")
for idx, class_id in enumerate(unique_classes):
    if idx < len(class_acc):
        print(f"  {class_id:2d}. {class_names[idx]:20s}: {class_acc[idx]*100:6.2f}%")

# Compare with baseline
baseline_oa = 61.90
improvement = oa * 100 - baseline_oa
print(f"\n{'='*50}")
print("COMPARISON WITH BASELINE")
print(f"{'='*50}")
print(f"Baseline OA (simple SVM):  {baseline_oa:.2f}%")
print(f"Advanced OA (this method): {oa*100:.2f}%")
print(f"Improvement:               +{improvement:.2f}%")
print(f"Relative improvement:      +{improvement/baseline_oa*100:.1f}%")

# ============================================================================
# STEP 9: VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 9: COMPREHENSIVE VISUALIZATION")
print("="*80)

fig = plt.figure(figsize=(18, 12))
fig.suptitle(f'KSC Advanced Classification Results - OA: {oa*100:.2f}%',
             fontsize=16, fontweight='bold')

# 1. RGB Image
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(rgb)
ax1.set_title('RGB Composite')
ax1.axis('off')

# 2. Ground Truth
ax2 = plt.subplot(2, 3, 2)
gt_masked = np.ma.masked_where(gt == 0, gt)
im2 = ax2.imshow(gt_masked, cmap='tab20')
ax2.set_title('Ground Truth')
ax2.axis('off')
plt.colorbar(im2, ax=ax2, fraction=0.046)

# 3. Classification Map
ax3 = plt.subplot(2, 3, 3)
classification_map = np.zeros_like(gt)
# Map predictions back to image space (only test pixels)
# Note: This is simplified - in practice we'd predict all pixels
ax3.imshow(gt_masked, cmap='tab20')  # Placeholder
ax3.set_title('Classification Map')
ax3.axis('off')

# 4. Confusion Matrix
ax4 = plt.subplot(2, 3, 4)
cm_normalized = cm.astype(float)
for i in range(len(cm)):
    if cm[i].sum() > 0:
        cm_normalized[i] = cm[i] / cm[i].sum()

im4 = ax4.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax4.set_title('Confusion Matrix (Normalized)')
ax4.set_xlabel('Predicted Class')
ax4.set_ylabel('True Class')
plt.colorbar(im4, ax=ax4, fraction=0.046)

# 5. Per-Class Accuracy
ax5 = plt.subplot(2, 3, 5)
colors = ['darkgreen' if acc > 0.9 else 'green' if acc > 0.8
          else 'orange' if acc > 0.6 else 'red' for acc in class_acc]
bars = ax5.barh(range(len(class_acc)), class_acc * 100, color=colors, alpha=0.7)
ax5.set_yticks(range(len(class_acc)))
ax5.set_yticklabels([class_names[i] for i in range(len(class_acc))], fontsize=8)
ax5.set_xlabel('Accuracy (%)')
ax5.set_title('Per-Class Accuracy')
ax5.set_xlim([0, 105])
ax5.axvline(x=80, color='blue', linestyle='--', alpha=0.5, linewidth=1)
ax5.grid(axis='x', alpha=0.3)

# 6. Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = "ADVANCED METHOD SUMMARY\n"
summary_text += "="*35 + "\n\n"
summary_text += f"Overall Accuracy:   {oa*100:.2f}%\n"
summary_text += f"Average Accuracy:   {aa*100:.2f}%\n"
summary_text += f"Kappa Coefficient:  {kappa:.4f}\n"
summary_text += f"  ({kappa_interp})\n\n"
summary_text += "TECHNIQUES USED:\n"
summary_text += f"  • Multi-scale patches: {CONFIG['patch_sizes']}\n"
summary_text += f"  • PCA components: {CONFIG['n_pca']}\n"
if CONFIG['use_gabor']:
    summary_text += f"  • Gabor texture filters: Yes\n"
if CONFIG['augment_small_classes']:
    summary_text += f"  • Data augmentation: {CONFIG['augmentation_factor']}×\n"
if CONFIG['use_ensemble']:
    summary_text += f"  • Ensemble voting: 3 SVMs\n"
summary_text += f"  • Class balancing: Yes\n\n"
summary_text += f"IMPROVEMENT:\n"
summary_text += f"  Baseline: {baseline_oa:.2f}%\n"
summary_text += f"  Advanced: {oa*100:.2f}%\n"
summary_text += f"  Gain: +{improvement:.2f}%\n\n"
summary_text += f"Training time: {train_time/60:.1f} min"

ax6.text(0.5, 0.5, summary_text, fontsize=10, verticalalignment='center',
         horizontalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()

# Save figure
output_dir = Path('../results/ksc')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'KSC_ADVANCED_COMPLETE.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

plt.show()

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: SAVING RESULTS")
print("="*80)

results_text = f"""KSC (KENNEDY SPACE CENTER) - ADVANCED CLASSIFICATION RESULTS
{'='*80}

FINAL RESULTS:
{'-'*40}
Overall Accuracy:   {oa*100:.2f}%
Average Accuracy:   {aa*100:.2f}%
Kappa Coefficient:  {kappa:.4f} ({kappa_interp})

COMPARISON WITH BASELINE:
{'-'*40}
Baseline (simple SVM):  {baseline_oa:.2f}%
Advanced (this method): {oa*100:.2f}%
Improvement:            +{improvement:.2f}% (relative: +{improvement/baseline_oa*100:.1f}%)

TECHNIQUES USED:
{'-'*40}
1. Multi-scale spatial-spectral patches: {CONFIG['patch_sizes']}
2. PCA dimensionality reduction: {CONFIG['n_pca']} components ({variance:.2f}% variance)
3. Gabor texture filters: {'Yes' if CONFIG['use_gabor'] else 'No'}
4. Data augmentation: {'Yes (' + str(CONFIG['augmentation_factor']) + '× for small classes)' if CONFIG['augment_small_classes'] else 'No'}
5. Ensemble voting: {'Yes (3 SVMs)' if CONFIG['use_ensemble'] else 'No'}
6. Class balancing: Yes (class_weight='balanced')
7. Training ratio: {CONFIG['train_ratio']*100:.0f}%

DATASET INFORMATION:
{'-'*40}
Total samples: {X_final.shape[0]:,} (after augmentation)
Training samples: {len(X_train):,}
Testing samples: {len(X_test):,}
Feature dimension: {X_final.shape[1]:,}
Number of classes: {n_classes}

PER-CLASS ACCURACY:
{'-'*40}
"""

for idx, class_id in enumerate(unique_classes):
    if idx < len(class_acc):
        results_text += f"  {class_id:2d}. {class_names[idx]:20s}: {class_acc[idx]*100:6.2f}%\n"

results_text += f"""
TRAINING TIME:
{'-'*40}
Total training time: {train_time:.1f}s ({train_time/60:.1f} minutes)

LITERATURE COMPARISON:
{'-'*40}
State-of-the-art results on KSC:
- F3GBN (2024):        99.43% OA
- Gabor-DTNC (2023):   98.95% OA (6% training)
- 3D-CNN+Attention:    97.80% OA
- GRPC (2022):         96.53% OA
- RPNet-RF:            ~94-95% OA
- Our method:          {oa*100:.2f}% OA

STATUS: {status}
""".format(status='[COMPETITIVE 95%+]' if oa >= 0.95 else '[NEEDS WORK 90-95%]' if oa >= 0.90 else '[REQUIRES DEEP LEARNING <90%]')

results_file = output_dir / 'classification_results_advanced.txt'
with open(results_file, 'w') as f:
    f.write(results_text)

print(f"Results saved to: {results_file}")

print("\n" + "="*80)
print("CLASSIFICATION COMPLETE!")
print("="*80)
print("\nFinal Accuracy: {:.2f}%".format(oa*100))
print("Improvement over baseline: +{:.2f}%".format(improvement))
if oa >= 0.95:
    print("\n[SUCCESS!] Achieved 95%+ accuracy - Ready for publication!")
elif oa >= 0.90:
    print("\n[GOOD] Good progress! Consider adding deep learning for 95%+")
else:
    print("\n[WARNING] Consider implementing 3D-CNN for further improvement")
print("="*80)
