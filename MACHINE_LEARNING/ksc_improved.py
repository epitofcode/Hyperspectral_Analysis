"""
KSC (Kennedy Space Center) - IMPROVED Classification Pipeline
==============================================================

IMPROVEMENTS OVER BASELINE:
1. Data Augmentation (rotation + flipping) for small classes - 8x more samples
2. Class-balanced SVM to handle class imbalance
3. Grid search for optimal hyperparameters
4. 50/50 train/test split (more training data for small classes)
5. More PCA components (50 instead of 30)

Target: 95%+ Overall Accuracy (vs baseline 62%)
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
import matplotlib.patches as mpatches
import time

sys.path.append(str(Path(__file__).parent))
from image_utils import load_hyperspectral_mat, load_ground_truth, select_rgb_bands

print("="*80)
print("KSC - IMPROVED CLASSIFICATION PIPELINE")
print("="*80)
print("\nIMPROVEMENTS:")
print("  [+] Data augmentation (8x samples for small classes)")
print("  [+] Class-balanced SVM")
print("  [+] Hyperparameter optimization")
print("  [+] 50/50 train/test split")
print("\nTarget: 95%+ OA (baseline was 62%)\n")

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
    augmented = []

    # Original
    augmented.append(patch)

    # Rotations (along spatial dimensions)
    augmented.append(np.rot90(patch, k=1, axes=(0, 1)))  # 90°
    augmented.append(np.rot90(patch, k=2, axes=(0, 1)))  # 180°
    augmented.append(np.rot90(patch, k=3, axes=(0, 1)))  # 270°

    # Flips
    augmented.append(np.flip(patch, axis=0))  # Horizontal flip
    augmented.append(np.flip(patch, axis=1))  # Vertical flip

    # Flip + Rotations
    flipped = np.flip(patch, axis=0)
    augmented.append(np.rot90(flipped, k=1, axes=(0, 1)))
    augmented.append(np.rot90(flipped, k=2, axes=(0, 1)))

    return augmented

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("="*80)
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

# ============================================================================
# STEP 2: PCA DIMENSIONALITY REDUCTION (More components)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PCA DIMENSIONALITY REDUCTION")
print("="*80)

image_2d = image.reshape(-1, bands)
pca = PCA(n_components=50)  # More components for better representation
pca_data = pca.fit_transform(image_2d)
pca_image = pca_data.reshape(h, w, 50)
variance = np.sum(pca.explained_variance_ratio_)*100

print(f"\nReduced from {bands} bands to 50 PCA components")
print(f"Variance preserved: {variance:.2f}%")

# ============================================================================
# STEP 3: SPATIAL-SPECTRAL WITH DATA AUGMENTATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: SPATIAL-SPECTRAL WITH DATA AUGMENTATION")
print("="*80)

PATCH_SIZE = 7
half_patch = PATCH_SIZE // 2

# Pad image
padded_image = np.pad(
    pca_image,
    ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
    mode='symmetric'
)

# Extract patches with augmentation
labeled_mask = gt > 0
labeled_positions = np.argwhere(labeled_mask)
y_labels = gt[labeled_mask]

# Count samples per class
unique_classes, class_counts = np.unique(y_labels, return_counts=True)
class_dict = dict(zip(unique_classes, class_counts))

print(f"\nClass distribution:")
for cls, count in class_dict.items():
    print(f"  Class {cls}: {count} samples")

# Identify small classes (< 500 samples) that need augmentation
small_class_threshold = 500
small_classes = [cls for cls, count in class_dict.items() if count < small_class_threshold]
print(f"\nSmall classes needing augmentation: {small_classes}")
print(f"  (Classes with < {small_class_threshold} samples)")

print(f"\nExtracting patches...")
start_time = time.time()

features = []
labels = []

for i, pos in enumerate(labeled_positions):
    row, col = pos
    label = gt[row, col]

    row_pad = row + half_patch
    col_pad = col + half_patch

    patch = padded_image[
        row_pad - half_patch : row_pad + half_patch + 1,
        col_pad - half_patch : col_pad + half_patch + 1,
        :
    ]

    # Augment small classes
    if label in small_classes:
        augmented_patches = augment_patch(patch)
        for aug_patch in augmented_patches:
            features.append(aug_patch.flatten())
            labels.append(label)
    else:
        # No augmentation for large classes
        features.append(patch.flatten())
        labels.append(label)

    if (i + 1) % 2000 == 0:
        print(f"  Progress: {i+1:,}/{len(labeled_positions):,} ({100*(i+1)/len(labeled_positions):.1f}%)")

X_all = np.array(features)
y_all = np.array(labels)

extract_time = time.time() - start_time
print(f"\nPatch extraction completed in {extract_time:.1f}s")
print(f"Total samples after augmentation: {len(X_all):,}")
print(f"Feature dimension: {X_all.shape[1]} ({PATCH_SIZE}x{PATCH_SIZE}x50)")

# Show augmented class distribution
unique_aug, counts_aug = np.unique(y_all, return_counts=True)
print(f"\nAugmented class distribution:")
for cls, count in zip(unique_aug, counts_aug):
    original_count = class_dict[cls]
    multiplier = count / original_count
    print(f"  Class {cls}: {original_count} -> {count} samples ({multiplier:.1f}x)")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT (50/50 for more training data)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN/TEST SPLIT (50/50)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.5, random_state=42, stratify=y_all
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 5: HYPERPARAMETER TUNING WITH GRID SEARCH
# ============================================================================
print("\n" + "="*80)
print("STEP 5: HYPERPARAMETER OPTIMIZATION")
print("="*80)

print("\nSearching for optimal SVM parameters...")
print("This may take a few minutes...")

# Grid search with class balancing
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1]
}

svm_base = SVC(kernel='rbf', cache_size=2000, class_weight='balanced')

# Use smaller subset for grid search (faster)
n_cv_samples = min(10000, len(X_train))
indices = np.random.choice(len(X_train), n_cv_samples, replace=False)
X_cv = X_train_scaled[indices]
y_cv = y_train[indices]

grid_search = GridSearchCV(
    svm_base,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

start_time = time.time()
grid_search.fit(X_cv, y_cv)
grid_time = time.time() - start_time

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"\nGrid search completed in {grid_time:.1f}s")
print(f"Best parameters: C={best_params['C']}, gamma={best_params['gamma']}")
print(f"Best CV accuracy: {best_score*100:.2f}%")

# ============================================================================
# STEP 6: TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TRAINING FINAL MODEL")
print("="*80)

print(f"\nTraining SVM with optimized parameters...")
print(f"  C = {best_params['C']}")
print(f"  gamma = {best_params['gamma']}")
print(f"  class_weight = 'balanced'")

start_time = time.time()
svm_final = SVC(
    C=best_params['C'],
    kernel='rbf',
    gamma=best_params['gamma'],
    cache_size=2000,
    class_weight='balanced'
)
svm_final.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

print(f"Training completed in {train_time:.1f}s")

# ============================================================================
# STEP 7: EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: EVALUATION")
print("="*80)

print("\nPredicting on test set...")
y_pred = svm_final.predict(X_test_scaled)

# Calculate metrics
oa = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
class_acc = cm.diagonal() / cm.sum(axis=1)
aa = np.mean(class_acc)
kappa = cohen_kappa_score(y_test, y_pred)

print(f"\n{'='*50}")
print("IMPROVED RESULTS")
print(f"{'='*50}")
print(f"Overall Accuracy (OA):  {oa*100:.2f}%")
print(f"Average Accuracy (AA):  {aa*100:.2f}%")
print(f"Kappa Coefficient:      {kappa:.4f}")

print(f"\n{'='*50}")
print("COMPARISON WITH BASELINE")
print(f"{'='*50}")
print(f"Baseline OA:            62.86%")
print(f"Improved OA:            {oa*100:.2f}%")
print(f"Improvement:            +{oa*100 - 62.86:.2f}%")

print(f"\n{'='*50}")
print("PER-CLASS ACCURACY")
print(f"{'='*50}")

class_names = [
    'Scrub', 'Willow swamp', 'CP hammock', 'CP/Oak', 'Slash pine',
    'Oak/Broadleaf', 'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh',
    'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water'
]

for i, cls in enumerate(unique_aug):
    name = class_names[cls-1]
    acc = class_acc[i] * 100
    status = "[OK]" if acc > 70 else "[WARN]" if acc > 40 else "[FAIL]"
    print(f"{status} Class {cls:2d} ({name:20s}): {acc:6.2f}%")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: SAVING RESULTS")
print("="*80)

output_dir = Path('../results/ksc')
output_dir.mkdir(parents=True, exist_ok=True)

results_file = output_dir / 'improved_results.txt'
with open(results_file, 'w') as f:
    f.write("KSC - IMPROVED CLASSIFICATION RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("IMPROVEMENTS APPLIED:\n")
    f.write("-"*40 + "\n")
    f.write("1. Data Augmentation (8x for small classes)\n")
    f.write("2. Class-balanced SVM\n")
    f.write("3. Hyperparameter optimization (Grid Search)\n")
    f.write("4. 50/50 train/test split\n")
    f.write("5. 50 PCA components\n\n")

    f.write("OPTIMIZED PARAMETERS:\n")
    f.write("-"*40 + "\n")
    f.write(f"C:           {best_params['C']}\n")
    f.write(f"gamma:       {best_params['gamma']}\n")
    f.write(f"class_weight: balanced\n\n")

    f.write("RESULTS:\n")
    f.write("-"*40 + "\n")
    f.write(f"Overall Accuracy:  {oa*100:.2f}%\n")
    f.write(f"Average Accuracy:  {aa*100:.2f}%\n")
    f.write(f"Kappa Coefficient: {kappa:.4f}\n\n")

    f.write("COMPARISON WITH BASELINE:\n")
    f.write("-"*40 + "\n")
    f.write(f"Baseline OA:       62.86%\n")
    f.write(f"Improved OA:       {oa*100:.2f}%\n")
    f.write(f"Improvement:       +{oa*100 - 62.86:.2f}%\n\n")

    f.write("PER-CLASS ACCURACY:\n")
    f.write("-"*40 + "\n")
    for i, cls in enumerate(unique_aug):
        name = class_names[cls-1]
        acc = class_acc[i] * 100
        f.write(f"Class {cls:2d} ({name:20s}): {acc:6.2f}%\n")

    f.write(f"\nTraining time: {train_time:.1f}s\n")
    f.write(f"Total samples after augmentation: {len(X_all):,}\n")

print(f"Results saved to: {results_file}")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nFinal Accuracy: {oa*100:.2f}%")
print(f"Target: 95%+")
if oa >= 0.95:
    print("[SUCCESS] TARGET ACHIEVED!")
elif oa >= 0.90:
    print("[WARN] Close to target - may need deep learning for 95%+")
else:
    print("[INFO] Need further improvements")
print("="*80)
