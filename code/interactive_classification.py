"""
Interactive Hyperspectral Image Classification Pipeline

Step-by-step spatial-spectral classification with comprehensive visualizations.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import time

# ===== Configuration =====
DATASET = "indian_pines"
IMAGE_PATH = f"../data/{DATASET}/{DATASET}_image.mat"
GT_PATH = f"../data/{DATASET}/{DATASET}_gt.mat"

N_PCA_COMPONENTS = 50
PATCH_SIZE = 11
TRAIN_RATIO = 0.30
RANDOM_STATE = 42

sys.path.append(str(Path(__file__).parent))
from image_utils import load_hyperspectral_mat, load_ground_truth


def pause_for_visualization():
    """Display plot and wait for user input."""
    plt.tight_layout()
    plt.show()
    input("\nPress Enter to continue to next step...")
    print("\n")


print("="*80)
print("INTERACTIVE HYPERSPECTRAL IMAGE CLASSIFICATION")
print("="*80)
print(f"\nDataset: {DATASET}")
print(f"Configuration: {N_PCA_COMPONENTS} PCA | {PATCH_SIZE}x{PATCH_SIZE} patches | {TRAIN_RATIO*100:.0f}% train")
print("\nVisualizations will be shown at each step.")
input("Press Enter to start...")

# ===== STEP 1: Load Data =====
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

image = load_hyperspectral_mat(IMAGE_PATH)
gt = load_ground_truth(GT_PATH)

height, width, bands = image.shape
classes = np.unique(gt)
classes = classes[classes > 0]
n_classes = len(classes)

print(f"Image: {height}×{width}×{bands} | Classes: {n_classes}")
print("Creating visualization...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'{DATASET.upper()} - Data Overview', fontsize=16, fontweight='bold')

# RGB Composite
from image_utils import select_rgb_bands
rgb = select_rgb_bands(image)
axes[0, 0].imshow(rgb)
axes[0, 0].set_title('RGB Composite', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Ground Truth
cmap = plt.cm.get_cmap('tab20', n_classes)
im1 = axes[0, 1].imshow(gt, cmap=cmap)
axes[0, 1].set_title('Ground Truth Labels', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], label='Class', fraction=0.046)

# Class distribution
class_counts = [np.sum(gt == c) for c in classes]
axes[0, 2].bar(classes, class_counts, color='steelblue', edgecolor='black')
axes[0, 2].set_title('Class Distribution', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Class ID')
axes[0, 2].set_ylabel('Number of Pixels')
axes[0, 2].grid(axis='y', alpha=0.3)

# Sample bands
band_indices = [0, bands//4, bands//2, 3*bands//4, bands-1]
for idx, band_idx in enumerate(band_indices[:3]):
    if idx < 3:
        axes[1, idx].imshow(image[:, :, band_idx], cmap='gray')
        axes[1, idx].set_title(f'Band {band_idx}', fontsize=10)
        axes[1, idx].axis('off')

pause_for_visualization()

# ===== STEP 2: PCA Dimensionality Reduction =====
print("="*80)
print("STEP 2: PCA DIMENSIONALITY REDUCTION")
print("="*80)

image_2d = image.reshape(-1, bands)
print(f"Reducing {bands} bands → {N_PCA_COMPONENTS} components...")
start_time = time.time()
pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
reduced_2d = pca.fit_transform(image_2d)
pca_time = time.time() - start_time

processed_image = reduced_2d.reshape(height, width, N_PCA_COMPONENTS)
total_variance = pca.explained_variance_ratio_.sum()
print(f"Completed in {pca_time:.2f}s | Variance: {total_variance*100:.2f}%")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('PCA Dimensionality Reduction Results', fontsize=16, fontweight='bold')

# First 3 PCA components
for i in range(3):
    axes[0, i].imshow(processed_image[:, :, i], cmap='viridis')
    axes[0, i].set_title(f'PC {i+1} ({pca.explained_variance_ratio_[i]*100:.2f}% var)',
                        fontsize=10, fontweight='bold')
    axes[0, i].axis('off')

# Variance explained by each component
axes[1, 0].bar(range(1, N_PCA_COMPONENTS+1), pca.explained_variance_ratio_*100,
               color='coral', edgecolor='black')
axes[1, 0].set_title('Variance Explained per Component', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Principal Component')
axes[1, 0].set_ylabel('Variance Explained (%)')
axes[1, 0].grid(axis='y', alpha=0.3)

# Cumulative variance
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)*100
axes[1, 1].plot(range(1, N_PCA_COMPONENTS+1), cumsum_variance,
                linewidth=2, color='darkblue', marker='o', markersize=4)
axes[1, 1].axhline(y=99, color='red', linestyle='--', label='99% threshold')
axes[1, 1].set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_ylabel('Cumulative Variance (%)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Stats box
stats_text = f"Original bands: {bands}\n"
stats_text += f"PCA components: {N_PCA_COMPONENTS}\n"
stats_text += f"Total variance: {total_variance*100:.2f}%\n"
stats_text += f"First 3 PCs: {pca.explained_variance_ratio_[:3].sum()*100:.2f}%\n"
stats_text += f"Processing time: {pca_time:.2f}s"
axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 2].axis('off')

pause_for_visualization()

# ===== STEP 3: Spatial-Spectral Feature Extraction =====
print("="*80)
print("STEP 3: SPATIAL-SPECTRAL FEATURE EXTRACTION")
print("="*80)
print(f"Extracting {PATCH_SIZE}×{PATCH_SIZE} spatial patches (key for 90%+ accuracy)")
half_patch = PATCH_SIZE // 2
padded_image = np.pad(
    processed_image,
    ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
    mode='symmetric'
)

labeled_mask = gt > 0
labeled_positions = np.argwhere(labeled_mask)
n_labeled = len(labeled_positions)

feature_dim = PATCH_SIZE * PATCH_SIZE * N_PCA_COMPONENTS
print(f"Labeled pixels: {n_labeled} | Feature dim: {feature_dim}")

start_time = time.time()
features = []
labels = []

for i, pos in enumerate(labeled_positions):
    row, col = pos
    row_padded = row + half_patch
    col_padded = col + half_patch

    # Extract patch
    patch = padded_image[
        row_padded - half_patch : row_padded + half_patch + 1,
        col_padded - half_patch : col_padded + half_patch + 1,
        :
    ]

    features.append(patch.flatten())
    labels.append(gt[row, col])

    if (i+1) % 1000 == 0:
        print(f"  Processed {i+1}/{n_labeled} pixels...")

X_all = np.array(features)
y_all = np.array(labels)
extract_time = time.time() - start_time
print(f"Completed in {extract_time:.2f}s | Shape: {X_all.shape}")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'Spatial-Spectral Patch Extraction ({PATCH_SIZE}×{PATCH_SIZE} Neighborhood)',
             fontsize=16, fontweight='bold')
example_classes = classes[:min(4, len(classes))]
for idx, class_id in enumerate(example_classes):
    class_positions = labeled_positions[y_all == class_id]
    if len(class_positions) > 0:
        pos = class_positions[len(class_positions)//2]
        row, col = pos
        marker_img = rgb.copy()
        marker_size = PATCH_SIZE
        row_start = max(0, row - marker_size//2)
        row_end = min(height, row + marker_size//2 + 1)
        col_start = max(0, col - marker_size//2)
        col_end = min(width, col + marker_size//2 + 1)

        axes[0, idx].imshow(rgb)
        rect = plt.Rectangle((col_start, row_start), col_end-col_start, row_end-row_start,
                             linewidth=2, edgecolor='red', facecolor='none')
        axes[0, idx].add_patch(rect)
        axes[0, idx].plot(col, row, 'r*', markersize=15)
        axes[0, idx].set_title(f'Class {class_id} - Location', fontsize=10, fontweight='bold')
        axes[0, idx].axis('off')
        axes[0, idx].set_xlim(max(0, col-50), min(width, col+50))
        axes[0, idx].set_ylim(min(height, row+50), max(0, row-50))

        row_padded = row + half_patch
        col_padded = col + half_patch
        patch = padded_image[
            row_padded - half_patch : row_padded + half_patch + 1,
            col_padded - half_patch : col_padded + half_patch + 1,
            0
        ]
        axes[1, idx].imshow(patch, cmap='viridis')
        axes[1, idx].set_title(f'{PATCH_SIZE}x{PATCH_SIZE} Patch (PC1)', fontsize=10, fontweight='bold')
        axes[1, idx].axis('off')

pause_for_visualization()

# ===== STEP 4: Train/Test Split =====
print("="*80)
print("STEP 4: TRAIN/TEST SPLIT")
print("="*80)
print(f"Split: {TRAIN_RATIO*100:.0f}% train / {(1-TRAIN_RATIO)*100:.0f}% test")

np.random.seed(RANDOM_STATE)

train_indices = []
test_indices = []

print("\nSplit per class:")
for class_id in classes:
    class_mask = y_all == class_id
    class_indices = np.where(class_mask)[0]

    n_class = len(class_indices)
    n_train = max(5, int(n_class * TRAIN_RATIO))

    np.random.shuffle(class_indices)
    train_indices.extend(class_indices[:n_train])
    test_indices.extend(class_indices[n_train:])

    print(f"  Class {class_id}: {n_class} total, {n_train} train, {n_class-n_train} test")

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

X_train = X_all[train_indices]
y_train = y_all[train_indices]
train_positions = labeled_positions[train_indices]

X_test = X_all[test_indices]
y_test = y_all[test_indices]
test_positions = labeled_positions[test_indices]

print(f"\nFinal split: {len(X_train)} train, {len(X_test)} test")

# Visualize split
print("\nCreating train/test split visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Train/Test Split Visualization', fontsize=16, fontweight='bold')

# Split overlay
split_vis = np.zeros((height, width, 3), dtype=np.uint8)
split_vis[test_positions[:, 0], test_positions[:, 1], 1] = 255  # Green = test
split_vis[train_positions[:, 0], train_positions[:, 1], 0] = 255  # Red = train

axes[0].imshow(split_vis)
axes[0].set_title('Red=Train, Green=Test', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Class distribution in train/test
train_class_counts = [np.sum(y_train == c) for c in classes]
test_class_counts = [np.sum(y_test == c) for c in classes]

x = np.arange(len(classes))
width_bar = 0.35
axes[1].bar(x - width_bar/2, train_class_counts, width_bar, label='Train', color='red', alpha=0.7)
axes[1].bar(x + width_bar/2, test_class_counts, width_bar, label='Test', color='green', alpha=0.7)
axes[1].set_xlabel('Class ID')
axes[1].set_ylabel('Number of Samples')
axes[1].set_title('Train/Test Distribution per Class', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

pause_for_visualization()

# ===== STEP 5: SVM Training with Grid Search =====
print("="*80)
print("STEP 5: SVM TRAINING WITH GRID SEARCH")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Grid search for optimal C and gamma (this takes a few minutes)...\n")

param_grid = {
    'C': [10, 100, 1000],
    'gamma': ['scale', 0.001, 0.01, 0.1]
}

svm = SVC(kernel='rbf', random_state=RANDOM_STATE)
grid = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=2)

start_time = time.time()
grid.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

print(f"\nCompleted in {train_time:.2f}s")
print(f"Best: C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']} | CV={grid.best_score_*100:.2f}%")

classifier = grid.best_estimator_
cv_results = grid.cv_results_
means = cv_results['mean_test_score']
params = cv_results['params']

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Grid Search Results - SVM Hyperparameter Optimization',
             fontsize=16, fontweight='bold')

param_labels = [f"C={p['C']}, γ={p['gamma']}" for p in params]
colors = ['green' if m == max(means) else 'steelblue' for m in means]

ax.barh(range(len(means)), means, color=colors, edgecolor='black')
ax.set_yticks(range(len(means)))
ax.set_yticklabels(param_labels, fontsize=8)
ax.set_xlabel('Cross-Validation Accuracy', fontsize=12)
ax.set_title(f'Best: C={grid.best_params_["C"]}, γ={grid.best_params_["gamma"]} '
             f'({grid.best_score_*100:.2f}%)', fontsize=12, fontweight='bold')
ax.axvline(x=grid.best_score_, color='red', linestyle='--', linewidth=2, label='Best score')
ax.legend()
ax.grid(axis='x', alpha=0.3)

pause_for_visualization()

# ===== STEP 6: Prediction and Evaluation =====
print("="*80)
print("STEP 6: MAKING PREDICTIONS")
print("="*80)
y_pred = classifier.predict(X_test_scaled)

# Calculate metrics
oa = np.sum(y_test == y_pred) / len(y_test)

class_accuracies = []
for class_id in classes:
    class_mask = y_test == class_id
    if np.sum(class_mask) > 0:
        class_acc = np.sum((y_test == y_pred) & class_mask) / np.sum(class_mask)
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

aa = np.mean(class_accuracies)
kappa = cohen_kappa_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

if kappa < 0.20:
    interpretation = "Slight"
elif kappa < 0.40:
    interpretation = "Fair"
elif kappa < 0.60:
    interpretation = "Moderate"
elif kappa < 0.80:
    interpretation = "Substantial"
else:
    interpretation = "Almost perfect"

print(f"\nOA: {oa*100:.2f}% | AA: {aa*100:.2f}% | Kappa: {kappa:.4f} ({interpretation})")
print("\nPer-Class Accuracy:")
for i, class_id in enumerate(classes):
    print(f"  Class {class_id:2d}: {class_accuracies[i]*100:6.2f}%")

# ===== STEP 7: Final Comprehensive Visualization =====
print("\n" + "="*80)
print("STEP 7: CREATING FINAL VISUALIZATIONS")
print("="*80)

classification_map = np.zeros((height, width), dtype=np.int32)
classification_map[test_positions[:, 0], test_positions[:, 1]] = y_pred

fig = plt.figure(figsize=(20, 10))
fig.suptitle(f'{DATASET.upper()} - Classification Results | OA: {oa*100:.2f}% | AA: {aa*100:.2f}% | Kappa: {kappa:.4f}',
             fontsize=18, fontweight='bold')

gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.25)

# Row 1: Original RGB
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(rgb)
ax1.set_title('Original RGB Image', fontsize=13, fontweight='bold')
ax1.axis('off')

# Row 1: Ground Truth
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(gt, cmap=cmap)
ax2.set_title('Ground Truth Labels', fontsize=13, fontweight='bold')
ax2.axis('off')

# Row 1: Classification Map
ax3 = fig.add_subplot(gs[0, 2])
classification_masked = np.ma.masked_where(classification_map == 0, classification_map)
ax3.imshow(classification_masked, cmap=cmap)
ax3.set_title('Predicted Classification', fontsize=13, fontweight='bold')
ax3.axis('off')

# Row 1: Overlay
ax4 = fig.add_subplot(gs[0, 3])
ax4.imshow(rgb)
ax4.imshow(classification_masked, cmap=cmap, alpha=0.5)
ax4.set_title('Overlay on RGB', fontsize=13, fontweight='bold')
ax4.axis('off')

# Row 2: Per-class accuracy
ax5 = fig.add_subplot(gs[1, 0])
colors_bar = ['red' if acc < 50 else 'orange' if acc < 80 else 'green'
              for acc in np.array(class_accuracies)*100]
ax5.bar(classes, np.array(class_accuracies)*100, color=colors_bar, edgecolor='black', alpha=0.8)
ax5.set_xlabel('Class ID', fontsize=11)
ax5.set_ylabel('Accuracy (%)', fontsize=11)
ax5.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
ax5.axhline(y=80, color='blue', linestyle='--', alpha=0.5, linewidth=2)
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim([0, 105])

# Row 2: Confusion matrix
ax6 = fig.add_subplot(gs[1, 1:3])
im = ax6.imshow(cm, cmap='Blues', aspect='auto')
ax6.set_xlabel('Predicted Class', fontsize=11)
ax6.set_ylabel('True Class', fontsize=11)
ax6.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
ax6.set_xticks(range(len(classes)))
ax6.set_yticks(range(len(classes)))
ax6.set_xticklabels(classes, fontsize=9)
ax6.set_yticklabels(classes, fontsize=9)
plt.colorbar(im, ax=ax6, fraction=0.03)

# Row 2: Summary statistics
ax7 = fig.add_subplot(gs[1, 3])
summary_text = f"RESULTS\n"
summary_text += f"{'─'*32}\n"
summary_text += f"Overall Accuracy:  {oa*100:.2f}%\n"
summary_text += f"Average Accuracy:  {aa*100:.2f}%\n"
summary_text += f"Kappa:             {kappa:.4f}\n"
summary_text += f"Status:            {interpretation}\n\n"
summary_text += f"CONFIG\n"
summary_text += f"{'─'*32}\n"
summary_text += f"Dataset:           {DATASET}\n"
summary_text += f"PCA:               {N_PCA_COMPONENTS}\n"
summary_text += f"Patch:             {PATCH_SIZE}×{PATCH_SIZE}\n"
summary_text += f"Train:             {len(X_train)}\n"
summary_text += f"Test:              {len(X_test)}\n"
summary_text += f"Best C:            {grid.best_params_['C']}\n"
summary_text += f"Best gamma:        {grid.best_params_['gamma']}\n"

ax7.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax7.axis('off')

print("Showing comprehensive results...")
pause_for_visualization()

print("\n" + "="*80)
print("CLASSIFICATION COMPLETE!")
print("="*80)
print(f"Final Accuracy: {oa*100:.2f}% | Modify parameters at top to experiment")
print("="*80)
