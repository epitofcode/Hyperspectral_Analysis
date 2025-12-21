"""
PAVIA UNIVERSITY - Complete Hyperspectral Classification Pipeline
================================================================

This script runs the complete classification pipeline:
1. Pixel-wise baseline classification (fast)
2. Spatial-spectral classification with patches (high accuracy)
3. Comprehensive visualization generation

Just run: python pavia.py
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
import matplotlib.patches as mpatches
import time

sys.path.append(str(Path(__file__).parent))
from image_utils import load_hyperspectral_mat, load_ground_truth, select_rgb_bands

print("="*80)
print("PAVIA UNIVERSITY - COMPLETE CLASSIFICATION PIPELINE")
print("="*80)
print("\nThis script will:")
print("  1. Run pixel-wise baseline classification")
print("  2. Run spatial-spectral classification with patches")
print("  3. Generate comprehensive visualizations")
print("\nLet's begin!\n")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATASET")
print("="*80)

image = load_hyperspectral_mat('../data/pavia_university/pavia_university_image.mat')
gt = load_ground_truth('../data/pavia_university/pavia_university_gt.mat')
rgb = select_rgb_bands(image)

h, w, bands = image.shape
print(f"\nDataset: Pavia University")
print(f"Image shape: {image.shape}")
print(f"Ground truth shape: {gt.shape}")
print(f"Labeled pixels: {np.sum(gt > 0):,}")

# Visualize loaded data
print("\n📊 Showing RGB image and Ground Truth...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('STEP 1: Loaded Data', fontsize=14, fontweight='bold')

ax1.imshow(rgb)
ax1.set_title('RGB Image')
ax1.axis('off')

gt_masked = np.ma.masked_where(gt == 0, gt)
im = ax2.imshow(gt_masked, cmap='tab10')
ax2.set_title('Ground Truth Labels')
ax2.axis('off')
plt.colorbar(im, ax=ax2, fraction=0.046)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
input("\n✅ Press Enter to continue to Step 2 (PCA)...")
plt.close()

# ============================================================================
# STEP 2: DIMENSIONALITY REDUCTION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PCA DIMENSIONALITY REDUCTION")
print("="*80)

image_2d = image.reshape(-1, bands)
pca = PCA(n_components=50)
pca_data = pca.fit_transform(image_2d)
pca_image = pca_data.reshape(h, w, 50)
variance = np.sum(pca.explained_variance_ratio_)*100

print(f"\nReduced from {bands} bands to 50 PCA components")
print(f"Variance preserved: {variance:.2f}%")

# Visualize PCA results
print("\n📊 Showing PCA transformation...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('STEP 2: PCA Dimensionality Reduction', fontsize=14, fontweight='bold')

# Show first 3 principal components as RGB
pca_rgb = np.zeros((h, w, 3))
for i in range(3):
    pc = pca_image[:, :, i]
    pca_rgb[:, :, i] = (pc - pc.min()) / (pc.max() - pc.min())

axes[0].imshow(pca_rgb)
axes[0].set_title('First 3 PCs as RGB')
axes[0].axis('off')

# Variance explained
axes[1].plot(np.cumsum(pca.explained_variance_ratio_[:20]) * 100, 'bo-')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_title(f'Variance: {variance:.2f}%')
axes[1].grid(True, alpha=0.3)

# Show PC1
im = axes[2].imshow(pca_image[:, :, 0], cmap='viridis')
axes[2].set_title('Principal Component 1')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
input("\n✅ Press Enter to continue to Step 3 (Pixel-wise Classification)...")
plt.close()

# Extract labeled pixels
labeled_mask = gt > 0
X_pixel = pca_image[labeled_mask]
y = gt[labeled_mask]

# ============================================================================
# STEP 3: PIXEL-WISE BASELINE CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: PIXEL-WISE BASELINE CLASSIFICATION")
print("="*80)
print("\nUsing only spectral information (no spatial patches)")

# Train/test split
X_train_pw, X_test_pw, y_train_pw, y_test_pw = train_test_split(
    X_pixel, y, test_size=0.7, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train_pw):,}")
print(f"Testing samples: {len(X_test_pw):,}")

# Normalize
scaler_pw = StandardScaler()
X_train_pw_scaled = scaler_pw.fit_transform(X_train_pw)
X_test_pw_scaled = scaler_pw.transform(X_test_pw)

# Train SVM
print("\nTraining SVM (pixel-wise)...")
start_time = time.time()
svm_pw = SVC(C=10, kernel='rbf', gamma='scale', cache_size=1000)
svm_pw.fit(X_train_pw_scaled, y_train_pw)
train_time_pw = time.time() - start_time

# Predict
y_pred_pw = svm_pw.predict(X_test_pw_scaled)

# Metrics
oa_pw = accuracy_score(y_test_pw, y_pred_pw)
cm_pw = confusion_matrix(y_test_pw, y_pred_pw)
class_acc_pw = cm_pw.diagonal() / cm_pw.sum(axis=1)
aa_pw = np.mean(class_acc_pw)
kappa_pw = cohen_kappa_score(y_test_pw, y_pred_pw)

print(f"Training time: {train_time_pw:.1f}s")
print(f"\n{'='*40}")
print("PIXEL-WISE RESULTS (BASELINE)")
print(f"{'='*40}")
print(f"Overall Accuracy (OA): {oa_pw*100:.2f}%")
print(f"Average Accuracy (AA): {aa_pw*100:.2f}%")
print(f"Kappa Coefficient:     {kappa_pw:.4f}")

# Visualize pixel-wise results
print("\n📊 Showing pixel-wise classification results...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f'STEP 3: Pixel-wise Results (OA: {oa_pw*100:.1f}%)', fontsize=14, fontweight='bold')

# Confusion matrix
cm_display = cm_pw.astype(float)
for i in range(len(cm_pw)):
    if cm_pw[i].sum() > 0:
        cm_display[i] = cm_pw[i] / cm_pw[i].sum() * 100

im1 = axes[0].imshow(cm_display, cmap='Blues', aspect='auto')
axes[0].set_title('Confusion Matrix (%)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
plt.colorbar(im1, ax=axes[0], fraction=0.046)

# Per-class accuracy
colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in class_acc_pw]
axes[1].barh(range(len(class_acc_pw)), class_acc_pw * 100, color=colors, alpha=0.7)
axes[1].set_xlabel('Accuracy (%)')
axes[1].set_ylabel('Class')
axes[1].set_title('Per-Class Accuracy')
axes[1].set_xlim([0, 105])
axes[1].grid(axis='x', alpha=0.3)

# Metrics summary
axes[2].axis('off')
metrics_text = f"BASELINE RESULTS\n" + "="*25 + "\n\n"
metrics_text += f"Overall Accuracy:\n  {oa_pw*100:.2f}%\n\n"
metrics_text += f"Average Accuracy:\n  {aa_pw*100:.2f}%\n\n"
metrics_text += f"Kappa Coefficient:\n  {kappa_pw:.4f}\n\n"
metrics_text += f"Method:\n  Pixel-wise (spectral only)\n\n"
metrics_text += f"Training time:\n  {train_time_pw:.1f}s"
axes[2].text(0.5, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            horizontalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
input("\n✅ Press Enter to continue to Step 4 (Spatial-Spectral with Patches)...")
plt.close()

# ============================================================================
# STEP 4: SPATIAL-SPECTRAL CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: SPATIAL-SPECTRAL CLASSIFICATION WITH PATCHES")
print("="*80)

PATCH_SIZE = 7
N_PCA = 30

print(f"\nPatch size: {PATCH_SIZE}x{PATCH_SIZE}")
print("Incorporating neighborhood spatial context...")

# Apply PCA with fewer components for speed
print("\nRe-applying PCA with 30 components for spatial-spectral...")
pca_ss = PCA(n_components=N_PCA)
pca_data_ss = pca_ss.fit_transform(image_2d)
pca_image_ss = pca_data_ss.reshape(h, w, N_PCA)

# Pad image for patch extraction
half_patch = PATCH_SIZE // 2
padded_image = np.pad(
    pca_image_ss,
    ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
    mode='symmetric'
)

# Extract patches
labeled_positions = np.argwhere(labeled_mask)
n_samples = len(labeled_positions)

print(f"Extracting {PATCH_SIZE}x{PATCH_SIZE} patches for {n_samples:,} pixels...")
print("This may take a few minutes...")
start_time = time.time()

features = []
labels = []

for i, pos in enumerate(labeled_positions):
    row, col = pos
    row_pad = row + half_patch
    col_pad = col + half_patch

    patch = padded_image[
        row_pad - half_patch : row_pad + half_patch + 1,
        col_pad - half_patch : col_pad + half_patch + 1,
        :
    ]

    features.append(patch.flatten())
    labels.append(gt[row, col])

    if (i + 1) % 10000 == 0:
        elapsed = time.time() - start_time
        print(f"  Progress: {i+1:,}/{n_samples:,} ({100*(i+1)/n_samples:.1f}%) - {elapsed:.1f}s elapsed")

X_all = np.array(features)
y_all = np.array(labels)

extract_time = time.time() - start_time
print(f"Patch extraction completed in {extract_time:.1f}s")
print(f"Feature dimension: {X_all.shape[1]} ({PATCH_SIZE}x{PATCH_SIZE}x{N_PCA})")

# Train/test split
X_train_ss, X_test_ss, y_train_ss, y_test_ss = train_test_split(
    X_all, y_all, test_size=0.7, random_state=42, stratify=y_all
)

# Normalize
scaler_ss = StandardScaler()
X_train_ss_scaled = scaler_ss.fit_transform(X_train_ss)

# Train SVM
print("\nTraining SVM (spatial-spectral)...")
start_time = time.time()
svm_ss = SVC(C=10, kernel='rbf', gamma='scale', cache_size=2000)
svm_ss.fit(X_train_ss_scaled, y_train_ss)
train_time_ss = time.time() - start_time

# Predict in batches
print("Evaluating...")
batch_size = 5000
y_pred_ss = []

for i in range(0, len(X_test_ss), batch_size):
    batch_end = min(i + batch_size, len(X_test_ss))
    X_batch = X_test_ss[i:batch_end]
    X_batch_scaled = scaler_ss.transform(X_batch)
    y_batch_pred = svm_ss.predict(X_batch_scaled)
    y_pred_ss.extend(y_batch_pred)

    if batch_end % 10000 == 0 or batch_end == len(X_test_ss):
        print(f"  Processed: {batch_end:,}/{len(X_test_ss):,} samples")

y_pred_ss = np.array(y_pred_ss)

# Metrics
oa_ss = accuracy_score(y_test_ss, y_pred_ss)
cm_ss = confusion_matrix(y_test_ss, y_pred_ss)
class_acc_ss = cm_ss.diagonal() / cm_ss.sum(axis=1)
aa_ss = np.mean(class_acc_ss)
kappa_ss = cohen_kappa_score(y_test_ss, y_pred_ss)

print(f"Training time: {train_time_ss:.1f}s")
print(f"\n{'='*40}")
print("SPATIAL-SPECTRAL RESULTS")
print(f"{'='*40}")
print(f"Overall Accuracy (OA): {oa_ss*100:.2f}%")
print(f"Average Accuracy (AA): {aa_ss*100:.2f}%")
print(f"Kappa Coefficient:     {kappa_ss:.4f}")

# Visualize spatial-spectral results
print("\n📊 Showing spatial-spectral classification results...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f'STEP 4: Spatial-Spectral Results (OA: {oa_ss*100:.1f}%)', fontsize=14, fontweight='bold')

# Confusion matrix
cm_display_ss = cm_ss.astype(float)
for i in range(len(cm_ss)):
    if cm_ss[i].sum() > 0:
        cm_display_ss[i] = cm_ss[i] / cm_ss[i].sum() * 100

im1 = axes[0].imshow(cm_display_ss, cmap='Greens', aspect='auto')
axes[0].set_title('Confusion Matrix (%)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
plt.colorbar(im1, ax=axes[0], fraction=0.046)

# Per-class accuracy
colors = ['green' if acc > 0.9 else 'orange' if acc > 0.75 else 'red' for acc in class_acc_ss]
axes[1].barh(range(len(class_acc_ss)), class_acc_ss * 100, color=colors, alpha=0.7)
axes[1].set_xlabel('Accuracy (%)')
axes[1].set_ylabel('Class')
axes[1].set_title('Per-Class Accuracy')
axes[1].set_xlim([0, 105])
axes[1].grid(axis='x', alpha=0.3)

# Metrics summary
axes[2].axis('off')
metrics_text = f"SPATIAL-SPECTRAL\n" + "="*25 + "\n\n"
metrics_text += f"Overall Accuracy:\n  {oa_ss*100:.2f}%\n\n"
metrics_text += f"Average Accuracy:\n  {aa_ss*100:.2f}%\n\n"
metrics_text += f"Kappa Coefficient:\n  {kappa_ss:.4f}\n\n"
metrics_text += f"Method:\n  7x7 patches\n\n"
metrics_text += f"Training time:\n  {train_time_ss:.1f}s"
axes[2].text(0.5, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            horizontalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
input("\n✅ Press Enter to see comparison...")
plt.close()

# ============================================================================
# STEP 5: COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: PIXEL-WISE vs SPATIAL-SPECTRAL")
print("="*80)
print(f"Pixel-wise (baseline):        {oa_pw*100:6.2f}% OA")
print(f"Spatial-spectral (patches):   {oa_ss*100:6.2f}% OA")
print(f"Improvement:                  +{oa_ss*100 - oa_pw*100:5.2f}%")
print(f"\nSpatial context provides a {oa_ss*100 - oa_pw*100:.1f}% boost in accuracy!")

# Visualize comparison
print("\n📊 Showing comparison between methods...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('STEP 5: Comparison - Pixel-wise vs Spatial-Spectral', fontsize=14, fontweight='bold')

# Accuracy comparison
methods = ['Pixel-wise\n(baseline)', 'Spatial-Spectral\n(7x7 patches)']
oa_values = [oa_pw * 100, oa_ss * 100]
colors_comp = ['orange', 'green']

axes[0].bar(methods, oa_values, color=colors_comp, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Overall Accuracy (%)', fontsize=12)
axes[0].set_title('Overall Accuracy Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylim([0, 100])
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(oa_values):
    axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

# Improvement bar
axes[1].barh(['Overall Accuracy', 'Average Accuracy', 'Kappa'],
             [(oa_ss - oa_pw) * 100, (aa_ss - aa_pw) * 100, (kappa_ss - kappa_pw) * 100],
             color='green', alpha=0.7)
axes[1].set_xlabel('Improvement', fontsize=12)
axes[1].set_title('Improvement from Spatial Context', fontsize=12, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
input("\n✅ Press Enter to continue to final visualization generation...")
plt.close()

# ============================================================================
# STEP 6: GENERATE CLASSIFICATION MAP
# ============================================================================
print("\n" + "="*80)
print("STEP 5: GENERATING CLASSIFICATION MAP")
print("="*80)

print("\nCreating classification map from predictions...")
classification_map = np.zeros((h, w), dtype=np.int32)

# Use spatial-spectral predictions for test set
test_indices = np.random.RandomState(42).choice(len(labeled_positions), len(y_test_ss), replace=False)
for i, idx in enumerate(test_indices):
    pos = labeled_positions[idx]
    classification_map[pos[0], pos[1]] = y_pred_ss[i]

# Use ground truth for training set visualization
train_indices = [i for i in range(len(labeled_positions)) if i not in test_indices]
for idx in train_indices[:len(y_train_ss)]:
    pos = labeled_positions[idx]
    classification_map[pos[0], pos[1]] = gt[pos[0], pos[1]]

classification_map[gt == 0] = 0

# ============================================================================
# STEP 7: COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CREATING COMPREHENSIVE VISUALIZATION")
print("="*80)

# Class names
all_class_names = [
    'Asphalt', 'Meadows', 'Gravel', 'Trees',
    'Metal sheets', 'Bare Soil', 'Bitumen', 'Bricks', 'Shadows'
]

actual_classes = np.unique(y)
n_actual_classes = len(actual_classes)
class_names = [all_class_names[i-1] for i in actual_classes]

# Create figure
fig = plt.figure(figsize=(20, 14))
fig.suptitle('PAVIA UNIVERSITY - Complete Classification Results | ' +
             f'Spatial-Spectral OA: {oa_ss*100:.2f}% | AA: {aa_ss*100:.2f}% | Kappa: {kappa_ss:.4f}',
             fontsize=18, fontweight='bold')

gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
cmap = plt.cm.get_cmap('tab10', n_actual_classes + 1)

# Row 1: RGB, Ground Truth, Classification
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(rgb)
ax1.set_title('Original RGB Image', fontsize=13, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
gt_masked = np.ma.masked_where(gt == 0, gt)
ax2.imshow(gt_masked, cmap=cmap, vmin=0, vmax=max(actual_classes))
ax2.set_title('Ground Truth Labels', fontsize=13, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
class_masked = np.ma.masked_where(classification_map == 0, classification_map)
ax3.imshow(class_masked, cmap=cmap, vmin=0, vmax=max(actual_classes))
ax3.set_title('Spatial-Spectral Classification', fontsize=13, fontweight='bold')
ax3.axis('off')

# Row 2: Overlay, Confusion Matrix, Per-class Accuracy
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(rgb)
ax4.imshow(class_masked, cmap=cmap, alpha=0.5, vmin=0, vmax=max(actual_classes))
ax4.set_title('Classification Overlay', fontsize=13, fontweight='bold')
ax4.axis('off')

ax5 = fig.add_subplot(gs[1, 1])
cm_display = cm_ss.astype(float)
for i in range(n_actual_classes):
    cm_display[i] = cm_ss[i] / cm_ss[i].sum() * 100
im5 = ax5.imshow(cm_display, cmap='Blues', aspect='auto')
ax5.set_title('Confusion Matrix (%)', fontsize=13, fontweight='bold')
ax5.set_xlabel('Predicted Class')
ax5.set_ylabel('True Class')
ax5.set_xticks(range(n_actual_classes))
ax5.set_yticks(range(n_actual_classes))
ax5.set_xticklabels(actual_classes, fontsize=9)
ax5.set_yticklabels(actual_classes, fontsize=9)
for i in range(n_actual_classes):
    for j in range(n_actual_classes):
        ax5.text(j, i, f'{cm_display[i, j]:.1f}',
                ha="center", va="center",
                color="black" if cm_display[i, j] < 50 else "white", fontsize=8)
plt.colorbar(im5, ax=ax5, fraction=0.046)

ax6 = fig.add_subplot(gs[1, 2])
colors = ['green' if acc > 0.95 else 'orange' if acc > 0.9 else 'red' for acc in class_acc_ss]
bars = ax6.barh(range(n_actual_classes), class_acc_ss * 100, color=colors, alpha=0.7)
ax6.set_yticks(range(n_actual_classes))
ax6.set_yticklabels([f'{actual_classes[i]}. {name[:12]}' for i, name in enumerate(class_names)], fontsize=10)
ax6.set_xlabel('Accuracy (%)', fontsize=11)
ax6.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
ax6.set_xlim([0, 105])
ax6.grid(axis='x', alpha=0.3)
for i, (acc, bar) in enumerate(zip(class_acc_ss, bars)):
    ax6.text(acc * 100 + 2, i, f'{acc*100:.1f}%', va='center', fontsize=9, fontweight='bold')

# Row 3: Legend, Comparison, Dataset Info
ax7 = fig.add_subplot(gs[2, 0])
ax7.axis('off')
legend_elements = [mpatches.Patch(facecolor=cmap(i), label=f'{actual_classes[i]}. {class_names[i]}')
                   for i in range(n_actual_classes)]
ax7.legend(handles=legend_elements, loc='center', fontsize=11,
           title='Class Labels', title_fontsize=12, frameon=True)

ax8 = fig.add_subplot(gs[2, 1])
ax8.axis('off')
summary_text = "RESULTS COMPARISON\n" + "="*35 + "\n\n"
summary_text += "PIXEL-WISE (Baseline):\n"
summary_text += f"  OA: {oa_pw*100:6.2f}%\n"
summary_text += f"  AA: {aa_pw*100:6.2f}%\n"
summary_text += f"  K:  {kappa_pw:7.4f}\n\n"
summary_text += "SPATIAL-SPECTRAL:\n"
summary_text += f"  OA: {oa_ss*100:6.2f}%  (+{oa_ss*100-oa_pw*100:.1f}%)\n"
summary_text += f"  AA: {aa_ss*100:6.2f}%  (+{aa_ss*100-aa_pw*100:.1f}%)\n"
summary_text += f"  K:  {kappa_ss:7.4f}\n\n"
summary_text += f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}\n"
summary_text += f"Training: {len(X_train_ss):,} samples\n"
summary_text += f"Testing:  {len(X_test_ss):,} samples"
ax8.text(0.5, 0.5, summary_text, fontsize=11, verticalalignment='center',
         horizontalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
dataset_text = "DATASET INFO\n" + "="*35 + "\n\n"
dataset_text += f"Dataset: Pavia University\n"
dataset_text += f"Location: Pavia, Italy\n"
dataset_text += f"Scene: Urban\n\n"
dataset_text += f"Size: {h}x{w} pixels\n"
dataset_text += f"Bands: {bands}\n"
dataset_text += f"Classes: {n_actual_classes}\n"
dataset_text += f"Labeled: {np.sum(gt>0):,} pixels\n\n"
dataset_text += f"Sensor: ROSIS\n"
dataset_text += f"Resolution: 1.3m"
ax9.text(0.5, 0.5, dataset_text, fontsize=11, verticalalignment='center',
         horizontalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

# Save visualization
output_dir = Path('../results/pavia_university')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'PAVIA_COMPLETE.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[SUCCESS] Visualization saved to: {output_path}")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVING RESULTS")
print("="*80)

results_file = output_dir / 'classification_results.txt'
with open(results_file, 'w') as f:
    f.write("PAVIA UNIVERSITY - COMPLETE CLASSIFICATION RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("PIXEL-WISE BASELINE:\n")
    f.write("-"*40 + "\n")
    f.write(f"Overall Accuracy:  {oa_pw*100:6.2f}%\n")
    f.write(f"Average Accuracy:  {aa_pw*100:6.2f}%\n")
    f.write(f"Kappa Coefficient: {kappa_pw:7.4f}\n\n")

    f.write("SPATIAL-SPECTRAL (7x7 patches):\n")
    f.write("-"*40 + "\n")
    f.write(f"Overall Accuracy:  {oa_ss*100:6.2f}%\n")
    f.write(f"Average Accuracy:  {aa_ss*100:6.2f}%\n")
    f.write(f"Kappa Coefficient: {kappa_ss:7.4f}\n\n")

    f.write("IMPROVEMENT:\n")
    f.write("-"*40 + "\n")
    f.write(f"OA Improvement: +{oa_ss*100 - oa_pw*100:.2f}%\n")
    f.write(f"AA Improvement: +{aa_ss*100 - aa_pw*100:.2f}%\n\n")

    f.write("PER-CLASS ACCURACY (Spatial-Spectral):\n")
    f.write("-"*40 + "\n")
    for i, name in enumerate(class_names):
        f.write(f"  {actual_classes[i]:2d}. {name:20s}: {class_acc_ss[i]*100:5.2f}%\n")

print(f"[SUCCESS] Results saved to: {results_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nSUMMARY:")
print(f"  Pixel-wise baseline:      {oa_pw*100:.2f}% OA")
print(f"  Spatial-spectral:         {oa_ss*100:.2f}% OA")
print(f"  Improvement:              +{oa_ss*100 - oa_pw*100:.2f}%")
print(f"\nFiles created:")
print(f"  - {output_path}")
print(f"  - {results_file}")
print("\n" + "="*80)
