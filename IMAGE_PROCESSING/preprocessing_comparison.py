"""
Preprocessing Techniques Comparison - Complete Pipeline

Tests all 5 preprocessing techniques and compares results with baseline.
Includes comprehensive visualizations.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score
import time

sys.path.append(str(Path(__file__).parent.parent / "code"))
from image_utils import load_hyperspectral_mat, load_ground_truth, select_rgb_bands

# Import preprocessing methods
from bad_band_removal import apply_bad_band_removal
from spectral_smoothing import apply_spectral_smoothing
from mnf_transform import apply_mnf
from atmospheric_correction import dark_object_subtraction

# Configuration
DATASET = "indian_pines"
IMAGE_PATH = f"../data/{DATASET}/{DATASET}_image.mat"
GT_PATH = f"../data/{DATASET}/{DATASET}_gt.mat"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_PCA_COMPONENTS = 50
PATCH_SIZE = 11
TRAIN_RATIO = 0.30
RANDOM_STATE = 42


def extract_spatial_spectral_features(image, gt, patch_size=11):
    """Extract spatial-spectral patches."""
    height, width, bands = image.shape
    half_patch = patch_size // 2

    # Pad image
    padded_image = np.pad(
        image,
        ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
        mode='symmetric'
    )

    # Get labeled pixels
    labeled_mask = gt > 0
    labeled_positions = np.argwhere(labeled_mask)

    features = []
    labels = []

    for pos in labeled_positions:
        row, col = pos
        row_padded = row + half_patch
        col_padded = col + half_patch

        patch = padded_image[
            row_padded - half_patch : row_padded + half_patch + 1,
            col_padded - half_patch : col_padded + half_patch + 1,
            :
        ]

        features.append(patch.flatten())
        labels.append(gt[row, col])

    return np.array(features), np.array(labels), labeled_positions


def train_and_evaluate(X_all, y_all, train_ratio=0.30):
    """Train SVM and evaluate."""
    np.random.seed(RANDOM_STATE)

    # Split
    classes = np.unique(y_all)
    train_indices = []
    test_indices = []

    for class_id in classes:
        class_mask = y_all == class_id
        class_indices = np.where(class_mask)[0]
        n_class = len(class_indices)
        n_train = max(5, int(n_class * train_ratio))

        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:n_train])
        test_indices.extend(class_indices[n_train:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE)
    svm.fit(X_train_scaled, y_train)

    # Predict
    y_pred = svm.predict(X_test_scaled)

    # Metrics
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

    return {
        'oa': oa,
        'aa': aa,
        'kappa': kappa,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


def run_baseline(image, gt):
    """Baseline: PCA only (no additional preprocessing)."""
    print("\n" + "="*80)
    print("BASELINE: PCA ONLY (NO ADDITIONAL PREPROCESSING)")
    print("="*80)

    start_time = time.time()

    # PCA
    height, width, bands = image.shape
    image_2d = image.reshape(-1, bands)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    reduced_2d = pca.fit_transform(image_2d)
    processed_image = reduced_2d.reshape(height, width, N_PCA_COMPONENTS)

    print(f"PCA variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Extract features
    X_all, y_all, positions = extract_spatial_spectral_features(processed_image, gt, PATCH_SIZE)
    print(f"Features extracted: {X_all.shape}")

    # Train and evaluate
    results = train_and_evaluate(X_all, y_all, TRAIN_RATIO)

    elapsed = time.time() - start_time
    results['time'] = elapsed

    print(f"\nResults:")
    print(f"  OA: {results['oa']*100:.2f}%")
    print(f"  AA: {results['aa']*100:.2f}%")
    print(f"  Kappa: {results['kappa']:.4f}")
    print(f"  Time: {elapsed:.2f}s")

    return results, processed_image


def run_with_bad_band_removal(image, gt):
    """Test with bad band removal."""
    print("\n" + "="*80)
    print("METHOD 1: BAD BAND REMOVAL + PCA")
    print("="*80)

    start_time = time.time()

    # Apply bad band removal
    cleaned_image, good_bands = apply_bad_band_removal(image, method='snr', snr_threshold=5)
    print(f"Bands: {image.shape[2]} -> {cleaned_image.shape[2]}")

    # PCA
    height, width, bands = cleaned_image.shape
    image_2d = cleaned_image.reshape(-1, bands)
    pca = PCA(n_components=min(N_PCA_COMPONENTS, bands), random_state=RANDOM_STATE)
    reduced_2d = pca.fit_transform(image_2d)
    processed_image = reduced_2d.reshape(height, width, min(N_PCA_COMPONENTS, bands))

    print(f"PCA variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Extract features
    X_all, y_all, positions = extract_spatial_spectral_features(processed_image, gt, PATCH_SIZE)
    print(f"Features extracted: {X_all.shape}")

    # Train and evaluate
    results = train_and_evaluate(X_all, y_all, TRAIN_RATIO)

    elapsed = time.time() - start_time
    results['time'] = elapsed

    print(f"\nResults:")
    print(f"  OA: {results['oa']*100:.2f}%")
    print(f"  AA: {results['aa']*100:.2f}%")
    print(f"  Kappa: {results['kappa']:.4f}")
    print(f"  Time: {elapsed:.2f}s")

    return results, processed_image


def run_with_spectral_smoothing(image, gt):
    """Test with spectral smoothing."""
    print("\n" + "="*80)
    print("METHOD 2: SPECTRAL SMOOTHING + PCA")
    print("="*80)

    start_time = time.time()

    # Apply smoothing
    smoothed_image = apply_spectral_smoothing(image, window_length=11, polyorder=2)
    print(f"Smoothing applied")

    # PCA
    height, width, bands = smoothed_image.shape
    image_2d = smoothed_image.reshape(-1, bands)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    reduced_2d = pca.fit_transform(image_2d)
    processed_image = reduced_2d.reshape(height, width, N_PCA_COMPONENTS)

    print(f"PCA variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Extract features
    X_all, y_all, positions = extract_spatial_spectral_features(processed_image, gt, PATCH_SIZE)
    print(f"Features extracted: {X_all.shape}")

    # Train and evaluate
    results = train_and_evaluate(X_all, y_all, TRAIN_RATIO)

    elapsed = time.time() - start_time
    results['time'] = elapsed

    print(f"\nResults:")
    print(f"  OA: {results['oa']*100:.2f}%")
    print(f"  AA: {results['aa']*100:.2f}%")
    print(f"  Kappa: {results['kappa']:.4f}")
    print(f"  Time: {elapsed:.2f}s")

    return results, processed_image


def run_with_mnf(image, gt):
    """Test with MNF instead of PCA."""
    print("\n" + "="*80)
    print("METHOD 3: MNF TRANSFORM (INSTEAD OF PCA)")
    print("="*80)

    start_time = time.time()

    # Apply MNF
    mnf_image, mnf_model, snr = apply_mnf(image, n_components=N_PCA_COMPONENTS)
    print(f"MNF applied: {mnf_image.shape}")

    # Extract features (MNF already reduced)
    X_all, y_all, positions = extract_spatial_spectral_features(mnf_image, gt, PATCH_SIZE)
    print(f"Features extracted: {X_all.shape}")

    # Train and evaluate
    results = train_and_evaluate(X_all, y_all, TRAIN_RATIO)

    elapsed = time.time() - start_time
    results['time'] = elapsed

    print(f"\nResults:")
    print(f"  OA: {results['oa']*100:.2f}%")
    print(f"  AA: {results['aa']*100:.2f}%")
    print(f"  Kappa: {results['kappa']:.4f}")
    print(f"  Time: {elapsed:.2f}s")

    return results, mnf_image


def run_with_atmospheric_correction(image, gt):
    """Test with atmospheric correction."""
    print("\n" + "="*80)
    print("METHOD 4: ATMOSPHERIC CORRECTION + PCA")
    print("="*80)

    start_time = time.time()

    # Apply atmospheric correction
    corrected_image = dark_object_subtraction(image, percentile=1)
    print(f"Atmospheric correction applied")

    # PCA
    height, width, bands = corrected_image.shape
    image_2d = corrected_image.reshape(-1, bands)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    reduced_2d = pca.fit_transform(image_2d)
    processed_image = reduced_2d.reshape(height, width, N_PCA_COMPONENTS)

    print(f"PCA variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Extract features
    X_all, y_all, positions = extract_spatial_spectral_features(processed_image, gt, PATCH_SIZE)
    print(f"Features extracted: {X_all.shape}")

    # Train and evaluate
    results = train_and_evaluate(X_all, y_all, TRAIN_RATIO)

    elapsed = time.time() - start_time
    results['time'] = elapsed

    print(f"\nResults:")
    print(f"  OA: {results['oa']*100:.2f}%")
    print(f"  AA: {results['aa']*100:.2f}%")
    print(f"  Kappa: {results['kappa']:.4f}")
    print(f"  Time: {elapsed:.2f}s")

    return results, processed_image


def visualize_comparison(results_dict, original_rgb):
    """Create comprehensive comparison visualization."""
    methods = list(results_dict.keys())
    n_methods = len(methods)

    # Extract metrics
    oa_values = [results_dict[m]['oa'] * 100 for m in methods]
    aa_values = [results_dict[m]['aa'] * 100 for m in methods]
    kappa_values = [results_dict[m]['kappa'] for m in methods]
    time_values = [results_dict[m]['time'] for m in methods]

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Preprocessing Techniques Comparison - Indian Pines Dataset',
                 fontsize=18, fontweight='bold')

    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Overall Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['green' if oa == max(oa_values) else 'steelblue' for oa in oa_values]
    bars = ax1.bar(range(n_methods), oa_values, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_xticks(range(n_methods))
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Overall Accuracy (%)', fontsize=11)
    ax1.set_title('Overall Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=2, label='90% threshold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([min(oa_values)-2, max(oa_values)+2])

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, oa_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Average Accuracy comparison
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['green' if aa == max(aa_values) else 'orange' for aa in aa_values]
    bars = ax2.bar(range(n_methods), aa_values, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_xticks(range(n_methods))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Average Accuracy (%)', fontsize=11)
    ax2.set_title('Average Accuracy Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([min(aa_values)-2, max(aa_values)+2])

    for i, (bar, val) in enumerate(zip(bars, aa_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Kappa comparison
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['green' if k == max(kappa_values) else 'purple' for k in kappa_values]
    bars = ax3.bar(range(n_methods), kappa_values, color=colors, edgecolor='black', alpha=0.8)
    ax3.set_xticks(range(n_methods))
    ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Kappa Coefficient', fontsize=11)
    ax3.set_title('Kappa Coefficient Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([min(kappa_values)-0.05, max(kappa_values)+0.05])

    for i, (bar, val) in enumerate(zip(bars, kappa_values)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Processing time comparison
    ax4 = fig.add_subplot(gs[1, 0])
    colors = ['red' if t == max(time_values) else 'teal' for t in time_values]
    bars = ax4.bar(range(n_methods), time_values, color=colors, edgecolor='black', alpha=0.8)
    ax4.set_xticks(range(n_methods))
    ax4.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Processing Time (seconds)', fontsize=11)
    ax4.set_title('Computational Time Comparison', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, time_values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Gain/Loss from baseline
    baseline_oa = oa_values[0]
    oa_gains = [oa - baseline_oa for oa in oa_values]

    ax5 = fig.add_subplot(gs[1, 1])
    colors = ['green' if g > 0 else 'red' if g < 0 else 'gray' for g in oa_gains]
    bars = ax5.bar(range(n_methods), oa_gains, color=colors, edgecolor='black', alpha=0.8)
    ax5.set_xticks(range(n_methods))
    ax5.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('OA Gain/Loss vs Baseline (%)', fontsize=11)
    ax5.set_title('Accuracy Gain/Loss from Baseline', fontsize=12, fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.grid(axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, oa_gains)):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.2f}%', ha='center', va=va, fontsize=9, fontweight='bold')

    # Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_text = "SUMMARY\n"
    summary_text += "─"*45 + "\n\n"
    summary_text += f"Best OA:    {methods[oa_values.index(max(oa_values))]}\n"
    summary_text += f"            {max(oa_values):.2f}%\n\n"
    summary_text += f"Best AA:    {methods[aa_values.index(max(aa_values))]}\n"
    summary_text += f"            {max(aa_values):.2f}%\n\n"
    summary_text += f"Best Kappa: {methods[kappa_values.index(max(kappa_values))]}\n"
    summary_text += f"            {max(kappa_values):.4f}\n\n"
    summary_text += f"Fastest:    {methods[time_values.index(min(time_values))]}\n"
    summary_text += f"            {min(time_values):.1f}s\n\n"
    summary_text += "─"*45 + "\n"
    summary_text += "VERDICT:\n"

    best_method = methods[oa_values.index(max(oa_values))]
    if best_method == "Baseline":
        summary_text += "Baseline (PCA only) is\nbest! Additional\npreprocessing provides\nno gain."
    else:
        gain = max(oa_values) - baseline_oa
        summary_text += f"{best_method}\nwins with +{gain:.2f}%\naccuracy gain."

    ax6.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Original RGB
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.imshow(original_rgb)
    ax7.set_title('Original RGB Image', fontsize=12, fontweight='bold')
    ax7.axis('off')

    # Detailed metrics table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')

    table_data = []
    table_data.append(['Method', 'OA (%)', 'AA (%)', 'Kappa', 'Time (s)', 'Gain (%)'])

    for i, method in enumerate(methods):
        table_data.append([
            method,
            f"{oa_values[i]:.2f}",
            f"{aa_values[i]:.2f}",
            f"{kappa_values[i]:.4f}",
            f"{time_values[i]:.1f}",
            f"{oa_gains[i]:+.2f}"
        ])

    table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best values
    best_oa_row = oa_values.index(max(oa_values)) + 1
    best_aa_row = aa_values.index(max(aa_values)) + 1
    best_kappa_row = kappa_values.index(max(kappa_values)) + 1

    table[(best_oa_row, 1)].set_facecolor('#90EE90')
    table[(best_aa_row, 2)].set_facecolor('#90EE90')
    table[(best_kappa_row, 3)].set_facecolor('#90EE90')

    ax8.set_title('Detailed Metrics Comparison', fontsize=12, fontweight='bold', pad=20)

    plt.savefig(RESULTS_DIR / 'preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {RESULTS_DIR / 'preprocessing_comparison.png'}")

    return fig


def main():
    """Main comparison pipeline."""
    print("="*80)
    print("PREPROCESSING TECHNIQUES COMPARISON")
    print("="*80)
    print(f"\nDataset: {DATASET}")
    print(f"Configuration: {N_PCA_COMPONENTS} PCA | {PATCH_SIZE}×{PATCH_SIZE} patches | {TRAIN_RATIO*100:.0f}% train")
    print("\nTesting 5 preprocessing methods...")

    # Load data
    print("\nLoading data...")
    image = load_hyperspectral_mat(IMAGE_PATH)
    gt = load_ground_truth(GT_PATH)
    rgb = select_rgb_bands(image)

    print(f"Image shape: {image.shape}")
    print(f"Ground truth shape: {gt.shape}")

    # Run all experiments
    results = {}

    # Baseline
    results['Baseline'], _ = run_baseline(image, gt)

    # Bad band removal
    results['Bad Band\nRemoval'], _ = run_with_bad_band_removal(image, gt)

    # Spectral smoothing
    results['Spectral\nSmoothing'], _ = run_with_spectral_smoothing(image, gt)

    # MNF
    results['MNF\nTransform'], _ = run_with_mnf(image, gt)

    # Atmospheric correction
    results['Atmospheric\nCorrection'], _ = run_with_atmospheric_correction(image, gt)

    # Visualize comparison
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATION")
    print("="*80)
    visualize_comparison(results, rgb)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    methods = list(results.keys())
    for method in methods:
        r = results[method]
        print(f"\n{method.replace(chr(10), ' ')}:")
        print(f"  OA: {r['oa']*100:.2f}% | AA: {r['aa']*100:.2f}% | Kappa: {r['kappa']:.4f} | Time: {r['time']:.1f}s")

    # Determine winner
    oa_values = [results[m]['oa'] * 100 for m in methods]
    best_idx = oa_values.index(max(oa_values))
    best_method = methods[best_idx]

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if best_method == "Baseline":
        print("\n✅ BASELINE (PCA ONLY) IS BEST!")
        print("   Additional preprocessing provides no benefit.")
        print("   This confirms our original pipeline design was optimal.")
    else:
        baseline_oa = results['Baseline']['oa'] * 100
        best_oa = max(oa_values)
        gain = best_oa - baseline_oa
        print(f"\n🏆 {best_method.replace(chr(10), ' ')} WINS!")
        print(f"   Accuracy gain: +{gain:.2f}%")
        print(f"   Final OA: {best_oa:.2f}%")

    print("\nAll results saved to: img_process/results/")
    print("="*80)


if __name__ == "__main__":
    main()
