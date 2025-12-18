# Spatial-Spectral Analysis Framework

A comprehensive Python framework for hyperspectral image analysis, featuring spatial-spectral processing, dimensionality reduction, and feature extraction capabilities.

## Overview

This framework provides tools for analyzing hyperspectral imagery with focus on:
- Spatial image processing (filtering, edge detection, morphology)
- Spectral analysis (PCA, ICA, NMF, spectral unmixing)
- Visualization and feature extraction
- Support for multiple benchmark datasets

## Project Structure

```
Spatial_Spectral_analysis/
├── code/                           # Core processing modules
│   ├── download_datasets.py        # Dataset acquisition utilities
│   ├── image_utils.py              # Image loading and visualization (SPy-enhanced)
│   ├── spatial_processing.py       # Spatial filtering and transforms
│   ├── spectral_processing.py      # Spectral analysis methods
│   ├── preprocessing.py            # Advanced preprocessing (MNF, normalization)
│   ├── data_utils.py               # Train/test splitting (spatial leakage-aware)
│   ├── ml_classifiers.py           # ML baseline classifiers (SVM, RF, k-NN)
│   ├── evaluation_metrics.py       # Evaluation metrics (OA, AA, Kappa)
│   ├── phase1_complete_workflow.py # Complete Phase 1 pipeline
│   └── example_basic_analysis.py   # Basic example workflow
├── data/                           # Organized hyperspectral datasets
├── results/                        # Analysis output directory
├── Knowledge_base/                 # Reference materials
├── venv/                           # Python virtual environment
└── requirements.txt                # Python dependencies
```

## Setup

### 1. Environment Setup

Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:
- numpy, scipy, pandas (scientific computing)
- scikit-learn, xgboost (machine learning)
- scikit-image, opencv-python (image processing)
- matplotlib, seaborn, plotly (visualization)
- spectral (Spectral Python - hyperspectral tools)
- torch, torchvision (deep learning - Phase 3)

### 3. Dataset Organization

Datasets should be organized in the `data/` directory with the following structure:
```
data/
├── dataset_name/
│   ├── dataset_name_image.mat
│   └── dataset_name_gt.mat
```

## Available Datasets

The framework supports the following benchmark hyperspectral datasets:

| Dataset | Dimensions | Bands | Scene Type |
|---------|-----------|-------|------------|
| Indian Pines | 145×145 | 200 | Agricultural |
| Salinas | 512×217 | 204 | Vegetation |
| Pavia University | 610×340 | 102 | Urban |
| Pavia Center | 1096×715 | 102 | Urban |
| Kennedy Space Center | 512×614 | 176 | Wetlands |
| Houston 2013/2018 | Variable | 144 | Urban |
| HyRANK (Dioni/Loukia) | Variable | Variable | Satellite |

## Usage

### Phase 1: Complete Classification Workflow

Run the complete Phase 1 pipeline with proper preprocessing, train/test splitting, and ML baselines:

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run Phase 1 workflow on Indian Pines
python code/phase1_complete_workflow.py
```

This executes a research-grade pipeline including:
1. Data loading and visualization
2. Advanced preprocessing (MNF transformation, normalization)
3. Spatial leakage-aware train/test splitting (5-pixel buffer)
4. ML baseline training (SVM, Random Forest, k-NN)
5. Evaluation with standard metrics (OA, AA, Kappa)
6. Classification map generation
7. Results saved to `results/indian_pines/`

### Basic Analysis Workflow

```python
import sys
sys.path.append('code')

from image_utils import load_hyperspectral_mat, load_ground_truth
from preprocessing import create_preprocessing_pipeline
from data_utils import create_disjoint_train_test_split
from ml_classifiers import SVMClassifier
from evaluation_metrics import evaluate_classification, print_evaluation_report

# Load dataset
image = load_hyperspectral_mat('data/indian_pines/indian_pines_image.mat')
gt = load_ground_truth('data/indian_pines/indian_pines_gt.mat')

# Preprocessing pipeline
pipeline_results = create_preprocessing_pipeline(
    image,
    apply_mnf=True,
    n_mnf_components=30,
    normalization='zscore'
)
processed_image = pipeline_results['processed_image']

# Proper train/test split (avoids spatial leakage)
X_train, X_test, y_train, y_test, _, _ = create_disjoint_train_test_split(
    processed_image,
    gt,
    train_ratio=0.1,
    spatial_buffer=5
)

# Train classifier
classifier = SVMClassifier(C=100.0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate
results = evaluate_classification(y_test, y_pred)
print_evaluation_report(results)
```

### Running Example Analysis

```bash
python code/example_basic_analysis.py
```

This generates basic visualizations including:
- RGB composites and ground truth
- Spatial filtering results
- Edge detection outputs
- PCA components and variance analysis
- Spectral signatures

## Features

### Advanced Preprocessing (`preprocessing.py`)

**Dimensionality Reduction:**
- Minimum Noise Fraction (MNF) - SNR-based component ordering
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- Non-negative Matrix Factorization (NMF)

**Band Management:**
- Water absorption band identification and removal
- Low SNR band removal
- Band selection by variance and correlation

**Normalization:**
- Z-score normalization (per-band or global)
- Min-Max scaling
- Pixel-wise L2 normalization

### Data Management (`data_utils.py`)

**Spatial Leakage-Aware Splitting:**
- Disjoint train/test split with spatial buffer
- Fixed samples per class splitting
- Stratified sampling
- Class balancing (oversampling/undersampling)
- Train/test split visualization
- Spatial autocorrelation analysis

### ML Baseline Classifiers (`ml_classifiers.py`)

**Classifiers:**
- Support Vector Machine (SVM) with RBF kernel
- Random Forest (RF)
- k-Nearest Neighbors (k-NN)

**Tools:**
- Hyperparameter tuning with grid search
- Full image classification
- Classification map generation

### Evaluation Metrics (`evaluation_metrics.py`)

**Standard Metrics:**
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Kappa Coefficient with interpretation
- Per-class precision, recall, F1-score
- Producer's and User's accuracy
- Confusion matrix

**Reporting:**
- Comprehensive evaluation reports
- Multi-classifier comparison
- Results export to text and JSON

### Spatial Processing Methods (`spatial_processing.py`)

**Filtering:**
- Gaussian, median, bilateral filtering
- Local mean and standard deviation
- Unsharp masking

**Feature Extraction:**
- Edge detection (Sobel, Canny, Prewitt, Roberts)
- Gradient magnitude computation
- Morphological operations (opening, closing, erosion, dilation)
- Texture features (entropy, variance)

**Analysis:**
- Spatial correlation computation
- Spatial downsampling

### Spectral Processing Methods (`spectral_processing.py`)

**Spectral Analysis:**
- Spectral derivatives (1st and 2nd order)
- Savitzky-Golay smoothing
- Continuum removal
- Band selection by variance

**Similarity Measures:**
- Spectral Angle Mapper (SAM)
- Spectral correlation
- Spectral Information Divergence (SID)

**Unmixing:**
- Linear spectral unmixing
- Endmember extraction

### Visualization Tools (`image_utils.py`)

**Standard Visualization:**
- RGB composite generation
- Multi-band visualization
- Ground truth overlay
- Spectral signature plotting
- Statistical analysis plots

**SPy Integration:**
- Interactive hypercube viewer (Spectral Python)
- Enhanced RGB visualization
- Spectral library extraction
- 3D PCA scatter plots
- Classification overlays
- ENVI format I/O

## Example Outputs

The framework generates various analysis outputs:
- Filtered and processed images
- Dimensionality-reduced representations
- Feature maps (edges, texture, gradients)
- Statistical summaries
- Comparative visualizations

Results are saved to the `results/` directory organized by dataset.

## Development Roadmap

### Phase 1: Image Processing & ML Baselines [COMPLETED]
- ✓ Spatial filtering and transforms
- ✓ Spectral analysis and dimensionality reduction
- ✓ Advanced preprocessing (MNF, normalization)
- ✓ Spatial leakage-aware train/test splitting
- ✓ ML baseline classifiers (SVM, RF, k-NN)
- ✓ Standard evaluation metrics (OA, AA, Kappa)
- ✓ Visualization utilities with SPy integration
- ✓ Multi-dataset support
- ✓ Complete workflow pipeline

### Phase 2: Advanced Machine Learning [PLANNED]
- RX Detector and variants for anomaly detection
- Morphological Profiles (MPs) and Extended MPs
- Superpixel segmentation (SLIC)
- Feature selection algorithms
- Ensemble methods
- Cross-validation framework

### Phase 3: Deep Learning [PLANNED]
- 1D-CNN, 2D-CNN, 3D-CNN architectures
- HybridSN (spatial-spectral CNN)
- Autoencoder-based anomaly detection
- Attention mechanisms
- Graph Convolutional Networks (GCN)
- Transformer models (SSFTT, SpectralFormer)
- Transfer learning strategies

## Technical Notes

### Data Format
- Input: MATLAB `.mat` files containing 3D arrays (height × width × bands)
- Ground truth: 2D arrays with integer class labels
- Output: NumPy arrays, matplotlib figures, saved `.npy` files

### Memory Considerations
- Large datasets (Pavia Center) require ~500MB RAM
- Processing operations create temporary copies
- Use spatial downsampling for memory-constrained systems
- Process bands sequentially for very large datasets

### Performance Tips
- Start with smaller datasets (Indian Pines) for testing
- Use PCA to reduce spectral dimensions before heavy processing
- Leverage vectorized NumPy operations
- Consider batch processing for multiple datasets

## Citation

When using this framework or datasets in research, please cite the original dataset sources:

- Indian Pines, Salinas: GIC, Universidad del País Vasco (UPV/EHU)
- Pavia University/Center: Università di Pavia
- Kennedy Space Center: NASA
- Houston: IEEE GRSS Data Fusion Contest
- HyRANK: Satellite hyperspectral benchmark

## License

This framework is provided for research and educational purposes. Individual datasets retain their original licenses.

## Requirements

- Python 3.8+
- See `requirements.txt` for complete dependency list
- Recommended: 8GB RAM minimum, 16GB for large datasets
- Optional: CUDA-capable GPU for future deep learning modules

## Troubleshooting

**Import errors:**
```python
import sys
sys.path.append('code')
```

**Memory issues:**
- Use smaller datasets or spatial subsets
- Reduce number of PCA components
- Process one band at a time

**Visualization issues:**
- Ensure matplotlib backend is properly configured
- For interactive plots, use `plt.show()` at the end of plotting code

## Contact

For questions, issues, or contributions, please refer to the project repository.
