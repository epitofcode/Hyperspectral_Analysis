# Code Directory

## 🎯 Main Scripts - One Per Dataset

Each dataset has **ONE comprehensive script** that does everything:

### 🌾 **`indian_pines.py`** - Complete Indian Pines Pipeline
- **What it does:**
  1. Pixel-wise baseline classification (~75-80% OA)
  2. Spatial-spectral classification with 7×7 patches (~92-95% OA)
  3. Comprehensive visualization (3×3 grid)
  4. Saves results and metrics

- **Run:** `python indian_pines.py`
- **Time:** ~2-3 minutes total
- **Output:**
  - `../results/indian_pines/INDIAN_PINES_COMPLETE.png` (visualization)
  - `../results/indian_pines/classification_results.txt` (metrics)

---

### 🏛️ **`pavia.py`** - Complete Pavia University Pipeline
- **What it does:**
  1. Pixel-wise baseline classification (~93% OA)
  2. Spatial-spectral classification with 7×7 patches (~99% OA)
  3. Comprehensive visualization (3×3 grid)
  4. Saves results and metrics

- **Run:** `python pavia.py`
- **Time:** ~2-3 minutes total
- **Output:**
  - `../results/pavia_university/PAVIA_COMPLETE.png` (visualization)
  - `../results/pavia_university/classification_results.txt` (metrics)

---

## 🔧 Shared Utilities

**`image_utils.py`** - Common functions used by all scripts:
- `load_hyperspectral_mat()` - Load .mat image files
- `load_ground_truth()` - Load .mat ground truth files
- `select_rgb_bands()` - Extract RGB for visualization

---

## 🚀 Quick Start

**Indian Pines:**
```bash
cd code
python indian_pines.py
```

**Pavia University:**
```bash
python pavia.py
```

That's it! Each script does everything automatically.

---

## 📊 What Each Script Shows

The comprehensive output includes:

### Console Output:
```
STEP 1: LOADING DATASET
STEP 2: PCA DIMENSIONALITY REDUCTION
STEP 3: PIXEL-WISE BASELINE CLASSIFICATION
  → Results: OA, AA, Kappa
STEP 4: SPATIAL-SPECTRAL CLASSIFICATION WITH PATCHES
  → Progress tracking
  → Results: OA, AA, Kappa
STEP 5: COMPARISON
  → Improvement percentage
STEP 6: COMPREHENSIVE VISUALIZATION
STEP 7: SAVING RESULTS
```

### Visualization (3×3 Grid):
```
┌─────────────┬─────────────┬─────────────┐
│ Original    │ Ground      │ Predicted   │
│ RGB Image   │ Truth       │ Classes     │
├─────────────┼─────────────┼─────────────┤
│ Classification│ Confusion  │ Per-Class   │
│ Overlay     │ Matrix      │ Accuracy    │
├─────────────┼─────────────┼─────────────┤
│ Class       │ Comparison  │ Dataset     │
│ Legend      │ Metrics     │ Info        │
└─────────────┴─────────────┴─────────────┘
```

### Text File:
- Pixel-wise results
- Spatial-spectral results
- Improvement metrics
- Per-class accuracy breakdown

---

## 📈 Expected Results

| Dataset | Pixel-wise (Step 3) | Spatial-spectral (Step 4) | Improvement |
|---------|---------------------|---------------------------|-------------|
| **Indian Pines** | ~75-80% OA | ~92-95% OA | +15% |
| **Pavia University** | ~93% OA | ~99% OA | +6% |

**Key Insight:** Spatial context (patches) dramatically improves accuracy!

---

## 🎨 Pipeline Steps Explained

### Step 1: Load Data
- Loads hyperspectral image (.mat file)
- Loads ground truth labels
- Extracts RGB for visualization

### Step 2: PCA Reduction
- Reduces spectral bands (200 → 50 or 102 → 50)
- Preserves ~99% variance
- Speeds up processing

### Step 3: Pixel-wise Baseline
- Uses only spectral information
- Fast classification (~2-3 seconds)
- Establishes baseline accuracy

### Step 4: Spatial-Spectral Classification
- Extracts 7×7 spatial patches around each pixel
- Incorporates neighborhood context
- Achieves significantly higher accuracy
- Progress tracking during patch extraction

### Step 5: Comparison
- Shows improvement from adding spatial context
- Quantifies the benefit of patches

### Step 6: Visualization
- Generates comprehensive 3×3 grid
- Publication-quality PNG image

### Step 7: Save Results
- Saves visualization and metrics
- Text file with complete results breakdown

---

## 🛠️ Adding New Datasets

To add a new dataset (e.g., Salinas):

1. **Copy template:**
   ```bash
   cp indian_pines.py salinas.py
   ```

2. **Update paths in salinas.py:**
   ```python
   # Line ~35 and ~36
   image = load_hyperspectral_mat('../data/salinas/salinas_image.mat')
   gt = load_ground_truth('../data/salinas/salinas_gt.mat')

   # Update output directory
   output_dir = Path('../results/salinas')
   ```

3. **Update class names:**
   ```python
   # Find the all_class_names list and update for your dataset
   all_class_names = ['Class1', 'Class2', ...]
   ```

4. **Run:**
   ```bash
   python salinas.py
   ```

---

## 📦 Dependencies

All scripts require:
- numpy
- scikit-learn
- scipy (for .mat files)
- matplotlib (for visualizations)

Install with:
```bash
pip install numpy scikit-learn scipy matplotlib
```

---

## 📝 Optional Scripts

The `code/` folder also contains optional individual scripts if you want to run steps separately:
- `*_test.py` - Just the pixel-wise baseline
- `*_spatial.py` - Just the spatial-spectral classification
- `*_visualize.py` - Just the visualization

But the main scripts (`indian_pines.py`, `pavia.py`) do everything, so these are optional.

---

## 💡 Tips

- **First time?** Run `python indian_pines.py` to see the complete pipeline
- **Want faster results?** The pixel-wise step (Step 3) gives quick results in seconds
- **Want best accuracy?** Wait for the spatial-spectral step (Step 4) - it's worth it!
- **Large dataset taking too long?** Reduce `PATCH_SIZE` from 7 to 5 for faster processing

---

## 🎯 That's It!

Just run one script per dataset and get:
✅ Baseline results
✅ High-accuracy results
✅ Comprehensive visualization
✅ Detailed metrics

**One command. Complete pipeline. All results.**
