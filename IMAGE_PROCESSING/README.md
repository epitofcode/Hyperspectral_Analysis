# Hyperspectral Image Preprocessing Techniques

This folder contains implementations of 5 common hyperspectral preprocessing techniques that we **didn't use** in the main pipeline, but are worth understanding and experimenting with.

## 📁 Files

1. **`bad_band_removal.py`** - Remove noisy or water absorption bands
2. **`spectral_smoothing.py`** - Savitzky-Golay filter for noise reduction
3. **`mnf_transform.py`** - Minimum Noise Fraction (alternative to PCA)
4. **`atmospheric_correction.py`** - Atmospheric effects removal (simplified)
5. **`spectral_unmixing.py`** - Decompose mixed pixels into endmembers

---

## 🎯 Why These Techniques?

Our main pipeline achieved **90.74% accuracy (Indian Pines)** and **98.73% (Pavia Center)** without these techniques. But they're commonly used in hyperspectral image processing literature, so it's valuable to understand them.

### Expected Impact:

| Technique | Expected Gain | Complexity | Worth It? |
|-----------|---------------|------------|-----------|
| Bad band removal | +0.5% | Low | 🟡 Marginal (PCA handles this) |
| Spectral smoothing | +0.5-1% | Low | 🟡 Worth trying |
| MNF transform | +1-2% | Medium | 🟡 Research purposes |
| Atmospheric correction | +2-5% | Very High | 🟡 If you have noisy data |
| Spectral unmixing | +1-2% | Very High | 🟡 For mixed pixels |

---

## 🚀 How to Run

### Run Individual Scripts:

```bash
cd img_process

# 1. Bad Band Removal
python bad_band_removal.py

# 2. Spectral Smoothing
python spectral_smoothing.py

# 3. MNF Transform
python mnf_transform.py

# 4. Atmospheric Correction
python atmospheric_correction.py

# 5. Spectral Unmixing
python spectral_unmixing.py
```

Each script will:
- Load Indian Pines dataset
- Apply the preprocessing technique
- Show statistics and visualizations
- Save results

---

## 📚 Technique Details

### 1. Bad Band Removal

**What it does:**
- Removes spectral bands with low Signal-to-Noise Ratio (SNR)
- Removes water absorption bands (1400nm, 1900nm regions)

**Methods:**
- **SNR-based**: Calculate SNR per band, remove low SNR bands
- **Water absorption**: Remove known absorption regions
- **Combined**: Apply both methods

**Example output:**
```
Original bands: 200
After SNR removal: 180 (90%)
After water removal: 160 (80%)
After combined: 145 (72.5%)
```

**Why we don't need it:**
- PCA already handles this by assigning low weights to noisy bands
- They end up in PC51-200 which we discard
- **Verdict**: Already effectively done via PCA

---

### 2. Spectral Smoothing (Savitzky-Golay Filter)

**What it does:**
- Smooths spectral curves to reduce high-frequency noise
- Preserves spectral shape better than moving average

**Parameters:**
- `window_length`: Size of smoothing window (default: 11)
- `polyorder`: Polynomial order for fitting (default: 2)

**Example:**
```
Before: Noisy spectrum with fluctuations
After: Smooth curve following the trend
Mean noise removed: 0.12 reflectance units
```

**Pros:**
- Easy to implement
- Preserves spectral features
- +0.5-1% potential accuracy gain

**Cons:**
- May remove subtle spectral features
- Requires parameter tuning

---

### 3. Minimum Noise Fraction (MNF)

**What it does:**
- Alternative to PCA specifically designed for hyperspectral data
- Orders components by Signal-to-Noise Ratio instead of variance
- Two-step process:
  1. Noise whitening (decorrelate noise)
  2. PCA on whitened data

**Comparison with PCA:**

| Aspect | PCA | MNF |
|--------|-----|-----|
| **Ordering** | Variance | SNR |
| **Best for** | Clean data | Noisy data |
| **Complexity** | Low | Medium |
| **Accuracy gain** | Baseline | +1-2% |
| **Speed** | Fast | Slower |

**When to use:**
- Very noisy datasets
- When you need noise-robust features
- Research comparisons

**Limitation:**
- Requires noise covariance estimation
- More computationally expensive

---

### 4. Atmospheric Correction

**What it does:**
- Removes atmospheric effects:
  - Path radiance (scattered light)
  - Gas absorption (water vapor, O2, CO2)
  - Illumination variations

**Methods implemented (simplified):**

1. **Dark Object Subtraction (DOS)**
   - Assumes darkest pixels should have ~0 reflectance
   - Subtracts minimum value per band

2. **Flat Field Correction**
   - Normalizes by a reference bright spectrum
   - Removes illumination variations

3. **Gain/Offset Correction**
   - Simple per-band normalization

**⚠️ Important Notes:**
- These are **simplified** educational implementations
- Real atmospheric correction requires:
  - Atmospheric models (MODTRAN, FLAASH)
  - Sensor calibration parameters
  - Illumination/viewing geometry
  - Atmospheric conditions

**When needed:**
- Airborne/satellite data with atmospheric effects
- Multi-temporal studies (different atmospheric conditions)
- Absolute reflectance required

**For our case:**
- Indian Pines is pre-calibrated benchmark data
- Atmospheric effects already minimized
- **Verdict**: Not needed for benchmark datasets

---

### 5. Spectral Unmixing

**What it does:**
- Decomposes mixed pixels into pure material spectra (endmembers)
- Each pixel = weighted sum of endmembers
- Weights = abundances (proportions)

**Example:**
```
Mixed pixel = 0.3 × Corn + 0.5 × Soil + 0.2 × Grass
            ↑         ↑         ↑         ↑
         Pixel   Abundance  Endmember  Abundance
```

**Steps:**
1. **Endmember Extraction (VCA)**
   - Find pure material spectra
   - Automated vertex detection

2. **Abundance Estimation (NNLS)**
   - Solve: pixel = Σ(abundance_i × endmember_i)
   - Constraints: abundances ≥ 0, Σ abundances = 1

**Use cases:**
- Sub-pixel classification
- Coarse spatial resolution images
- Mixed land cover analysis

**Limitations:**
- **Very slow**: NNLS for each pixel
- Assumes linear mixing
- Number of endmembers must be specified
- Indian Pines has relatively pure pixels (high spatial resolution)

**Expected gain:**
- +1-2% for coarse resolution images with mixed pixels
- Minimal gain for fine resolution images like Indian Pines

---

## 🎓 Educational Value

Even though we don't use these in our main pipeline, understanding them is valuable because:

1. **Literature Knowledge**: These techniques appear in most hyperspectral papers
2. **Dataset-Specific**: They may be necessary for other datasets
3. **Research Depth**: Understanding trade-offs makes you a better researcher
4. **Problem Solving**: Know what tools exist when facing specific challenges

---

## 📊 When Would You Actually Use These?

### Bad Band Removal
✅ **Use when:**
- Working with raw, uncalibrated data
- You have wavelength information
- Sensor-specific noise patterns

❌ **Skip when:**
- Using benchmark datasets (already clean)
- PCA is part of your pipeline (it handles this)

### Spectral Smoothing
✅ **Use when:**
- Dealing with very noisy spectra
- Sensor has high-frequency noise
- Marginal accuracy gains are worth the effort

❌ **Skip when:**
- Data is already clean
- Risk of removing subtle features

### MNF Transform
✅ **Use when:**
- Comparing with literature (PCA vs MNF studies)
- Data is extremely noisy
- Need maximum SNR in components

❌ **Skip when:**
- Computational speed matters
- PCA already gives good results
- Data quality is decent

### Atmospheric Correction
✅ **Use when:**
- Working with raw airborne/satellite data
- Multi-temporal studies
- Absolute reflectance needed

❌ **Skip when:**
- Using benchmark datasets (pre-corrected)
- Only need relative measurements
- Don't have atmospheric parameters

### Spectral Unmixing
✅ **Use when:**
- Coarse spatial resolution (mixed pixels)
- Sub-pixel classification needed
- Abundance estimation required

❌ **Skip when:**
- High spatial resolution (pure pixels)
- Classification accuracy is priority
- Computational time is limited

---

## 🔬 Experiment Suggestions

Want to test these techniques? Try:

1. **Run baseline (no preprocessing):**
   ```bash
   cd ../code
   python spatial_spectral_pipeline.py
   # Note accuracy: 90.74%
   ```

2. **Run with spectral smoothing:**
   - Apply smoothing before PCA
   - Compare accuracy

3. **Run with MNF instead of PCA:**
   - Replace PCA with MNF in main pipeline
   - Compare training time and accuracy

4. **Run with bad band removal:**
   - Remove bands first, then apply PCA
   - Check if accuracy improves

5. **Compare results:**
   - Document accuracy changes
   - Note computational time differences
   - Analyze trade-offs

---

## 📖 Further Reading

**Bad Band Removal:**
- Gao et al., "Atmospheric correction algorithms for hyperspectral remote sensing data of land and ocean", RSE, 2009

**Spectral Smoothing:**
- Savitzky & Golay, "Smoothing and Differentiation of Data by Simplified Least Squares Procedures", Analytical Chemistry, 1964

**MNF Transform:**
- Green et al., "A transformation for ordering multispectral data in terms of image quality", IEEE TGRS, 1988
- Lee et al., "Enhancement of high spectral resolution remote-sensing data by a noise-adjusted principal components transform", IEEE TGRS, 1990

**Atmospheric Correction:**
- Vermote et al., "Second Simulation of the Satellite Signal in the Solar Spectrum (6S)", IEEE TGRS, 1997

**Spectral Unmixing:**
- Nascimento & Dias, "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data", IEEE TGRS, 2005
- Bioucas-Dias et al., "Hyperspectral Unmixing Overview: Geometrical, Statistical, and Sparse Regression-Based Approaches", IEEE JSTSP, 2012

---

## 🎯 Summary

**Bottom Line:**
- Our main pipeline (PCA + Spatial-spectral + SVM) achieves 90-98% accuracy
- These preprocessing techniques offer +1-3% potential gain
- They increase complexity significantly
- Best for specific scenarios, not general improvement

**Recommendation:**
- Understand these techniques (educational value)
- Use only when specifically needed
- Don't overcomplicate your pipeline
- Focus on data quality > preprocessing tricks

---

**Questions? Check the main wiki.md for complete pipeline documentation!**
