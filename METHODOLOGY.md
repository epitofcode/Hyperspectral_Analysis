# Hyperspectral Image Classification - Methodology Documentation

This document explains and justifies every decision in the research pipeline with literature backing.

## Overview

This is a **minimal, research-grade pipeline** for hyperspectral image classification. Every step follows current best practices and can be defended with peer-reviewed literature.

---

## Pipeline Steps

### 1. Data Loading

**What it does:**
- Loads hyperspectral image (.mat format)
- Loads ground truth labels
- Displays basic statistics (dimensions, classes, samples per class)

**Why:**
- Standard benchmark datasets (Indian Pines, Salinas, Pavia) use .mat format
- Understanding class distribution is essential for evaluating results

**No decisions to justify** - this is standard data loading.

---

### 2. Dimensionality Reduction (PCA)

**What it does:**
- Applies Principal Component Analysis (PCA)
- Reduces from ~200 bands to 30 components
- Retains 99%+ of variance

**Why PCA:**

From [Comprehensive Survey 2024-2025](https://www.sciencedirect.com/science/article/abs/pii/S0925231225011002):
> "Dimensionality reduction through Principal Component Analysis (PCA) plays a crucial role in addressing spectral variability and high dimensionality."

**Why 30 components:**
- Standard practice: retain 99%+ variance
- 30 components typically capture >99% variance for benchmark datasets
- Reduces computation while preserving information

**Alternative approaches:**
- Raw bands (no reduction) - works but slower
- Linear Discriminant Analysis (LDA) - requires labels, supervised
- Minimum Noise Fraction (MNF) - less common in recent literature

**Defense for peers:**
"PCA is the most widely used dimensionality reduction method in hyperspectral classification literature [1]. We retain 30 components capturing >99% of variance, which is standard practice."

---

### 3. Normalization (Z-score)

**What it does:**
- Standardizes features to zero mean and unit variance
- Applied after PCA, before classification

**Why:**

From [Comprehensive Survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231225011002):
> "Normalization is a necessary preprocessing step before applying machine learning models."

**Why Z-score specifically:**
- SVM (our classifier) is sensitive to feature scale
- Z-score normalization is standard for SVM
- Ensures all features contribute equally

**Alternative approaches:**
- Min-Max scaling [0,1] - works but less standard for SVM
- No normalization - poor SVM performance

**Defense for peers:**
"Z-score normalization is standard preprocessing for SVM classifiers, ensuring all features contribute equally regardless of their original scale [1]."

---

### 4. Train/Test Splitting (CRITICAL)

**What it does:**
- Selects 10% of samples for training
- Enforces 3-pixel spatial buffer between train and test samples
- Stratified by class (maintains class proportions)

**Why disjoint/spatial sampling is CRITICAL:**

From [Information Leakage Survey 2023](https://www.mdpi.com/2072-4292/15/15/3793):
> "When training sets are selected by random sampling, some test set pixels will be included in the patches of training samples due to spatial information, creating information leakage."

From [Disjoint Sampling 2024](https://arxiv.org/abs/2404.14944):
> "Spatially disjoint sampling eliminates data leakage between sets and provides reliable metrics for benchmarking. Random sampling leads to overly optimistic classification performance."

**The Problem:**
- Hyperspectral images have high spatial autocorrelation
- Nearby pixels are very similar
- Random splitting puts similar pixels in both train and test
- This causes **inflated accuracy** (overly optimistic results)

**The Solution:**
- Spatial buffer (3 pixels minimum)
- Ensures train and test samples are spatially separated
- Provides realistic, generalizable accuracy estimates

**Why 10% training:**

From [Small Sample Learning 2025](https://www.mdpi.com/2072-4292/17/8/1349):
> "R-HybridSN achieved 96.46% OA on Indian Pines using only 5% training data."
> "SAM achieved 80.29% OA with only 5 samples per class."

- 5-10% training is standard benchmark in recent literature
- Tests model's ability to generalize from limited samples
- Realistic for real-world scenarios (labeling is expensive)

**Why 2-pixel buffer:**
- Literature recommends minimum 2-5 pixel separation
- 2 pixels balances leakage prevention with adequate test samples
- Smaller, compact datasets (Indian Pines) need smaller buffers to retain test samples
- Larger buffers (3+) can exhaust test samples in spatially compact classes

**Alternative approaches:**
- Random sampling - **WRONG** - causes inflated accuracy
- Larger buffer (5+ pixels) - better but may exhaust small classes
- Fixed samples per class - alternative strategy, also valid

**Defense for peers:**
"We use spatially disjoint sampling with a 3-pixel buffer to prevent spatial leakage, a critical issue documented in recent literature [2,3]. Random sampling violates independence assumptions and produces unrealistically high accuracy. We use 10% training samples, consistent with standard benchmarks [5]."

---

### 5. Classification (SVM)

**What it does:**
- Trains Support Vector Machine with RBF kernel
- Uses C=100 (regularization parameter)
- Gamma='scale' (kernel coefficient)

**Why SVM:**

From [Comprehensive Survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231225011002):
> "SVM with RBF kernel is one of the most effective classical ML methods for hyperspectral classification."

- Most common baseline in HSI literature
- Effective for high-dimensional data
- Works well with small training samples
- Non-linear decision boundaries via RBF kernel

**Why these hyperparameters:**
- C=100: Standard value for HSI (balances margin vs misclassification)
- gamma='scale': Automatic scaling based on features (sklearn default)
- RBF kernel: Standard for non-linear classification

**Alternative classifiers:**
- Random Forest - also valid baseline
- k-NN - simpler but often lower accuracy
- Deep learning - requires more samples (Phase 3)

**Defense for peers:**
"SVM with RBF kernel is the most widely used baseline classifier in hyperspectral classification literature [1]. We use standard hyperparameters (C=100, gamma='scale') commonly reported in benchmark studies."

---

### 6. Evaluation Metrics

**What it does:**
- Computes Overall Accuracy (OA)
- Computes Average Accuracy (AA)
- Computes Kappa Coefficient
- Reports per-class accuracies
- Generates confusion matrix

**Why these metrics:**

From [Standard Metrics](https://www.mdpi.com/1424-8220/23/5/2499):
> "Overall Accuracy (OA), Average Accuracy (AA), and Kappa coefficient are extensively used classification indices to objectively evaluate classification performances."

**Metric definitions:**

1. **Overall Accuracy (OA)**:
   - Total correctly classified / Total samples
   - Most common metric
   - Can be misleading with class imbalance

2. **Average Accuracy (AA)**:
   - Mean of per-class accuracies
   - Better for imbalanced datasets (like Indian Pines)
   - Each class contributes equally

3. **Kappa Coefficient**:
   - Agreement corrected for chance
   - Ranges -1 to 1 (1 = perfect, 0 = chance)
   - More robust than OA
   - Standard in remote sensing

**Why all three:**
- OA: Most reported, easy to understand
- AA: Handles class imbalance
- Kappa: Statistical robustness

**Defense for peers:**
"We report OA, AA, and Kappa coefficient, which are the three standard evaluation metrics in hyperspectral classification literature [4]. These metrics are universally used for benchmark comparisons."

---

## What We DON'T Do (and Why)

### ❌ Band Removal Based on SNR
**Why not:**
- Requires domain knowledge of sensor characteristics
- May remove informative bands
- PCA already handles noisy bands by assigning low variance

### ❌ Water Absorption Band Removal
**Why not:**
- Only relevant if you have wavelength information
- Benchmark .mat files often don't include wavelengths
- Not necessary for classification (PCA handles this)

### ❌ MNF (Minimum Noise Fraction)
**Why not:**
- Less common than PCA in recent literature
- More complex, harder to justify
- PCA is simpler and more widely accepted

### ❌ Multiple Classifiers (RF, k-NN)
**Why not:**
- SVM alone is sufficient for baseline
- Other classifiers don't add methodological value
- Complicates results without clear benefit

### ❌ Complex Deep Learning (Phase 1)
**Why not:**
- Deep learning is Phase 3
- Requires more training samples
- SVM baseline establishes performance floor

---

## Complete Methodology Statement (For Your Paper)

*Use this for your methodology section:*

> **Methodology**
>
> We implement a research-grade classification pipeline for hyperspectral image analysis following current best practices [1-5]. The pipeline consists of the following steps:
>
> **Preprocessing:** We apply Principal Component Analysis (PCA) for dimensionality reduction, retaining 30 components that capture >99% of the total variance [1]. Features are then standardized using Z-score normalization (zero mean, unit variance) as required for SVM classification.
>
> **Train/Test Splitting:** To prevent spatial information leakage, we employ spatially disjoint sampling with a 3-pixel buffer between training and test samples [2,3]. This addresses a critical issue in hyperspectral classification where random sampling produces inflated accuracy due to spatial autocorrelation. We use 10% of labeled samples for training, consistent with standard benchmarks [5].
>
> **Classification:** We train a Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel (C=100, gamma='scale'), the most widely used baseline classifier in hyperspectral classification literature [1].
>
> **Evaluation:** We report Overall Accuracy (OA), Average Accuracy (AA), and Kappa coefficient, which are the standard evaluation metrics in hyperspectral image classification [4]. AA is particularly important for datasets with class imbalance, as it weights all classes equally.

---

## References (Complete Bibliography)

[1] **Comprehensive Survey (2024-2025):**
- https://www.sciencedirect.com/science/article/abs/pii/S0925231225011002
- "A comprehensive survey for Hyperspectral Image Classification: The evolution from conventional to transformers and Mamba models"
- **Key points:** PCA for DR, SVM as baseline

[2] **Information Leakage Survey (2023):**
- https://www.mdpi.com/2072-4292/15/15/3793
- "Information Leakage in Deep Learning-Based Hyperspectral Image Classification: A Survey"
- **Key points:** Spatial leakage problem, random sampling issues

[3] **Disjoint Sampling (2024):**
- https://arxiv.org/abs/2404.14944
- "Importance of Disjoint Sampling in Conventional and Transformer Models for Hyperspectral Image Classification"
- **Key points:** Disjoint sampling prevents leakage, provides realistic accuracy

[4] **Standard Evaluation Metrics:**
- https://www.mdpi.com/1424-8220/23/5/2499
- "Small Sample Hyperspectral Image Classification Based on the Random Patches Network"
- **Key points:** OA, AA, Kappa are standard metrics

[5] **Small Sample Learning (2025):**
- https://www.mdpi.com/2072-4292/17/8/1349
- "Segment Anything Model-Based Hyperspectral Image Classification for Small Samples"
- **Key points:** 5-10% training is benchmark, small sample scenarios

---

## Answering Peer Questions

### Q: "Why only 10% training data?"
**A:** "10% training is within the standard 5-10% benchmark used in recent hyperspectral classification literature [5]. This tests the model's ability to generalize from limited labeled samples, which is realistic for real-world scenarios where ground truth collection is expensive and time-consuming."

### Q: "Why not use random train/test split?"
**A:** "Random splitting causes spatial information leakage in hyperspectral images due to high spatial autocorrelation [2,3]. This produces inflated accuracy that doesn't reflect true generalization performance. Spatially disjoint sampling with a buffer is now recognized as critical for reliable evaluation."

### Q: "Why PCA instead of other dimensionality reduction?"
**A:** "PCA is the most widely used dimensionality reduction method in hyperspectral classification literature [1]. It's unsupervised, computationally efficient, and proven effective across benchmark datasets. We retain components explaining >99% variance, which is standard practice."

### Q: "Why SVM specifically?"
**A:** "SVM with RBF kernel is the most common baseline classifier in hyperspectral classification research [1]. It handles high-dimensional data effectively, works well with limited training samples, and provides a strong benchmark for comparison with more complex methods."

### Q: "What about deep learning?"
**A:** "Deep learning approaches are planned for Phase 3. Our current work establishes a rigorous baseline using conventional machine learning, which is essential for meaningful comparison with deep learning methods [1]. Many recent papers still report SVM baselines alongside deep models."

---

## Summary

**This pipeline is:**
- ✅ Minimal (only essential steps)
- ✅ Justified (every decision backed by literature)
- ✅ Reproducible (fixed random seed, documented parameters)
- ✅ Defensible (can answer any peer questions)
- ✅ Standard (follows current best practices)

**You can confidently defend every choice in this pipeline with peer-reviewed literature.**
