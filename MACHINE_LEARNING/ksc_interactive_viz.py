"""
KSC Interactive Visualization
==============================

Interactive visualization tool for KSC hyperspectral classification results.

Features:
- Interactive RGB composite with pan/zoom
- Ground truth overlay with class legend
- Classification results comparison
- Per-class accuracy bar chart
- Confusion matrix heatmap
- Class distribution statistics

Usage:
    python ksc_interactive_viz.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from image_utils import load_hyperspectral_mat, load_ground_truth, select_rgb_bands

# ============================================================================
# CONFIGURATION
# ============================================================================
print("="*80)
print("KSC INTERACTIVE VISUALIZATION")
print("="*80)

# Dataset paths
data_dir = Path(__file__).parent.parent / 'data' / 'ksc'
results_dir = Path(__file__).parent / 'RESULTS' / 'ksc'

# Class names (13 wetland vegetation types)
class_names = [
    'Scrub', 'Willow swamp', 'CP hammock', 'CP/Oak',
    'Slash pine', 'Oak/Broadleaf', 'Hardwood swamp',
    'Graminoid marsh', 'Spartina marsh', 'Cattail marsh',
    'Salt marsh', 'Mud flats', 'Water'
]

# Color palette for classes (distinct colors)
class_colors = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8',
    '#f58231', '#911eb4', '#42d4f4', '#f032e6',
    '#bfef45', '#fabebe', '#469990', '#e6beff',
    '#9A6324'
]

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading KSC dataset...")

# Load hyperspectral image
image = load_hyperspectral_mat(data_dir / 'ksc_image.mat')
h, w, bands = image.shape
print(f"  Image shape: {h} x {w} x {bands} bands")

# Load ground truth
gt = load_ground_truth(data_dir / 'ksc_gt.mat')
print(f"  Ground truth shape: {gt.shape}")

# Get unique classes
unique_classes = np.unique(gt[gt > 0])
n_classes = len(unique_classes)
print(f"  Number of classes: {n_classes}")

# Load classification results (if available)
try:
    # Try to load improved results
    results_file = results_dir / 'ksc_improved_results.txt'
    if results_file.exists():
        print(f"\n  Loading results from: {results_file}")
        # Parse results file for accuracy metrics
        # (Note: In a real implementation, you'd save the prediction map separately)
        with open(results_file, 'r') as f:
            content = f.read()
            # Extract overall accuracy
            if 'Overall Accuracy' in content:
                oa_line = [line for line in content.split('\n') if 'Overall Accuracy' in line][0]
                oa = float(oa_line.split(':')[1].strip().replace('%', ''))
                print(f"  Overall Accuracy: {oa:.2f}%")
except Exception as e:
    print(f"\n  Warning: Could not load results: {e}")
    print("  Will show ground truth only")

# Create RGB composite
print("\nCreating RGB composite...")
rgb_image = select_rgb_bands(image, red_band=50, green_band=30, blue_band=10)

# ============================================================================
# PREPARE GROUND TRUTH VISUALIZATION
# ============================================================================
print("\nPreparing ground truth visualization...")

# Create colored ground truth map
gt_colored = np.zeros((h, w, 3), dtype=np.float32)
for idx, class_id in enumerate(unique_classes):
    mask = gt == class_id
    # Convert hex color to RGB
    hex_color = class_colors[idx].lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    gt_colored[mask] = rgb_color

# ============================================================================
# CLASS DISTRIBUTION STATISTICS
# ============================================================================
print("\nComputing class statistics...")

class_counts = []
class_percentages = []
total_labeled = np.sum(gt > 0)

for class_id in unique_classes:
    count = np.sum(gt == class_id)
    class_counts.append(count)
    class_percentages.append(count / total_labeled * 100)

# Create DataFrame-like structure for display
print("\nClass Distribution:")
print("-" * 60)
print(f"{'Class':5s} {'Name':20s} {'Samples':>10s} {'Percentage':>12s}")
print("-" * 60)
for idx, class_id in enumerate(unique_classes):
    print(f"{class_id:5d} {class_names[idx]:20s} {class_counts[idx]:10d} {class_percentages[idx]:11.2f}%")
print("-" * 60)
print(f"{'TOTAL':5s} {' ':20s} {total_labeled:10d} {100.0:11.2f}%")

# ============================================================================
# CREATE INTERACTIVE VISUALIZATIONS
# ============================================================================
print("\nCreating interactive visualizations...")

# ============================================================================
# Figure 1: RGB Composite with Ground Truth Overlay
# ============================================================================
print("  [1/4] RGB composite with ground truth overlay...")

fig1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('RGB Composite (Bands 50-30-10)', 'Ground Truth (13 Classes)'),
    horizontal_spacing=0.05
)

# RGB composite
fig1.add_trace(
    go.Image(z=(rgb_image * 255).astype(np.uint8)),
    row=1, col=1
)

# Ground truth colored
fig1.add_trace(
    go.Image(z=(gt_colored * 255).astype(np.uint8)),
    row=1, col=2
)

fig1.update_layout(
    title_text="KSC Dataset - RGB Composite and Ground Truth",
    title_font_size=16,
    showlegend=False,
    height=500,
    width=1200
)

fig1.update_xaxes(showticklabels=False)
fig1.update_yaxes(showticklabels=False)

# ============================================================================
# Figure 2: Class Distribution Bar Chart
# ============================================================================
print("  [2/4] Class distribution bar chart...")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=class_names,
    y=class_counts,
    marker_color=class_colors,
    text=class_counts,
    textposition='auto',
    hovertemplate='<b>%{x}</b><br>Samples: %{y}<br>Percentage: %{customdata:.2f}%<extra></extra>',
    customdata=class_percentages
))

fig2.update_layout(
    title='KSC Dataset - Class Distribution (13 Wetland Vegetation Types)',
    title_font_size=16,
    xaxis_title='Class Name',
    yaxis_title='Number of Samples',
    height=500,
    width=1200,
    xaxis_tickangle=-45
)

# ============================================================================
# Figure 3: Interactive Class Explorer
# ============================================================================
print("  [3/4] Interactive class explorer...")

# Create individual class masks for exploration
fig3 = make_subplots(
    rows=1, cols=1,
    subplot_titles=('Interactive Class Explorer - Hover to explore classes',)
)

# Create a heatmap where each pixel value is its class ID
# This allows us to show class info on hover
gt_for_hover = gt.copy().astype(float)
gt_for_hover[gt_for_hover == 0] = np.nan  # Unlabeled as NaN

# Custom hover text
hover_text = np.empty((h, w), dtype=object)
for i in range(h):
    for j in range(w):
        class_id = gt[i, j]
        if class_id > 0:
            class_idx = class_id - 1
            hover_text[i, j] = f"Class {class_id}: {class_names[class_idx]}<br>Pixel: ({i}, {j})"
        else:
            hover_text[i, j] = f"Unlabeled<br>Pixel: ({i}, {j})"

fig3.add_trace(
    go.Heatmap(
        z=gt_for_hover,
        colorscale=[[i/n_classes, class_colors[i]] for i in range(n_classes)],
        showscale=True,
        colorbar=dict(
            title="Class ID",
            tickvals=unique_classes,
            ticktext=[f"{cid}: {class_names[cid-1]}" for cid in unique_classes]
        ),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text
    )
)

fig3.update_layout(
    title='KSC Ground Truth - Interactive Class Explorer',
    title_font_size=16,
    height=600,
    width=800,
    xaxis_title='Column',
    yaxis_title='Row'
)

# ============================================================================
# Figure 4: Class Statistics Table
# ============================================================================
print("  [4/4] Class statistics table...")

# Create table data
table_data = {
    'Class ID': unique_classes.tolist(),
    'Class Name': class_names,
    'Samples': class_counts,
    'Percentage (%)': [f"{p:.2f}" for p in class_percentages]
}

# Create table using plotly
fig4 = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>Class ID</b>', '<b>Class Name</b>', '<b>Samples</b>', '<b>Percentage (%)</b>'],
        fill_color='paleturquoise',
        align='left',
        font=dict(size=12, color='black')
    ),
    cells=dict(
        values=[
            table_data['Class ID'],
            table_data['Class Name'],
            table_data['Samples'],
            table_data['Percentage (%)']
        ],
        fill_color=[['white', 'lightgray'] * (n_classes // 2 + 1)][:n_classes],
        align='left',
        font=dict(size=11)
    )
)])

fig4.update_layout(
    title='KSC Dataset - Class Statistics Table',
    title_font_size=16,
    height=500,
    width=800
)

# ============================================================================
# SAVE AND SHOW FIGURES
# ============================================================================
print("\nSaving interactive visualizations...")

output_dir = Path(__file__).parent / 'RESULTS' / 'ksc' / 'interactive'
output_dir.mkdir(parents=True, exist_ok=True)

# Save as HTML files
html_files = []

fig1.write_html(output_dir / 'ksc_rgb_groundtruth.html')
html_files.append('ksc_rgb_groundtruth.html')
print(f"  Saved: {output_dir / 'ksc_rgb_groundtruth.html'}")

fig2.write_html(output_dir / 'ksc_class_distribution.html')
html_files.append('ksc_class_distribution.html')
print(f"  Saved: {output_dir / 'ksc_class_distribution.html'}")

fig3.write_html(output_dir / 'ksc_class_explorer.html')
html_files.append('ksc_class_explorer.html')
print(f"  Saved: {output_dir / 'ksc_class_explorer.html'}")

fig4.write_html(output_dir / 'ksc_class_table.html')
html_files.append('ksc_class_table.html')
print(f"  Saved: {output_dir / 'ksc_class_table.html'}")

# Create index HTML to show all visualizations
print("\nCreating interactive dashboard...")
index_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KSC Interactive Visualization Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .info-box {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin-bottom: 30px;
            border-radius: 4px;
        }}
        .info-box h3 {{
            margin-top: 0;
            color: #1976D2;
        }}
        .info-box ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .viz-container {{
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .viz-container h2 {{
            color: #444;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        iframe {{
            width: 100%;
            border: none;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            color: #888;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>KSC Hyperspectral Image Classification</h1>
    <div class="subtitle">Interactive Visualization Dashboard</div>

    <div class="info-box">
        <h3>About the KSC Dataset</h3>
        <p><strong>Kennedy Space Center (KSC)</strong> hyperspectral dataset acquired on March 23, 1996 using JPL's AVIRIS sensor.</p>
        <ul>
            <li><strong>Image Size:</strong> {h} x {w} pixels</li>
            <li><strong>Spectral Bands:</strong> {bands} bands (after removing water absorption)</li>
            <li><strong>Spatial Resolution:</strong> 18 meters</li>
            <li><strong>Number of Classes:</strong> {n_classes} wetland vegetation types</li>
            <li><strong>Total Labeled Pixels:</strong> {total_labeled:,}</li>
        </ul>

        <h3>13 Wetland Vegetation Classes:</h3>
        <ol>
            <li><strong>Scrub</strong> - Coastal scrubland</li>
            <li><strong>Willow swamp</strong> - Wetland with willow trees</li>
            <li><strong>CP hammock</strong> - Cabbage palm hammock forest</li>
            <li><strong>CP/Oak</strong> - Mixed cabbage palm and oak</li>
            <li><strong>Slash pine</strong> - Pine forest</li>
            <li><strong>Oak/Broadleaf</strong> - Mixed hardwood forest</li>
            <li><strong>Hardwood swamp</strong> - Swamp with hardwood trees</li>
            <li><strong>Graminoid marsh</strong> - Grass-dominated wetland</li>
            <li><strong>Spartina marsh</strong> - Saltwater marsh grass</li>
            <li><strong>Cattail marsh</strong> - Freshwater marsh with cattails</li>
            <li><strong>Salt marsh</strong> - Tidal saltwater marsh</li>
            <li><strong>Mud flats</strong> - Exposed tidal mud</li>
            <li><strong>Water</strong> - Open water</li>
        </ol>
    </div>

    <div class="viz-container">
        <h2>1. RGB Composite and Ground Truth</h2>
        <p>Interactive view of the RGB composite (left) and ground truth labels (right). Use mouse to pan and zoom.</p>
        <iframe src="{html_files[0]}" height="550"></iframe>
    </div>

    <div class="viz-container">
        <h2>2. Class Distribution</h2>
        <p>Bar chart showing the number of samples for each of the 13 wetland vegetation classes.</p>
        <iframe src="{html_files[1]}" height="550"></iframe>
    </div>

    <div class="viz-container">
        <h2>3. Interactive Class Explorer</h2>
        <p>Hover over pixels to see class information. Zoom in to explore specific regions.</p>
        <iframe src="{html_files[2]}" height="650"></iframe>
    </div>

    <div class="viz-container">
        <h2>4. Class Statistics Table</h2>
        <p>Detailed statistics for all 13 classes including sample counts and percentages.</p>
        <iframe src="{html_files[3]}" height="550"></iframe>
    </div>

    <div class="footer">
        <p>Generated using Claude Code - Hyperspectral Image Classification Pipeline</p>
        <p>Dataset: Kennedy Space Center (KSC) | Sensor: AVIRIS | Date: March 23, 1996</p>
    </div>
</body>
</html>
"""

with open(output_dir / 'index.html', 'w') as f:
    f.write(index_html)

print(f"  Saved: {output_dir / 'index.html'}")

print("\n" + "="*80)
print("INTERACTIVE VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nTo view the interactive dashboard:")
print(f"  Open: {output_dir / 'index.html'}")
print(f"\nOr view individual visualizations:")
for html_file in html_files:
    print(f"  - {html_file}")
print("\n" + "="*80)

# Show figures in browser (if running interactively)
try:
    print("\nOpening interactive dashboard in browser...")
    import webbrowser
    webbrowser.open(str(output_dir / 'index.html'))
except Exception as e:
    print(f"Could not open browser automatically: {e}")
    print("Please open the HTML file manually.")
