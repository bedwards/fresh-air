# Visualization Suite

This document describes the visualization system for interpreting transformer metrics on ABC notation music files.

## Overview

The visualization suite generates interactive HTML charts using [Altair](https://altair-viz.github.io/), a declarative statistical visualization library for Python. All outputs are self-contained HTML files that can be viewed in any modern browser without an internet connection.

## Quick Start

```bash
# Activate the environment
source /path/to/fresh-air/.venv/bin/activate

# Generate all visualizations from metrics
python src/visualizations.py data/metrics/ --output docs/visualizations/

# Generate for a specific piece
python src/visualizations.py data/metrics/ --piece traditional_001
```

## Visualization Types

### Corpus-Level Visualizations

These charts compare metrics across the entire corpus of music pieces.

#### 1. Perplexity Comparison Bar Chart

**File:** `corpus/perplexity_comparison.html`

**Purpose:** Compare model perplexity across all pieces in the corpus.

**Interpretation:**
- **Low perplexity (<50):** The model finds the piece highly predictable, suggesting familiar patterns
- **Normal perplexity (50-100):** The model recognizes the piece as typical musical notation
- **Elevated perplexity (100-200):** Some unfamiliar sequences present
- **High perplexity (>200):** The model is significantly uncertain about token predictions

**Features:**
- Bars sorted by perplexity (highest first)
- Color-coded by genre
- Hover for exact values

---

#### 2. Genre Clustering (PCA)

**File:** `corpus/genre_clustering.html`

**Purpose:** Visualize how pieces cluster in metric space using Principal Component Analysis.

**Interpretation:**
- Pieces close together share similar metric profiles
- Tight genre clusters suggest consistent characteristics within that genre
- Outliers may indicate unusual pieces worth investigating
- Axis labels show explained variance (e.g., "PC1 (45.2% variance)")

**Features:**
- 2D projection of 4-dimensional metric space (perplexity, entropy, surprisal, token count)
- Points colored by genre
- Hover for piece details

---

#### 3. Perplexity Distribution

**File:** `corpus/perplexity_distribution.html`

**Purpose:** Compare perplexity distributions across genres.

**Interpretation:**
- Wide distributions indicate high variance within a genre
- Non-overlapping distributions suggest genres are well-separated
- Bimodal distributions may indicate sub-genres

**Features:**
- Semi-transparent overlapping areas for each genre
- Step interpolation for clear boundaries
- Hover for count details

---

#### 4. Entropy Distribution

**File:** `corpus/entropy_distribution.html`

**Purpose:** Compare attention entropy distributions across genres.

**Interpretation:**
- Low entropy (<2 bits): Focused attention, model knows where to look
- Moderate entropy (2-3.5 bits): Normal attention distribution
- High entropy (>4 bits): Diffuse attention, model is uncertain

### Per-Piece Visualizations

These charts provide detailed analysis of individual pieces.

#### 5. Surprisal Timeline

**File:** `{piece_name}/surprisal.html`

**Purpose:** Show per-token surprisal across the piece.

**Interpretation:**
- **Y-axis:** Surprisal in bits (information content)
- **Red threshold line:** 10 bits - tokens above this are highly surprising
- **Red dots:** Highlight tokens exceeding the threshold
- **Spikes:** Indicate unexpected tokens or transitions

**What to look for:**
- Consistent low surprisal: The model understands the piece well
- Periodic spikes: May indicate structural boundaries (bar lines, sections)
- Sustained high surprisal: The model struggles with this section
- Token clusters above threshold: Unusual melodic or harmonic sequences

---

#### 6. Activation Trajectory

**File:** `{piece_name}/activations.html`

**Purpose:** Track L2 norm of hidden states across transformer layers.

**Interpretation:**
- **Normal trajectory:** Gradual increase or plateau
- **Sudden drops:** May indicate processing bottlenecks
- **Exponential growth:** Could indicate representation drift
- **Final layer norm:** Correlates with output confidence

**What to look for:**
- Smooth curves suggest stable processing
- Erratic patterns may indicate the model struggling to represent the content
- Compare trajectories between genres to identify processing differences

---

#### 7. Attention Entropy Heatmap

**File:** `{piece_name}/attention.html`

**Purpose:** Visualize attention entropy across all layers and heads.

**Interpretation:**
- **Color scale:** Viridis (purple = low, yellow = high)
- **Low entropy (dark):** Head focuses on specific tokens
- **High entropy (bright):** Head distributes attention broadly

**What to look for:**
- **Structured patterns:** Indicate learned syntactic or semantic rules
- **Uniform high entropy:** Model may not understand the content
- **Specific high-entropy layers:** May indicate where the model "decides" structure
- **Vertical bands:** All heads in a layer agree (unusual)

## Color Palette

The visualization suite uses a colorblind-friendly palette:

| Genre | Color | Hex Code |
|-------|-------|----------|
| Traditional | Blue | `#4477AA` |
| Avant-garde | Red | `#EE6677` |
| Experimental | Green | `#228833` |
| Noise | Yellow | `#CCBB44` |
| Terrible | Cyan | `#66CCEE` |
| Silence | Purple | `#AA3377` |
| Unknown | Gray | `#999999` |

This palette is designed to be distinguishable by people with common forms of color blindness (deuteranopia, protanopia).

## Output Structure

```
docs/visualizations/
├── corpus/
│   ├── perplexity_comparison.html
│   ├── genre_clustering.html
│   ├── perplexity_distribution.html
│   └── entropy_distribution.html
├── traditional_001/
│   ├── surprisal.html
│   ├── activations.html
│   └── attention.html
├── avantgarde_001/
│   ├── surprisal.html
│   └── activations.html
└── manifest.json
```

### Manifest File

The `manifest.json` file provides a machine-readable index of all generated visualizations:

```json
{
  "corpus": [
    "perplexity_comparison.html",
    "genre_clustering.html",
    "perplexity_distribution.html",
    "entropy_distribution.html"
  ],
  "pieces": {
    "traditional_001": ["surprisal", "activations", "attention"],
    "avantgarde_001": ["surprisal", "activations"]
  }
}
```

## Technical Notes

### Self-Contained HTML

All charts are saved as self-contained HTML files using Altair's `chart.save()` method. The HTML includes:
- Inline Vega-Lite JavaScript
- Embedded data
- No external dependencies

This means charts work offline and can be shared as single files.

### Responsive Design

Charts use `width='container'` to automatically resize to their container. For best results, view in a modern browser and resize the window to see the responsive behavior.

### Interactive Features

All charts support:
- **Pan and zoom:** Click and drag to pan, scroll to zoom
- **Tooltips:** Hover over data points for details
- **Reset:** Double-click to reset the view

### Data Requirements

The visualization system expects metrics JSON files in the format produced by `model_harness.py`:

```json
{
  "filename": "traditional_001.abc",
  "token_count": 150,
  "perplexity": {
    "overall": 45.2,
    "per_token": [...]
  },
  "attention": {
    "mean_entropy": 2.8,
    "per_head": [[...], [...]]
  },
  "surprisal": {
    "mean": 4.5,
    "per_token": [...]
  },
  "activations": {
    "per_layer_norm": [...]
  }
}
```

### Graceful Degradation

The system handles missing data gracefully:
- Missing metrics files: Skipped with warning
- Missing fields in JSON: Charts show "No Data" message
- Insufficient data for PCA: Message displayed instead of chart
- Malformed JSON: Skipped with warning

## Programmatic Usage

The visualization functions can be imported and used directly:

```python
from visualizations import (
    perplexity_bar_chart,
    attention_entropy_heatmap,
    surprisal_timeline,
    layer_activation_trajectory,
    genre_clustering,
    genre_distribution,
    load_metrics,
    save_chart
)
import polars as pl

# Load metrics into a DataFrame
df = load_metrics(Path("data/metrics/"))

# Create a chart
chart = perplexity_bar_chart(df)

# Save to HTML
save_chart(chart, Path("output/perplexity.html"))

# Or display in Jupyter
chart.display()
```

## Troubleshooting

### "No metrics files found"

Ensure the metrics directory contains JSON files from `model_harness.py`. The directory should have files like `traditional_001.abc.json`.

### Charts not rendering

1. Ensure you have a modern browser (Chrome, Firefox, Safari, Edge)
2. Check that JavaScript is enabled
3. Try opening the HTML file directly (not via file:// on some systems)

### "Need at least 3 pieces for PCA"

The genre clustering chart requires at least 3 data points for PCA. Generate metrics for more pieces and try again.

### Colors not matching genres

The system normalizes genre names (e.g., "avant-garde" becomes "avantgarde"). Check the legend for the actual genre mapping.
