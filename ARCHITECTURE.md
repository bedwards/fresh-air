# System Architecture

## Overview

This system generates ABC notation music, analyzes it using transformer interpretability techniques, and produces a scholarly essay with visualizations. The architecture prioritizes reproducibility, modularity, and clear data flow.

## Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ABC Generator  │────▶│  ABC Files      │────▶│  Model Harness  │
│  (src/)         │     │  (data/abc/)    │     │  (Phi-4 14B)    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Essay Builder  │◀────│  Visualizations │◀────│  Metrics JSON   │
│  (HTML output)  │     │  (Altair/BertViz)│     │  (data/metrics/)│
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  GitHub Pages   │◀────│  Human: Audio   │
│  (docs/)        │     │  (Bitwig→MP3)   │
└─────────────────┘     └─────────────────┘
```

## Components

### 1. ABC Generator (`src/abc_generator.py`)

**Purpose**: Generate diverse ABC notation files across musical genres.

**Input**: Genre specification, count
**Output**: ABC files + metadata JSON

**Genres**:
- `traditional`: Folk tunes, hymns, classical themes (predictable patterns)
- `avant-garde`: Unusual time signatures, extended harmony
- `experimental`: Extreme registers, dense polyphony
- `noise`: Random sequences, no tonal center
- `terrible`: Intentionally bad voice leading
- `silence`: Sparse notes, extended rests

**Design Decisions**:
- Multi-voice support (1-4 voices)
- Deterministic generation with seed control
- Metadata captures all generation parameters

### 2. Model Harness (`src/model_harness.py`)

**Purpose**: Interface with Phi-4 14B and extract interpretability metrics.

**Input**: ABC notation text
**Output**: Comprehensive metrics JSON

**Architecture Options**:

**Option A: Ollama API (Recommended for Q4_0)**
```python
# Uses local Ollama server
# Advantages: Easy setup, handles quantization
# Disadvantages: Limited access to internals

POST http://localhost:11434/api/generate
{
    "model": "phi4:14b",
    "prompt": "<abc_text>",
    "options": {"num_predict": 1}
}
```

**Option B: Transformers + llama.cpp**
```python
# Direct model access via transformers
# Advantages: Full attention/activation access
# Disadvantages: Complex setup, memory intensive

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    output_attentions=True,
    output_hidden_states=True
)
```

**Metrics Computed**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Perplexity | `exp(mean(-log(p)))` | Effective vocabulary size per token |
| Attention Entropy | `H = -Σ(p log p)` | Focus vs diffusion of attention |
| Surprisal | `-log2(p(token))` | Unexpectedness of each token |
| Activation Norm | `||h||_2` per layer | Processing intensity |

### 3. Metrics Analyzer (`src/metrics_analyzer.py`)

**Purpose**: Interpret raw metrics and generate human-readable descriptions.

**Input**: Metrics JSON files
**Output**: Analyzed metrics with interpretations

**Analysis Pipeline**:
1. Compute baseline statistics from traditional pieces
2. Calculate z-scores for all metrics
3. Identify outliers (>2σ)
4. Generate natural language descriptions
5. Classify pieces by metric profiles

**Example Output**:
```json
{
  "filename": "experimental_002.abc",
  "perplexity": 418.2,
  "perplexity_zscore": 3.4,
  "interpretation": "Severely elevated perplexity indicates the model encounters unfamiliar token sequences. Layer 12+ shows activation divergence, suggesting harmonic processing breakdown.",
  "classification": "syntactically_valid_semantically_chaotic"
}
```

### 4. Visualizations (`src/visualizations.py`)

**Purpose**: Generate interactive charts for the essay.

**Input**: Metrics JSON files
**Output**: Self-contained HTML visualizations

**Visualization Types**:

| Type | Library | Purpose |
|------|---------|---------|
| Attention Heatmap | BertViz | Show attention flow |
| Perplexity Bars | Altair | Compare across pieces |
| Surprisal Timeline | Altair | Token-level analysis |
| Layer Trajectory | Altair | Activation patterns |
| Genre Clustering | Altair + UMAP | Metric space projection |

**Design Requirements**:
- All charts export as self-contained HTML
- No external dependencies in deployed site
- Colorblind-friendly palette
- Responsive design
- Data tables as accessibility fallback

### 5. Essay Builder (`src/essay_builder.py`)

**Purpose**: Assemble final HTML essay with embedded media.

**Input**: Essay content, visualizations, audio paths
**Output**: `docs/essay.html`

**Template Structure**:
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Substrate Melodies</title>
    <style>/* Embedded CSS */</style>
</head>
<body>
    <header>
        <h1>Substrate Melodies</h1>
        <div class="metadata">
            Author: Brian Edwards | Date: January 21, 2026
            Word Count: {word_count} | Reading Time: {reading_time}
        </div>
    </header>
    <main>
        <!-- Essay sections with embedded visualizations and audio -->
    </main>
</body>
</html>
```

## Model Configuration

### Phi-4 14B via Ollama

```yaml
Model: phi4:14b
Quantization: Q4_0 (4-bit)
Context Length: 16384 tokens
Memory Usage: ~9GB
Inference Speed: ~20 tokens/sec (M-series Mac)
```

### Key Considerations

1. **Quantization Impact**: Q4_0 reduces precision but maintains reasonable quality for interpretability work. Activation patterns may be noisier than full-precision.

2. **Context Window**: ABC notation is compact; typical pieces fit within 2K tokens. Use sliding window for longer pieces.

3. **Batch Processing**: Process one piece at a time to avoid memory pressure. Cache embeddings for repeated analysis.

## Data Schemas

### ABC Metadata (`data/abc_files/*.json`)
```json
{
  "filename": "traditional_001.abc",
  "genre": "traditional",
  "subgenre": "irish_jig",
  "description": "Irish jig in D major, two voices in parallel thirds",
  "voices": 2,
  "measures": 32,
  "time_signature": "6/8",
  "key": "D",
  "tempo": 120,
  "generation_params": {
    "seed": 42,
    "algorithm": "markov_chain",
    "corpus_source": "session_tunes"
  }
}
```

### Metrics Output (`data/metrics/*.json`)
```json
{
  "filename": "traditional_001.abc",
  "model": "phi4:14b-q4_0",
  "timestamp": "2026-01-21T10:30:00Z",
  "token_count": 847,
  "perplexity": {
    "overall": 62.4,
    "per_token": [12.3, 45.2, ...],
    "max": 89.1,
    "min": 8.2
  },
  "attention": {
    "mean_entropy": 2.4,
    "per_layer": [1.8, 2.1, ...],
    "per_head": [[1.2, 1.5, ...], ...],
    "high_entropy_heads": [
      {"layer": 18, "head": 7, "entropy": 4.8}
    ]
  },
  "surprisal": {
    "mean": 5.2,
    "per_token": [3.1, 4.2, ...],
    "high_surprisal_tokens": [
      {"position": 234, "token": "^c", "surprisal": 11.2}
    ]
  },
  "activations": {
    "per_layer_norm": [1.2, 1.4, ...],
    "variance_per_layer": [0.1, 0.2, ...]
  },
  "comparative": {
    "baseline": "traditional_mean",
    "perplexity_zscore": 0.3,
    "entropy_zscore": -0.1
  }
}
```

## Deployment

### GitHub Pages Structure

```
docs/
├── index.html          # Landing page with essay list
├── essay.html          # Main essay
├── assets/
│   ├── style.css      # Stylesheet
│   ├── script.js      # Audio player, interactions
│   └── images/        # Figures, diagrams
├── visualizations/
│   ├── traditional_001/
│   │   ├── attention.html
│   │   ├── perplexity.html
│   │   └── surprisal.html
│   └── ...
└── audio/
    ├── traditional_001.mp3
    └── ...
```

### Deployment Script

```bash
#!/bin/bash
# scripts/deploy.sh

# Update word count
python scripts/count_words.py docs/essay.html > docs/assets/wordcount.txt
python scripts/estimate_reading_time.py docs/essay.html > docs/assets/reading_time.txt

# Inject metadata
python scripts/inject_metadata.py docs/essay.html

# Commit docs changes
git add docs/
git commit -m "Update essay content"

# Deploy to gh-pages
git subtree push --prefix docs origin gh-pages
```

## Error Handling

### Model Failures
- Timeout: Retry with smaller context
- OOM: Reduce batch size to 1
- Connection: Check Ollama service status

### Visualization Failures
- Large datasets: Aggregate before plotting
- Missing data: Show placeholder with explanation
- Export failures: Fall back to PNG

### Build Failures
- Missing audio: Show placeholder, note pending
- Missing viz: Build without, log warning
- Invalid metrics: Skip piece, log error

## Performance Targets

| Operation | Target | Measured |
|-----------|--------|----------|
| ABC Generation (26 files) | <30s | TBD |
| Single File Analysis | <60s | TBD |
| Full Corpus Analysis | <30min | TBD |
| Visualization Build | <5min | TBD |
| Essay Build | <30s | TBD |
| Full Pipeline | <45min | TBD |
