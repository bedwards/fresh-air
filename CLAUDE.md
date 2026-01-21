# Claude Code Instructions for Music Analysis Project

## Project Overview

This project analyzes ABC notation music files using transformer interpretability techniques with Phi-4 14B. The goal is to produce a scholarly essay exploring how language models process musical notation, deployed as a GitHub Pages site.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify Ollama and Phi-4 are available
ollama list | grep phi4

# Run model harness test
python -c "import src.model_harness; print('Model harness OK')"
```

## Directory Structure

```
fresh-air/
├── README.md                 # Project overview for developers
├── CLAUDE.md                 # This file - instructions for Claude instances
├── ARCHITECTURE.md           # System design documentation
├── INITIAL_PROMPT.md         # Original project specification
├── requirements.txt          # Python dependencies
├── .venv/                    # Python virtual environment
├── scripts/
│   ├── setup.sh             # Environment setup
│   ├── count_words.py       # Word count utility for essay
│   ├── estimate_reading_time.py
│   └── deploy.sh            # GitHub Pages deployment
├── src/
│   ├── abc_generator.py     # ABC notation generation
│   ├── model_harness.py     # Phi-4 inference wrapper
│   ├── metrics_analyzer.py  # Interpretability metrics
│   ├── visualizations.py    # Polars + Altair viz
│   └── essay_builder.py     # HTML essay generation
├── data/
│   ├── abc_files/           # Generated ABC notation
│   ├── metrics/             # JSON metrics per song
│   └── audio/               # MP3 files (human-produced in Bitwig)
├── docs/                     # GitHub Pages site
│   ├── index.html           # Listing page
│   ├── essay.html           # Main essay
│   ├── assets/              # CSS, JS, images
│   └── visualizations/      # Embedded charts
└── tests/
```

## Key Files

### ABC Notation Files
- Location: `data/abc_files/`
- Naming: `{genre}_{number}.abc` (e.g., `traditional_001.abc`)
- Each ABC file has a sidecar JSON: `{genre}_{number}.json`
- Genre categories: traditional, avant-garde, experimental, noise, terrible, silence

### Metrics Files
- Location: `data/metrics/`
- Naming: `{abc_filename}.json` (e.g., `traditional_001.abc.json`)
- Contains: perplexity, attention entropy, surprisal, layer activations

### Visualizations
- Location: `docs/visualizations/{piece_name}/`
- Self-contained HTML files (Altair exports)
- BertViz attention visualizations

## Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Generate ABC files
python src/abc_generator.py --genre traditional --count 5

# Run analysis on a single file
python src/model_harness.py data/abc_files/traditional_001.abc

# Generate all metrics
python src/metrics_analyzer.py --all

# Create visualizations
python src/visualizations.py data/metrics/

# Build essay HTML
python src/essay_builder.py --output docs/essay.html

# Count words in essay
python scripts/count_words.py docs/essay.html

# Deploy to GitHub Pages
./scripts/deploy.sh
```

## Ollama / Phi-4 Usage

The model is accessed via Ollama API:

```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'phi4:14b',
    'prompt': abc_notation_text,
    'stream': False,
    'options': {
        'num_predict': 1,  # For perplexity calculation
    }
})
```

For detailed token-level analysis, use the `/api/embed` endpoint or direct transformers integration.

## GitHub Issue Workflow

### Creating Issues for Workers
Workers are spawned as separate Claude Code instances in git worktrees.

1. Main instance creates detailed GitHub Issue
2. Worker claims issue and creates branch
3. Worker completes task and creates PR
4. Main instance reviews and merges

### Issue Labels
- `generation` - ABC file generation tasks
- `abc-notation` - ABC notation related
- `core-infrastructure` - Model harness, metrics
- `metrics` - Analysis and metrics
- `visualization` - Charts and visualizations
- `analysis` - Deep analysis tasks
- `iteration` - Second-round generation

### Git Worktrees
```bash
# Create worktree for worker
git worktree add ../fresh-air-worker-1 -b worker-1

# List worktrees
git worktree list

# Remove when done
git worktree remove ../fresh-air-worker-1
```

## Interpretability Metrics

### Perplexity
- Formula: `exp(mean(-log(p(token))))`
- Baseline (traditional music): ~50-100
- High confusion: >300

### Attention Entropy
- Formula: `H = -sum(p * log(p))` per attention head
- Low entropy (<2 bits): focused attention
- High entropy (>4.5 bits): confused attention

### Surprisal
- Per-token: `-log2(p(token))`
- Normal: 2-6 bits
- High surprise: >10 bits

### Layer Activation Analysis
- Track L2 norm across layers
- Detect divergence from baseline
- Identify processing breakdown points

## Style Guidelines

### Essay Prose
- All prose, no bullet points in final output
- Academic but accessible
- First principles explanations
- Specific numerical results
- Visual examples for every claim

### Code Style
- Python 3.11+
- Type hints where helpful
- Docstrings for public functions
- Polars over pandas for data
- Altair for visualizations

## Audio Files

Audio is produced by the human in Bitwig from ABC notation:
1. Generate ABC files
2. Human converts to MIDI and produces in Bitwig
3. Human drops MP3 files into `data/audio/`
4. Essay builder embeds audio players

Do not attempt to generate audio programmatically.

## Constraints

1. **Never downgrade packages** - Ask human for help if conflicts
2. **Check before install** - Verify what's already present
3. **Document everything** - Future instances need context
4. **Commit frequently** - Small, atomic commits
5. **Test incrementally** - Don't build everything before testing

## Contact

Brian Edwards
- Email: brian.mabry.edwards@gmail.com
- Phone: 512-584-6841
- Location: Waco, Texas, USA
