# Substrate Melodies: Musical Transformer Interpretability Research

A research project analyzing ABC notation music files using transformer interpretability techniques with Phi-4 14B. The project generates diverse musical pieces, analyzes them using attention entropy, perplexity, and surprisal metrics, and produces a scholarly essay with interactive visualizations.

## Project Goals

1. Generate ABC notation music across various styles (traditional, avant-garde, experimental, noise, terrible, silence)
2. Analyze how Phi-4 14B processes these pieces using interpretability metrics
3. Produce a scholarly essay exploring the boundaries of machine musical understanding
4. Deploy as a professional GitHub Pages site with embedded audio and visualizations

## Quick Start

```bash
# Clone and setup
git clone https://github.com/bedwards/fresh-air.git
cd fresh-air
./scripts/setup.sh

# Activate environment
source .venv/bin/activate

# Generate ABC files
python src/abc_generator.py --all --count 5

# Analyze with Phi-4
python src/model_harness.py --all data/abc_files/

# Generate visualizations
python src/visualizations.py data/metrics/

# Build essay
python src/essay_builder.py
```

## Project Structure

```
├── src/                      # Source code
│   ├── abc_generator.py     # ABC notation generation
│   ├── model_harness.py     # Phi-4 inference wrapper
│   ├── metrics_analyzer.py  # Interpretability metrics
│   ├── visualizations.py    # Polars + Altair viz
│   └── essay_builder.py     # HTML essay generation
├── data/                     # Generated data
│   ├── abc_files/           # ABC notation files
│   ├── metrics/             # Analysis metrics (JSON)
│   └── audio/               # MP3 files (human-produced)
├── docs/                     # GitHub Pages site
├── scripts/                  # Utility scripts
└── tests/                    # Test suite
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Instructions for Claude Code instances
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and data flow
- [INITIAL_PROMPT.md](INITIAL_PROMPT.md) - Original project specification

## Requirements

- Python 3.11+
- Ollama with Phi-4 14B model
- Dependencies in `requirements.txt`

## Author

Brian Edwards
brian.mabry.edwards@gmail.com
Waco, Texas, USA

Built with Claude Code (Claude Opus 4.5)
