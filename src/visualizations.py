#!/usr/bin/env python3
"""Visualization suite for transformer interpretability metrics.

This module generates interactive visualizations using Altair and BertViz
for the essay and analysis.

Visualization types:
    - Perplexity comparison bar charts
    - Attention entropy heatmaps
    - Token surprisal timelines
    - Layer activation trajectories
    - Genre clustering (UMAP/PCA)
    - Comparative distributions

Usage:
    python src/visualizations.py data/metrics/
    python src/visualizations.py --piece traditional_001
"""

import argparse
import json
from pathlib import Path
from typing import Any

import altair as alt
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure Altair for self-contained HTML output
alt.data_transformers.disable_max_rows()

# Color palette (colorblind-friendly)
PALETTE = {
    "traditional": "#4477AA",
    "avant-garde": "#EE6677",
    "experimental": "#228833",
    "noise": "#CCBB44",
    "terrible": "#66CCEE",
    "silence": "#AA3377",
}


def load_metrics(metrics_dir: Path) -> pl.DataFrame:
    """Load all metrics into a DataFrame."""
    metrics_files = sorted(metrics_dir.glob("*.json"))

    records = []
    for path in metrics_files:
        data = json.loads(path.read_text())

        filename = data.get("filename", path.stem)
        genre = filename.split("_")[0] if "_" in filename else "unknown"

        record = {
            "filename": filename,
            "genre": genre,
            "perplexity": data.get("perplexity", {}).get("overall", 0),
            "attention_entropy": data.get("attention", {}).get("mean_entropy", 0),
            "surprisal_mean": data.get("surprisal", {}).get("mean", 0),
            "token_count": data.get("token_count", 0),
        }
        records.append(record)

    return pl.DataFrame(records)


def perplexity_bar_chart(df: pl.DataFrame) -> alt.Chart:
    """Create bar chart comparing perplexity across pieces."""
    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X('filename:N', sort='-y', title='Piece'),
        y=alt.Y('perplexity:Q', title='Perplexity'),
        color=alt.Color('genre:N', scale=alt.Scale(domain=list(PALETTE.keys()),
                                                    range=list(PALETTE.values())),
                        title='Genre'),
        tooltip=['filename', 'genre', 'perplexity']
    ).properties(
        title='Perplexity by Piece',
        width=600,
        height=400
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=12
    )

    return chart


def attention_entropy_heatmap(per_layer_entropy: list[list[float]], title: str) -> alt.Chart:
    """Create heatmap of attention entropy [layer x head]."""
    # Convert to DataFrame format
    records = []
    for layer_idx, layer_heads in enumerate(per_layer_entropy):
        for head_idx, entropy in enumerate(layer_heads):
            records.append({
                "layer": layer_idx,
                "head": head_idx,
                "entropy": entropy
            })

    df = pl.DataFrame(records)

    chart = alt.Chart(df.to_pandas()).mark_rect().encode(
        x=alt.X('head:O', title='Attention Head'),
        y=alt.Y('layer:O', title='Layer', sort='descending'),
        color=alt.Color('entropy:Q',
                        scale=alt.Scale(scheme='viridis'),
                        title='Entropy (bits)'),
        tooltip=['layer', 'head', 'entropy']
    ).properties(
        title=f'Attention Entropy: {title}',
        width=400,
        height=500
    )

    return chart


def surprisal_timeline(surprisal_values: list[float], tokens: list[str], title: str) -> alt.Chart:
    """Create line chart showing surprisal per token position."""
    df = pl.DataFrame({
        "position": list(range(len(surprisal_values))),
        "surprisal": surprisal_values,
        "token": tokens[:len(surprisal_values)] if tokens else [""] * len(surprisal_values)
    })

    # Base line chart
    line = alt.Chart(df.to_pandas()).mark_line(color='steelblue').encode(
        x=alt.X('position:Q', title='Token Position'),
        y=alt.Y('surprisal:Q', title='Surprisal (bits)'),
    )

    # Add points for high surprisal tokens
    points = alt.Chart(df.to_pandas()).mark_circle(color='red', size=50).encode(
        x='position:Q',
        y='surprisal:Q',
        tooltip=['position', 'token', 'surprisal']
    ).transform_filter(
        alt.datum.surprisal > 10  # Highlight high surprisal
    )

    # Add threshold line
    threshold = alt.Chart(pl.DataFrame({"y": [10]}).to_pandas()).mark_rule(
        color='red',
        strokeDash=[5, 5]
    ).encode(y='y:Q')

    chart = (line + points + threshold).properties(
        title=f'Token Surprisal: {title}',
        width=600,
        height=300
    )

    return chart


def layer_activation_trajectory(norms: list[float], title: str) -> alt.Chart:
    """Create line chart showing activation norms across layers."""
    df = pl.DataFrame({
        "layer": list(range(len(norms))),
        "norm": norms
    })

    chart = alt.Chart(df.to_pandas()).mark_line(point=True).encode(
        x=alt.X('layer:Q', title='Layer'),
        y=alt.Y('norm:Q', title='L2 Norm'),
        tooltip=['layer', 'norm']
    ).properties(
        title=f'Activation Trajectory: {title}',
        width=500,
        height=300
    )

    return chart


def genre_clustering(df: pl.DataFrame) -> alt.Chart:
    """Create 2D projection of metric vectors using PCA."""
    # Select numeric columns for clustering
    feature_cols = ['perplexity', 'attention_entropy', 'surprisal_mean', 'token_count']
    features = df.select(feature_cols).to_numpy()

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features_scaled)

    # Create DataFrame with coordinates
    cluster_df = df.with_columns([
        pl.lit(coords[:, 0]).alias("pc1"),
        pl.lit(coords[:, 1]).alias("pc2"),
    ])

    chart = alt.Chart(cluster_df.to_pandas()).mark_circle(size=100).encode(
        x=alt.X('pc1:Q', title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'),
        y=alt.Y('pc2:Q', title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'),
        color=alt.Color('genre:N', scale=alt.Scale(domain=list(PALETTE.keys()),
                                                    range=list(PALETTE.values()))),
        tooltip=['filename', 'genre', 'perplexity', 'attention_entropy']
    ).properties(
        title='Genre Clustering (PCA)',
        width=500,
        height=400
    )

    return chart


def genre_distribution(df: pl.DataFrame, metric: str, title: str) -> alt.Chart:
    """Create overlapping histograms comparing distributions across genres."""
    chart = alt.Chart(df.to_pandas()).mark_area(
        opacity=0.5,
        interpolate='step'
    ).encode(
        x=alt.X(f'{metric}:Q', bin=alt.Bin(maxbins=20), title=title),
        y=alt.Y('count()', stack=None, title='Count'),
        color=alt.Color('genre:N', scale=alt.Scale(domain=list(PALETTE.keys()),
                                                    range=list(PALETTE.values())))
    ).properties(
        title=f'{title} Distribution by Genre',
        width=500,
        height=300
    )

    return chart


def save_chart(chart: alt.Chart, output_path: Path) -> None:
    """Save chart as self-contained HTML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_path), format='html')
    print(f"Saved: {output_path}")


def generate_piece_visualizations(metrics_path: Path, output_dir: Path) -> None:
    """Generate all visualizations for a single piece."""
    data = json.loads(metrics_path.read_text())
    filename = data.get("filename", metrics_path.stem)
    piece_name = filename.replace(".abc", "")

    piece_output = output_dir / piece_name
    piece_output.mkdir(parents=True, exist_ok=True)

    # Attention entropy heatmap
    attention_data = data.get("attention", {})
    if attention_data.get("per_head"):
        chart = attention_entropy_heatmap(attention_data["per_head"], piece_name)
        save_chart(chart, piece_output / "attention.html")

    # Surprisal timeline
    surprisal_data = data.get("surprisal", {})
    if surprisal_data.get("per_token"):
        # Note: tokens would need to be stored in metrics
        tokens = [""] * len(surprisal_data["per_token"])
        chart = surprisal_timeline(surprisal_data["per_token"], tokens, piece_name)
        save_chart(chart, piece_output / "surprisal.html")

    # Layer activation trajectory
    activation_data = data.get("activations", {})
    if activation_data.get("per_layer_norm"):
        chart = layer_activation_trajectory(activation_data["per_layer_norm"], piece_name)
        save_chart(chart, piece_output / "activations.html")

    print(f"Generated visualizations for: {piece_name}")


def generate_corpus_visualizations(metrics_dir: Path, output_dir: Path) -> None:
    """Generate all visualizations for the corpus."""
    df = load_metrics(metrics_dir)

    if df.height == 0:
        print("No metrics found")
        return

    corpus_output = output_dir / "corpus"
    corpus_output.mkdir(parents=True, exist_ok=True)

    # Perplexity bar chart
    chart = perplexity_bar_chart(df)
    save_chart(chart, corpus_output / "perplexity_comparison.html")

    # Genre clustering
    if df.height >= 3:  # Need enough points for PCA
        chart = genre_clustering(df)
        save_chart(chart, corpus_output / "genre_clustering.html")

    # Distribution comparisons
    chart = genre_distribution(df, "perplexity", "Perplexity")
    save_chart(chart, corpus_output / "perplexity_distribution.html")

    chart = genre_distribution(df, "attention_entropy", "Attention Entropy")
    save_chart(chart, corpus_output / "entropy_distribution.html")

    # Generate per-piece visualizations
    for metrics_path in metrics_dir.glob("*.json"):
        generate_piece_visualizations(metrics_path, output_dir)

    print(f"\nGenerated corpus visualizations in: {output_dir}")


def create_manifest(output_dir: Path) -> None:
    """Create manifest JSON listing all visualizations."""
    manifest = {"pieces": {}, "corpus": []}

    # Corpus visualizations
    corpus_dir = output_dir / "corpus"
    if corpus_dir.exists():
        manifest["corpus"] = [f.name for f in corpus_dir.glob("*.html")]

    # Per-piece visualizations
    for piece_dir in output_dir.iterdir():
        if piece_dir.is_dir() and piece_dir.name != "corpus":
            manifest["pieces"][piece_dir.name] = [f.name for f in piece_dir.glob("*.html")]

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Created manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interpretability visualizations")
    parser.add_argument("metrics_dir", type=Path, help="Directory containing metrics JSON files")
    parser.add_argument("--output", type=Path, default=Path("docs/visualizations"),
                        help="Output directory")
    parser.add_argument("--piece", help="Generate visualizations for specific piece only")

    args = parser.parse_args()

    if args.piece:
        metrics_path = args.metrics_dir / f"{args.piece}.abc.json"
        if metrics_path.exists():
            generate_piece_visualizations(metrics_path, args.output)
        else:
            print(f"Metrics file not found: {metrics_path}")
    else:
        generate_corpus_visualizations(args.metrics_dir, args.output)
        create_manifest(args.output)


if __name__ == "__main__":
    main()
