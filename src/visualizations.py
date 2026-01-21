#!/usr/bin/env python3
"""Visualization suite for transformer interpretability metrics.

This module generates interactive visualizations using Altair for the essay
and analysis of ABC notation music files.

Visualization types:
    - Perplexity comparison bar charts
    - Attention entropy heatmaps
    - Token surprisal timelines
    - Layer activation trajectories
    - Genre clustering (PCA)
    - Comparative distributions

Usage:
    python src/visualizations.py data/metrics/
    python src/visualizations.py data/metrics/ --output docs/visualizations/
    python src/visualizations.py --piece traditional_001 data/metrics/
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

# Colorblind-friendly palette
PALETTE = {
    "traditional": "#4477AA",   # Blue
    "avantgarde": "#EE6677",    # Red
    "experimental": "#228833",  # Green
    "noise": "#CCBB44",         # Yellow
    "terrible": "#66CCEE",      # Cyan
    "silence": "#AA3377",       # Purple
}

# Common chart configuration for responsiveness
CHART_CONFIG = {
    "autosize": {"type": "fit", "contains": "padding"},
}


def normalize_genre(genre: str) -> str:
    """Normalize genre names for consistent palette mapping.

    Handles variations like 'avant-garde' vs 'avantgarde'.
    """
    normalized = genre.lower().replace("-", "").replace("_", "").strip()
    # Map common variations
    mapping = {
        "avantgarde": "avantgarde",
        "avant": "avantgarde",
    }
    return mapping.get(normalized, normalized)


def load_metrics(metrics_dir: Path) -> pl.DataFrame:
    """Load all metrics into a DataFrame.

    Searches for JSON files containing model analysis metrics.
    Handles gracefully if directory is empty or files are malformed.
    """
    metrics_files = sorted(metrics_dir.glob("*.json"))

    if not metrics_files:
        print(f"No metrics files found in {metrics_dir}")
        return pl.DataFrame()

    records = []
    for path in metrics_files:
        try:
            data = json.loads(path.read_text())

            # Skip files that don't look like metrics (e.g., manifest.json)
            if "perplexity" not in data and "attention" not in data:
                continue

            filename = data.get("filename", path.stem)
            # Extract genre from filename (e.g., "traditional_001.abc" -> "traditional")
            if "_" in filename:
                genre = filename.split("_")[0]
            else:
                genre = "unknown"

            # Normalize genre for consistent palette matching
            genre = normalize_genre(genre)

            record = {
                "filename": filename,
                "genre": genre,
                "perplexity": data.get("perplexity", {}).get("overall", 0),
                "attention_entropy": data.get("attention", {}).get("mean_entropy", 0),
                "surprisal_mean": data.get("surprisal", {}).get("mean", 0),
                "token_count": data.get("token_count", 0),
            }
            records.append(record)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {path}: {e}")
            continue

    if not records:
        print("No valid metrics records found")
        return pl.DataFrame()

    return pl.DataFrame(records)


def perplexity_bar_chart(df: pl.DataFrame) -> alt.Chart:
    """Create bar chart comparing perplexity across pieces.

    Features:
        - Color-coded by genre using colorblind-friendly palette
        - Sorted by perplexity value (highest first)
        - Interactive tooltips with full data
        - Responsive sizing

    Args:
        df: DataFrame with columns: filename, genre, perplexity

    Returns:
        Altair Chart object
    """
    # Get unique genres present in data for scale domain
    genres = df["genre"].unique().to_list()
    palette_domain = [g for g in PALETTE.keys() if g in genres]
    palette_range = [PALETTE[g] for g in palette_domain]

    # Add unknown genre if present
    if "unknown" in genres and "unknown" not in palette_domain:
        palette_domain.append("unknown")
        palette_range.append("#999999")

    chart = alt.Chart(df.to_pandas()).mark_bar().encode(
        x=alt.X(
            'filename:N',
            sort=alt.EncodingSortField(field='perplexity', order='descending'),
            title='Piece',
            axis=alt.Axis(labelAngle=-45)
        ),
        y=alt.Y('perplexity:Q', title='Perplexity'),
        color=alt.Color(
            'genre:N',
            scale=alt.Scale(domain=palette_domain, range=palette_range),
            title='Genre',
            legend=alt.Legend(orient='right')
        ),
        tooltip=[
            alt.Tooltip('filename:N', title='Piece'),
            alt.Tooltip('genre:N', title='Genre'),
            alt.Tooltip('perplexity:Q', title='Perplexity', format='.2f')
        ]
    ).properties(
        title='Perplexity Comparison Across Corpus',
        width='container',
        height=400
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=13
    ).configure_title(
        fontSize=16,
        anchor='start'
    ).interactive()

    return chart


def attention_entropy_heatmap(per_layer_entropy: list[list[float]], title: str) -> alt.Chart:
    """Create heatmap of attention entropy [layer x head].

    Features:
        - Viridis color scale for accessibility
        - Interactive tooltips showing exact values
        - Layer 0 at bottom, highest layer at top

    Args:
        per_layer_entropy: 2D list [layer][head] of entropy values in bits
        title: Title for the chart (typically the piece name)

    Returns:
        Altair Chart object
    """
    if not per_layer_entropy:
        # Return empty chart with message
        return alt.Chart(pl.DataFrame({"x": [0], "y": [0], "text": ["No data"]}).to_pandas()).mark_text().encode(
            text='text:N'
        ).properties(title=f'Attention Entropy: {title} (No Data)')

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
        color=alt.Color(
            'entropy:Q',
            scale=alt.Scale(scheme='viridis'),
            title='Entropy (bits)',
            legend=alt.Legend(orient='right')
        ),
        tooltip=[
            alt.Tooltip('layer:O', title='Layer'),
            alt.Tooltip('head:O', title='Head'),
            alt.Tooltip('entropy:Q', title='Entropy', format='.3f')
        ]
    ).properties(
        title=f'Attention Entropy: {title}',
        width='container',
        height=500
    ).configure_title(
        fontSize=16,
        anchor='start'
    ).interactive()

    return chart


def surprisal_timeline(surprisal_values: list[float], tokens: list[str], title: str) -> alt.Chart:
    """Create line chart showing surprisal per token position.

    Features:
        - Blue line showing surprisal trajectory
        - Red dots highlighting high surprisal tokens (>10 bits)
        - Dashed red horizontal threshold line at 10 bits
        - Interactive tooltips showing token and position

    Args:
        surprisal_values: List of surprisal values in bits
        tokens: List of token strings (same length as surprisal_values)
        title: Title for the chart

    Returns:
        Altair Chart object
    """
    if not surprisal_values:
        return alt.Chart(pl.DataFrame({"x": [0], "y": [0], "text": ["No data"]}).to_pandas()).mark_text().encode(
            text='text:N'
        ).properties(title=f'Token Surprisal: {title} (No Data)')

    # Ensure tokens list matches surprisal values
    if len(tokens) < len(surprisal_values):
        tokens = tokens + [""] * (len(surprisal_values) - len(tokens))
    elif len(tokens) > len(surprisal_values):
        tokens = tokens[:len(surprisal_values)]

    df = pl.DataFrame({
        "position": list(range(len(surprisal_values))),
        "surprisal": surprisal_values,
        "token": tokens
    })

    pandas_df = df.to_pandas()

    # Base line chart
    line = alt.Chart(pandas_df).mark_line(color='steelblue', strokeWidth=1.5).encode(
        x=alt.X('position:Q', title='Token Position'),
        y=alt.Y('surprisal:Q', title='Surprisal (bits)'),
        tooltip=[
            alt.Tooltip('position:Q', title='Position'),
            alt.Tooltip('token:N', title='Token'),
            alt.Tooltip('surprisal:Q', title='Surprisal', format='.2f')
        ]
    )

    # Points for all tokens (small, for hover)
    all_points = alt.Chart(pandas_df).mark_circle(
        color='steelblue',
        size=30,
        opacity=0.5
    ).encode(
        x='position:Q',
        y='surprisal:Q',
        tooltip=[
            alt.Tooltip('position:Q', title='Position'),
            alt.Tooltip('token:N', title='Token'),
            alt.Tooltip('surprisal:Q', title='Surprisal', format='.2f')
        ]
    )

    # Red dots for high surprisal tokens (>10 bits)
    high_surprisal_points = alt.Chart(pandas_df).mark_circle(
        color='#EE6677',
        size=80
    ).encode(
        x='position:Q',
        y='surprisal:Q',
        tooltip=[
            alt.Tooltip('position:Q', title='Position'),
            alt.Tooltip('token:N', title='Token'),
            alt.Tooltip('surprisal:Q', title='Surprisal', format='.2f')
        ]
    ).transform_filter(
        alt.datum.surprisal > 10
    )

    # Horizontal threshold line at 10 bits
    threshold_df = pl.DataFrame({"threshold": [10]}).to_pandas()
    threshold = alt.Chart(threshold_df).mark_rule(
        color='#EE6677',
        strokeDash=[5, 5],
        strokeWidth=1.5
    ).encode(
        y='threshold:Q'
    )

    # Combine all layers
    chart = (line + all_points + high_surprisal_points + threshold).properties(
        title=f'Token Surprisal: {title}',
        width='container',
        height=300
    ).configure_title(
        fontSize=16,
        anchor='start'
    ).interactive()

    return chart


def layer_activation_trajectory(norms: list[float], title: str) -> alt.Chart:
    """Create line chart showing activation L2 norms across layers.

    Tracks how the hidden state norm evolves through the transformer layers.
    Useful for detecting processing anomalies or saturation.

    Features:
        - Line with points at each layer
        - Interactive tooltips

    Args:
        norms: List of L2 norm values, one per layer
        title: Title for the chart

    Returns:
        Altair Chart object
    """
    if not norms:
        return alt.Chart(pl.DataFrame({"x": [0], "y": [0], "text": ["No data"]}).to_pandas()).mark_text().encode(
            text='text:N'
        ).properties(title=f'Activation Trajectory: {title} (No Data)')

    df = pl.DataFrame({
        "layer": list(range(len(norms))),
        "norm": norms
    })

    chart = alt.Chart(df.to_pandas()).mark_line(
        point=alt.OverlayMarkDef(color='#4477AA', size=60),
        color='#4477AA',
        strokeWidth=2
    ).encode(
        x=alt.X('layer:Q', title='Layer', axis=alt.Axis(tickMinStep=1)),
        y=alt.Y('norm:Q', title='L2 Norm'),
        tooltip=[
            alt.Tooltip('layer:Q', title='Layer'),
            alt.Tooltip('norm:Q', title='L2 Norm', format='.4f')
        ]
    ).properties(
        title=f'Activation Trajectory: {title}',
        width='container',
        height=300
    ).configure_title(
        fontSize=16,
        anchor='start'
    ).interactive()

    return chart


def genre_clustering(df: pl.DataFrame) -> alt.Chart:
    """Create 2D PCA projection of metric vectors colored by genre.

    Projects the high-dimensional metric space (perplexity, entropy, etc.)
    down to 2D for visualization. Shows explained variance in axis labels.

    Features:
        - PCA projection with explained variance percentages
        - Points colored by genre
        - Interactive tooltips with piece details

    Args:
        df: DataFrame with columns: filename, genre, perplexity, attention_entropy,
            surprisal_mean, token_count

    Returns:
        Altair Chart object
    """
    if df.height < 3:
        return alt.Chart(pl.DataFrame({"x": [0], "y": [0], "text": ["Need at least 3 pieces for PCA"]}).to_pandas()).mark_text().encode(
            text='text:N'
        ).properties(title='Genre Clustering (Insufficient Data)')

    # Select numeric columns for clustering
    feature_cols = ['perplexity', 'attention_entropy', 'surprisal_mean', 'token_count']

    # Filter out rows with missing data
    valid_df = df.drop_nulls(subset=feature_cols)

    if valid_df.height < 3:
        return alt.Chart(pl.DataFrame({"x": [0], "y": [0], "text": ["Insufficient valid data for PCA"]}).to_pandas()).mark_text().encode(
            text='text:N'
        ).properties(title='Genre Clustering (Insufficient Data)')

    features = valid_df.select(feature_cols).to_numpy()

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA to 2 components
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features_scaled)

    # Create DataFrame with coordinates
    cluster_df = valid_df.with_columns([
        pl.Series("pc1", coords[:, 0]),
        pl.Series("pc2", coords[:, 1]),
    ])

    # Get unique genres for palette
    genres = cluster_df["genre"].unique().to_list()
    palette_domain = [g for g in PALETTE.keys() if g in genres]
    palette_range = [PALETTE[g] for g in palette_domain]

    if "unknown" in genres and "unknown" not in palette_domain:
        palette_domain.append("unknown")
        palette_range.append("#999999")

    # Calculate explained variance percentages
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    chart = alt.Chart(cluster_df.to_pandas()).mark_circle(size=120).encode(
        x=alt.X('pc1:Q', title=f'PC1 ({var1:.1f}% variance)'),
        y=alt.Y('pc2:Q', title=f'PC2 ({var2:.1f}% variance)'),
        color=alt.Color(
            'genre:N',
            scale=alt.Scale(domain=palette_domain, range=palette_range),
            title='Genre',
            legend=alt.Legend(orient='right')
        ),
        tooltip=[
            alt.Tooltip('filename:N', title='Piece'),
            alt.Tooltip('genre:N', title='Genre'),
            alt.Tooltip('perplexity:Q', title='Perplexity', format='.2f'),
            alt.Tooltip('attention_entropy:Q', title='Entropy', format='.3f')
        ]
    ).properties(
        title='Genre Clustering (PCA Projection)',
        width='container',
        height=400
    ).configure_title(
        fontSize=16,
        anchor='start'
    ).interactive()

    return chart


def genre_distribution(df: pl.DataFrame, metric: str, title: str) -> alt.Chart:
    """Create overlapping area charts comparing metric distributions by genre.

    Shows how a metric (e.g., perplexity) is distributed across different genres,
    with semi-transparent overlapping areas for easy comparison.

    Features:
        - One color per genre (semi-transparent)
        - Step interpolation for histogram-like appearance
        - Overlapping for visibility of distribution differences

    Args:
        df: DataFrame with genre column and the specified metric column
        metric: Column name to plot (e.g., 'perplexity', 'attention_entropy')
        title: Human-readable name for the metric

    Returns:
        Altair Chart object
    """
    if df.height == 0:
        return alt.Chart(pl.DataFrame({"x": [0], "y": [0], "text": ["No data"]}).to_pandas()).mark_text().encode(
            text='text:N'
        ).properties(title=f'{title} Distribution (No Data)')

    # Get unique genres for palette
    genres = df["genre"].unique().to_list()
    palette_domain = [g for g in PALETTE.keys() if g in genres]
    palette_range = [PALETTE[g] for g in palette_domain]

    if "unknown" in genres and "unknown" not in palette_domain:
        palette_domain.append("unknown")
        palette_range.append("#999999")

    chart = alt.Chart(df.to_pandas()).mark_area(
        opacity=0.5,
        interpolate='step'
    ).encode(
        x=alt.X(
            f'{metric}:Q',
            bin=alt.Bin(maxbins=20),
            title=title
        ),
        y=alt.Y(
            'count()',
            stack=None,
            title='Count'
        ),
        color=alt.Color(
            'genre:N',
            scale=alt.Scale(domain=palette_domain, range=palette_range),
            title='Genre',
            legend=alt.Legend(orient='right')
        ),
        tooltip=[
            alt.Tooltip('genre:N', title='Genre'),
            alt.Tooltip('count()', title='Count')
        ]
    ).properties(
        title=f'{title} Distribution by Genre',
        width='container',
        height=300
    ).configure_title(
        fontSize=16,
        anchor='start'
    ).interactive()

    return chart


def save_chart(chart: alt.Chart, output_path: Path) -> None:
    """Save chart as self-contained HTML file.

    The HTML file includes all necessary Vega-Lite JS inline,
    so it can be viewed without an internet connection.

    Args:
        chart: Altair Chart object
        output_path: Path to save HTML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_path), format='html')
    print(f"  Saved: {output_path}")


def generate_piece_visualizations(metrics_path: Path, output_dir: Path) -> dict:
    """Generate all visualizations for a single piece.

    Creates:
        - Surprisal timeline chart
        - Activation trajectory chart
        - Attention entropy heatmap (if per-head data available)

    Args:
        metrics_path: Path to the metrics JSON file
        output_dir: Base output directory

    Returns:
        Dictionary mapping chart type to output path
    """
    try:
        data = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load {metrics_path}: {e}")
        return {}

    filename = data.get("filename", metrics_path.stem)
    piece_name = filename.replace(".abc", "").replace(".json", "")

    piece_output = output_dir / piece_name
    piece_output.mkdir(parents=True, exist_ok=True)

    generated = {}

    # Surprisal timeline
    surprisal_data = data.get("surprisal", {})
    if surprisal_data.get("per_token"):
        tokens = data.get("tokens", [""] * len(surprisal_data["per_token"]))
        chart = surprisal_timeline(surprisal_data["per_token"], tokens, piece_name)
        path = piece_output / "surprisal.html"
        save_chart(chart, path)
        generated["surprisal"] = str(path)

    # Layer activation trajectory
    activation_data = data.get("activations", {})
    if activation_data.get("per_layer_norm"):
        chart = layer_activation_trajectory(activation_data["per_layer_norm"], piece_name)
        path = piece_output / "activations.html"
        save_chart(chart, path)
        generated["activations"] = str(path)

    # Attention entropy heatmap
    attention_data = data.get("attention", {})
    if attention_data.get("per_head"):
        chart = attention_entropy_heatmap(attention_data["per_head"], piece_name)
        path = piece_output / "attention.html"
        save_chart(chart, path)
        generated["attention"] = str(path)

    if generated:
        print(f"Generated {len(generated)} visualization(s) for: {piece_name}")

    return generated


def generate_corpus_visualizations(metrics_dir: Path, output_dir: Path) -> dict:
    """Generate all corpus-level visualizations.

    Creates:
        - Perplexity comparison bar chart
        - Genre clustering PCA plot
        - Perplexity distribution by genre
        - Attention entropy distribution by genre

    Also generates per-piece visualizations for each metrics file.

    Args:
        metrics_dir: Directory containing metrics JSON files
        output_dir: Directory to save visualizations

    Returns:
        Manifest dictionary of all generated files
    """
    print(f"Loading metrics from: {metrics_dir}")
    df = load_metrics(metrics_dir)

    manifest = {"corpus": [], "pieces": {}}

    if df.height == 0:
        print("No metrics data found. Generating empty manifest.")
        return manifest

    print(f"Loaded {df.height} pieces across {df['genre'].n_unique()} genres")

    corpus_output = output_dir / "corpus"
    corpus_output.mkdir(parents=True, exist_ok=True)

    # Perplexity bar chart
    print("\nGenerating corpus visualizations...")
    chart = perplexity_bar_chart(df)
    path = corpus_output / "perplexity_comparison.html"
    save_chart(chart, path)
    manifest["corpus"].append("perplexity_comparison.html")

    # Genre clustering (only if enough data)
    if df.height >= 3:
        chart = genre_clustering(df)
        path = corpus_output / "genre_clustering.html"
        save_chart(chart, path)
        manifest["corpus"].append("genre_clustering.html")
    else:
        print("  Skipping genre clustering (need at least 3 pieces)")

    # Perplexity distribution
    chart = genre_distribution(df, "perplexity", "Perplexity")
    path = corpus_output / "perplexity_distribution.html"
    save_chart(chart, path)
    manifest["corpus"].append("perplexity_distribution.html")

    # Entropy distribution
    chart = genre_distribution(df, "attention_entropy", "Attention Entropy")
    path = corpus_output / "entropy_distribution.html"
    save_chart(chart, path)
    manifest["corpus"].append("entropy_distribution.html")

    # Generate per-piece visualizations
    print("\nGenerating per-piece visualizations...")
    for metrics_path in sorted(metrics_dir.glob("*.json")):
        # Skip non-metrics files
        if metrics_path.name in ["manifest.json", "interpretations.json"]:
            continue

        piece_files = generate_piece_visualizations(metrics_path, output_dir)
        if piece_files:
            piece_name = metrics_path.stem.replace(".abc", "")
            manifest["pieces"][piece_name] = list(piece_files.keys())

    return manifest


def create_manifest(output_dir: Path, manifest_data: dict | None = None) -> None:
    """Create or update manifest JSON listing all visualizations.

    The manifest provides a machine-readable index of all generated
    visualization files for programmatic access.

    Args:
        output_dir: Visualizations directory
        manifest_data: Pre-computed manifest (if None, scans directory)
    """
    if manifest_data is None:
        manifest_data = {"pieces": {}, "corpus": []}

        # Scan corpus directory
        corpus_dir = output_dir / "corpus"
        if corpus_dir.exists():
            manifest_data["corpus"] = sorted([f.name for f in corpus_dir.glob("*.html")])

        # Scan per-piece directories
        for piece_dir in sorted(output_dir.iterdir()):
            if piece_dir.is_dir() and piece_dir.name != "corpus":
                html_files = sorted([f.name for f in piece_dir.glob("*.html")])
                if html_files:
                    manifest_data["pieces"][piece_dir.name] = html_files

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2, sort_keys=True))
    print(f"\nCreated manifest: {manifest_path}")


def main():
    """Main entry point for visualization generation."""
    parser = argparse.ArgumentParser(
        description="Generate interpretability visualizations from metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all visualizations
    python src/visualizations.py data/metrics/

    # Generate to specific output directory
    python src/visualizations.py data/metrics/ --output docs/visualizations/

    # Generate for a single piece
    python src/visualizations.py data/metrics/ --piece traditional_001
        """
    )
    parser.add_argument(
        "metrics_dir",
        type=Path,
        help="Directory containing metrics JSON files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/visualizations"),
        help="Output directory for visualizations (default: docs/visualizations)"
    )
    parser.add_argument(
        "--piece",
        help="Generate visualizations for a specific piece only"
    )

    args = parser.parse_args()

    if not args.metrics_dir.exists():
        print(f"Error: Metrics directory not found: {args.metrics_dir}")
        return 1

    print(f"Visualization Suite")
    print(f"==================")
    print(f"Metrics: {args.metrics_dir}")
    print(f"Output:  {args.output}")

    if args.piece:
        # Generate for single piece
        metrics_path = args.metrics_dir / f"{args.piece}.abc.json"
        if not metrics_path.exists():
            # Try without .abc extension
            metrics_path = args.metrics_dir / f"{args.piece}.json"

        if metrics_path.exists():
            generate_piece_visualizations(metrics_path, args.output)
            create_manifest(args.output)
        else:
            print(f"Error: Metrics file not found: {metrics_path}")
            return 1
    else:
        # Generate all visualizations
        manifest = generate_corpus_visualizations(args.metrics_dir, args.output)
        create_manifest(args.output, manifest)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
