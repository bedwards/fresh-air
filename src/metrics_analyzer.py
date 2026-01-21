#!/usr/bin/env python3
"""Interpretability metrics analyzer for ABC music files.

This module interprets raw metrics from model_harness and generates
human-readable descriptions and classifications.

Usage:
    python src/metrics_analyzer.py data/metrics/
    python src/metrics_analyzer.py data/metrics/ --update-comparative
    python src/metrics_analyzer.py data/metrics/ --summary
    python src/metrics_analyzer.py --compare traditional experimental
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


Classification = Literal[
    "syntactically_valid_semantically_coherent",
    "syntactically_valid_semantically_chaotic",
    "syntactically_unusual_semantically_coherent",
    "syntactically_unusual_semantically_chaotic",
    "statistically_normal",
    "statistically_anomalous"
]

# Threshold for outlier detection (standard deviations from mean)
OUTLIER_THRESHOLD = 2.0


@dataclass
class MetricInterpretation:
    """Human-readable interpretation of metrics."""
    filename: str
    genre: str
    classification: Classification
    perplexity_assessment: str
    attention_assessment: str
    surprisal_assessment: str
    activation_assessment: str
    overall_interpretation: str
    key_findings: list[str]


def load_metrics(metrics_dir: Path) -> pl.DataFrame:
    """Load all metrics JSON files into a Polars DataFrame."""
    metrics_files = sorted(metrics_dir.glob("*.json"))

    records = []
    for path in metrics_files:
        if path.name == ".gitkeep":
            continue
        data = json.loads(path.read_text())

        # Extract genre from filename
        filename = data.get("filename", path.stem)
        genre = filename.split("_")[0] if "_" in filename else "unknown"

        # Get attention mean_entropy, defaulting to None if not available
        attention_entropy = data.get("attention", {}).get("mean_entropy")
        # If attention entropy is not available, use surprisal mean as proxy
        entropy_value = attention_entropy if attention_entropy is not None else data.get("surprisal", {}).get("mean", 0)

        record = {
            "filename": filename,
            "genre": genre,
            "perplexity_overall": data.get("perplexity", {}).get("overall", 0),
            "perplexity_max": data.get("perplexity", {}).get("max", 0),
            "perplexity_min": data.get("perplexity", {}).get("min", 0),
            "attention_mean_entropy": entropy_value,
            "surprisal_mean": data.get("surprisal", {}).get("mean", 0),
            "token_count": data.get("token_count", 0),
        }
        records.append(record)

    return pl.DataFrame(records)


def load_raw_metrics(metrics_dir: Path) -> list[dict]:
    """Load all metrics JSON files as raw dictionaries."""
    metrics_files = sorted(metrics_dir.glob("*.json"))

    results = []
    for path in metrics_files:
        if path.name == ".gitkeep":
            continue
        data = json.loads(path.read_text())
        data["_path"] = str(path)
        results.append(data)

    return results


def compute_baseline(metrics_dir: Path, baseline_genre: str = "traditional") -> dict:
    """Compute baseline statistics from traditional pieces.

    Args:
        metrics_dir: Path to directory containing metrics JSON files
        baseline_genre: Genre to use as baseline (default: "traditional")

    Returns:
        Dictionary containing mean and std for perplexity, entropy, and surprisal
    """
    metrics_files = sorted(metrics_dir.glob("*.json"))

    perplexities = []
    entropies = []
    surprisals = []

    for path in metrics_files:
        if path.name == ".gitkeep":
            continue
        data = json.loads(path.read_text())
        filename = data.get("filename", path.stem)
        genre = filename.split("_")[0] if "_" in filename else "unknown"

        if genre == baseline_genre:
            perplexity = data.get("perplexity", {}).get("overall", 0)
            # Use surprisal mean as proxy for entropy when attention is not available
            attention_entropy = data.get("attention", {}).get("mean_entropy")
            entropy = attention_entropy if attention_entropy is not None else data.get("surprisal", {}).get("mean", 0)
            surprisal = data.get("surprisal", {}).get("mean", 0)

            if perplexity > 0:
                perplexities.append(perplexity)
            if entropy > 0:
                entropies.append(entropy)
            if surprisal > 0:
                surprisals.append(surprisal)

    # If no baseline pieces found, use all pieces
    if not perplexities:
        for path in metrics_files:
            if path.name == ".gitkeep":
                continue
            data = json.loads(path.read_text())
            perplexity = data.get("perplexity", {}).get("overall", 0)
            attention_entropy = data.get("attention", {}).get("mean_entropy")
            entropy = attention_entropy if attention_entropy is not None else data.get("surprisal", {}).get("mean", 0)
            surprisal = data.get("surprisal", {}).get("mean", 0)

            if perplexity > 0:
                perplexities.append(perplexity)
            if entropy > 0:
                entropies.append(entropy)
            if surprisal > 0:
                surprisals.append(surprisal)

    def safe_mean(values):
        return sum(values) / len(values) if values else 0.0

    def safe_std(values):
        if len(values) < 2:
            return 1.0  # Avoid division by zero
        mean = safe_mean(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5 if variance > 0 else 1.0

    return {
        "perplexity_mean": safe_mean(perplexities),
        "perplexity_std": safe_std(perplexities),
        "entropy_mean": safe_mean(entropies),
        "entropy_std": safe_std(entropies),
        "surprisal_mean": safe_mean(surprisals),
        "surprisal_std": safe_std(surprisals),
    }


def compute_zscores_for_metrics(metrics: dict, baseline: dict) -> dict:
    """Compute z-scores for a single piece's metrics relative to baseline.

    Args:
        metrics: Dictionary containing piece metrics (from JSON file)
        baseline: Dictionary containing baseline statistics

    Returns:
        Dictionary containing z-scores for perplexity, entropy, and surprisal
    """
    perplexity = metrics.get("perplexity", {}).get("overall", 0)
    attention_entropy = metrics.get("attention", {}).get("mean_entropy")
    entropy = attention_entropy if attention_entropy is not None else metrics.get("surprisal", {}).get("mean", 0)
    surprisal = metrics.get("surprisal", {}).get("mean", 0)

    # Compute z-scores, avoiding division by zero
    perplexity_std = baseline.get("perplexity_std", 1.0)
    entropy_std = baseline.get("entropy_std", 1.0)
    surprisal_std = baseline.get("surprisal_std", 1.0)

    return {
        "perplexity_zscore": round((perplexity - baseline["perplexity_mean"]) / max(perplexity_std, 0.001), 4),
        "entropy_zscore": round((entropy - baseline["entropy_mean"]) / max(entropy_std, 0.001), 4),
        "surprisal_zscore": round((surprisal - baseline["surprisal_mean"]) / max(surprisal_std, 0.001), 4),
    }


def classify_piece_from_zscores(zscores: dict) -> Classification:
    """Classify a piece based on its z-score profile.

    Classification logic:
    - High perplexity (>2 std) = syntactically unusual (unfamiliar token sequences)
    - High entropy (>2 std) = semantically chaotic (diffuse attention, no clear structure)
    - Both high = outside learned patterns
    - Both low (<1 std) = statistically normal

    Args:
        zscores: Dictionary containing perplexity_zscore and entropy_zscore

    Returns:
        Classification literal string
    """
    perp_z = zscores.get("perplexity_zscore", 0)
    ent_z = zscores.get("entropy_zscore", 0)

    # Check for extreme outliers first
    if abs(perp_z) > 3 or abs(ent_z) > 3:
        return "statistically_anomalous"

    # High perplexity and high entropy = chaotic
    if perp_z > OUTLIER_THRESHOLD and ent_z > OUTLIER_THRESHOLD:
        return "syntactically_unusual_semantically_chaotic"

    # High perplexity but normal entropy = unusual syntax
    if perp_z > OUTLIER_THRESHOLD and ent_z <= OUTLIER_THRESHOLD:
        return "syntactically_unusual_semantically_coherent"

    # Normal perplexity but high entropy = semantically confused
    if perp_z <= OUTLIER_THRESHOLD and ent_z > OUTLIER_THRESHOLD:
        return "syntactically_valid_semantically_chaotic"

    # Normal perplexity and entropy = well-understood
    if abs(perp_z) <= 1 and abs(ent_z) <= 1:
        return "statistically_normal"

    return "syntactically_valid_semantically_coherent"


def generate_classification_interpretation(classification: Classification, zscores: dict) -> str:
    """Generate human-readable interpretation of a piece's classification.

    Args:
        classification: The piece classification
        zscores: Dictionary containing z-scores

    Returns:
        Prose description of what the metrics indicate
    """
    interpretations = {
        "syntactically_valid_semantically_coherent": (
            "The model processes this piece comfortably, recognizing familiar patterns. "
            "Both token sequences and structural relationships align with training data."
        ),
        "syntactically_valid_semantically_chaotic": (
            "The model recognizes valid ABC syntax but fails to identify coherent musical relationships. "
            "The piece may contain unexpected harmonic or melodic progressions."
        ),
        "syntactically_unusual_semantically_coherent": (
            "The token sequences are unusual but the model still identifies some structural coherence. "
            "This may indicate novel but valid musical patterns."
        ),
        "syntactically_unusual_semantically_chaotic": (
            "Both syntax and semantics are unfamiliar. The model is operating outside its learned patterns. "
            "This piece exhibits characteristics the model has rarely or never encountered."
        ),
        "statistically_normal": (
            "This piece falls well within the normal range of the model's experience with musical notation. "
            "All metrics indicate familiar, predictable patterns."
        ),
        "statistically_anomalous": (
            "This piece exhibits extreme statistical characteristics that warrant investigation. "
            "Metrics deviate significantly (>3 standard deviations) from baseline."
        ),
    }

    base_interpretation = interpretations.get(classification, "Classification pending.")

    # Add specific z-score context
    perp_z = zscores.get("perplexity_zscore", 0)
    ent_z = zscores.get("entropy_zscore", 0)
    surp_z = zscores.get("surprisal_zscore", 0)

    if abs(perp_z) > 2:
        direction = "above" if perp_z > 0 else "below"
        base_interpretation += f" Perplexity is {abs(perp_z):.1f} standard deviations {direction} baseline."

    if abs(ent_z) > 2:
        direction = "above" if ent_z > 0 else "below"
        base_interpretation += f" Entropy is {abs(ent_z):.1f} standard deviations {direction} baseline."

    return base_interpretation


def identify_outlier_flags(zscores: dict) -> list[str]:
    """Identify which metrics are outliers (>2 standard deviations).

    Args:
        zscores: Dictionary containing z-scores

    Returns:
        List of metric names that are outliers
    """
    outliers = []

    if abs(zscores.get("perplexity_zscore", 0)) > OUTLIER_THRESHOLD:
        outliers.append("perplexity")

    if abs(zscores.get("entropy_zscore", 0)) > OUTLIER_THRESHOLD:
        outliers.append("entropy")

    if abs(zscores.get("surprisal_zscore", 0)) > OUTLIER_THRESHOLD:
        outliers.append("surprisal")

    return outliers


def update_metrics_with_comparative(metrics_dir: Path, baseline_genre: str = "traditional") -> int:
    """Update all metrics JSON files with comparative analysis data.

    Args:
        metrics_dir: Path to directory containing metrics JSON files
        baseline_genre: Genre to use as baseline

    Returns:
        Number of files updated
    """
    # First compute baseline from traditional pieces
    baseline = compute_baseline(metrics_dir, baseline_genre)

    print(f"Baseline statistics (from {baseline_genre} pieces):")
    print(f"  Perplexity: mean={baseline['perplexity_mean']:.4f}, std={baseline['perplexity_std']:.4f}")
    print(f"  Entropy: mean={baseline['entropy_mean']:.4f}, std={baseline['entropy_std']:.4f}")
    print(f"  Surprisal: mean={baseline['surprisal_mean']:.4f}, std={baseline['surprisal_std']:.4f}")
    print()

    # Load and update each metrics file
    metrics_files = sorted(metrics_dir.glob("*.json"))
    updated_count = 0

    for path in metrics_files:
        if path.name == ".gitkeep":
            continue

        data = json.loads(path.read_text())

        # Compute z-scores for this piece
        zscores = compute_zscores_for_metrics(data, baseline)

        # Classify the piece
        classification = classify_piece_from_zscores(zscores)

        # Generate interpretation
        interpretation = generate_classification_interpretation(classification, zscores)

        # Identify outliers
        outliers = identify_outlier_flags(zscores)

        # Build comparative data structure
        comparative = {
            "baseline": f"{baseline_genre}_mean",
            "perplexity_zscore": zscores["perplexity_zscore"],
            "entropy_zscore": zscores["entropy_zscore"],
            "surprisal_zscore": zscores["surprisal_zscore"],
            "outlier_flags": outliers,
            "classification": classification,
            "interpretation": interpretation,
        }

        # Update the metrics data
        data["comparative"] = comparative

        # Write back to file
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        updated_count += 1

        filename = data.get("filename", path.stem)
        print(f"Updated: {filename}")
        print(f"  Classification: {classification}")
        print(f"  Z-scores: perplexity={zscores['perplexity_zscore']:.2f}, entropy={zscores['entropy_zscore']:.2f}, surprisal={zscores['surprisal_zscore']:.2f}")
        if outliers:
            print(f"  Outliers: {', '.join(outliers)}")
        print()

    return updated_count


def print_summary(metrics_dir: Path) -> None:
    """Print a summary of comparative analysis across all pieces.

    Args:
        metrics_dir: Path to directory containing metrics JSON files
    """
    metrics_files = sorted(metrics_dir.glob("*.json"))

    # Group by classification
    classifications: dict[str, list[str]] = {}
    outlier_counts: dict[str, int] = {"perplexity": 0, "entropy": 0, "surprisal": 0}
    genre_stats: dict[str, list[dict]] = {}

    for path in metrics_files:
        if path.name == ".gitkeep":
            continue

        data = json.loads(path.read_text())
        filename = data.get("filename", path.stem)
        genre = filename.split("_")[0] if "_" in filename else "unknown"

        comparative = data.get("comparative")
        if not comparative:
            print(f"Warning: {filename} has no comparative data. Run with --update-comparative first.")
            continue

        classification = comparative.get("classification", "unknown")
        if classification not in classifications:
            classifications[classification] = []
        classifications[classification].append(filename)

        # Count outliers
        for outlier in comparative.get("outlier_flags", []):
            outlier_counts[outlier] = outlier_counts.get(outlier, 0) + 1

        # Collect genre stats
        if genre not in genre_stats:
            genre_stats[genre] = []
        genre_stats[genre].append({
            "perplexity_zscore": comparative.get("perplexity_zscore", 0),
            "entropy_zscore": comparative.get("entropy_zscore", 0),
            "surprisal_zscore": comparative.get("surprisal_zscore", 0),
        })

    print("=" * 60)
    print("COMPARATIVE METRICS SUMMARY")
    print("=" * 60)
    print()

    # Classification breakdown
    print("Classification Distribution:")
    print("-" * 40)
    for classification, files in sorted(classifications.items()):
        print(f"  {classification}: {len(files)} pieces")
        for f in files[:3]:  # Show first 3
            print(f"    - {f}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")
    print()

    # Outlier counts
    print("Outlier Counts (>2 standard deviations):")
    print("-" * 40)
    for metric, count in sorted(outlier_counts.items()):
        print(f"  {metric}: {count} pieces")
    print()

    # Genre comparison
    print("Genre Statistics (mean z-scores):")
    print("-" * 40)
    for genre, stats in sorted(genre_stats.items()):
        if stats:
            mean_perp = sum(s["perplexity_zscore"] for s in stats) / len(stats)
            mean_ent = sum(s["entropy_zscore"] for s in stats) / len(stats)
            mean_surp = sum(s["surprisal_zscore"] for s in stats) / len(stats)
            print(f"  {genre} ({len(stats)} pieces):")
            print(f"    perplexity: {mean_perp:+.2f}, entropy: {mean_ent:+.2f}, surprisal: {mean_surp:+.2f}")
    print()


def compute_baseline_statistics(df: pl.DataFrame, genre: str = "traditional") -> dict:
    """Compute baseline statistics from traditional music pieces."""
    baseline = df.filter(pl.col("genre") == genre)

    if baseline.height == 0:
        # Use all data if no traditional pieces
        baseline = df

    return {
        "perplexity_mean": baseline["perplexity_overall"].mean(),
        "perplexity_std": baseline["perplexity_overall"].std(),
        "entropy_mean": baseline["attention_mean_entropy"].mean(),
        "entropy_std": baseline["attention_mean_entropy"].std(),
        "surprisal_mean": baseline["surprisal_mean"].mean(),
        "surprisal_std": baseline["surprisal_mean"].std(),
    }


def compute_zscores(df: pl.DataFrame, baseline: dict) -> pl.DataFrame:
    """Add z-score columns relative to baseline."""
    return df.with_columns([
        ((pl.col("perplexity_overall") - baseline["perplexity_mean"]) / baseline["perplexity_std"])
        .alias("perplexity_zscore"),
        ((pl.col("attention_mean_entropy") - baseline["entropy_mean"]) / baseline["entropy_std"])
        .alias("entropy_zscore"),
        ((pl.col("surprisal_mean") - baseline["surprisal_mean"]) / baseline["surprisal_std"])
        .alias("surprisal_zscore"),
    ])


def classify_piece(row: dict) -> Classification:
    """Classify a piece based on its metric profile."""
    perp_z = row.get("perplexity_zscore", 0)
    ent_z = row.get("entropy_zscore", 0)

    # High perplexity and high entropy = chaotic
    if perp_z > 2 and ent_z > 2:
        return "syntactically_unusual_semantically_chaotic"

    # High perplexity but normal entropy = unusual syntax
    if perp_z > 2 and ent_z <= 2:
        return "syntactically_unusual_semantically_coherent"

    # Normal perplexity but high entropy = semantically confused
    if perp_z <= 2 and ent_z > 2:
        return "syntactically_valid_semantically_chaotic"

    # Normal perplexity and entropy = well-understood
    if perp_z <= 1 and ent_z <= 1:
        return "statistically_normal"

    return "syntactically_valid_semantically_coherent"


def assess_perplexity(overall: float, zscore: float) -> str:
    """Generate human-readable perplexity assessment."""
    if overall < 50:
        return f"Very low perplexity ({overall:.1f}) indicates highly predictable token sequences."
    elif overall < 100:
        return f"Normal perplexity ({overall:.1f}) suggests the model recognizes familiar patterns."
    elif overall < 200:
        return f"Elevated perplexity ({overall:.1f}) indicates some unfamiliar sequences."
    elif overall < 400:
        return f"High perplexity ({overall:.1f}, z={zscore:.1f}σ) shows significant uncertainty."
    else:
        return f"Very high perplexity ({overall:.1f}, z={zscore:.1f}σ) indicates severely unfamiliar material."


def assess_attention(mean_entropy: float, zscore: float) -> str:
    """Generate human-readable attention entropy assessment."""
    if mean_entropy < 2.0:
        return f"Low attention entropy ({mean_entropy:.2f} bits) indicates focused, directed attention."
    elif mean_entropy < 3.5:
        return f"Moderate attention entropy ({mean_entropy:.2f} bits) shows normal attention patterns."
    elif mean_entropy < 4.5:
        return f"Elevated attention entropy ({mean_entropy:.2f} bits) suggests diffuse attention."
    else:
        return f"High attention entropy ({mean_entropy:.2f} bits, z={zscore:.1f}σ) indicates the model cannot identify relevant context."


def assess_surprisal(mean_surprisal: float, zscore: float) -> str:
    """Generate human-readable surprisal assessment."""
    if mean_surprisal < 4:
        return f"Low mean surprisal ({mean_surprisal:.2f} bits) indicates predictable tokens."
    elif mean_surprisal < 6:
        return f"Normal surprisal ({mean_surprisal:.2f} bits) shows expected variability."
    elif mean_surprisal < 10:
        return f"Elevated surprisal ({mean_surprisal:.2f} bits) indicates frequent unexpected tokens."
    else:
        return f"Very high surprisal ({mean_surprisal:.2f} bits, z={zscore:.1f}σ) shows consistent unexpectedness."


def generate_interpretation(row: dict) -> MetricInterpretation:
    """Generate complete interpretation for a piece."""
    classification = classify_piece(row)

    perp_assessment = assess_perplexity(
        row["perplexity_overall"],
        row.get("perplexity_zscore", 0)
    )
    att_assessment = assess_attention(
        row["attention_mean_entropy"],
        row.get("entropy_zscore", 0)
    )
    surp_assessment = assess_surprisal(
        row["surprisal_mean"],
        row.get("surprisal_zscore", 0)
    )

    # Placeholder for activation assessment
    act_assessment = "Activation patterns require further analysis."

    # Generate overall interpretation based on classification
    interpretations = {
        "syntactically_valid_semantically_coherent": (
            "The model processes this piece comfortably, recognizing both "
            "syntactic patterns and semantic relationships."
        ),
        "syntactically_valid_semantically_chaotic": (
            "The model recognizes valid ABC syntax but fails to identify "
            "coherent musical relationships."
        ),
        "syntactically_unusual_semantically_coherent": (
            "The token sequences are unusual but the model still identifies "
            "some structural coherence."
        ),
        "syntactically_unusual_semantically_chaotic": (
            "Both syntax and semantics are unfamiliar. The model is operating "
            "outside its learned patterns."
        ),
        "statistically_normal": (
            "This piece falls within the normal range of the model's experience "
            "with musical notation."
        ),
        "statistically_anomalous": (
            "This piece exhibits statistically unusual characteristics that "
            "warrant further investigation."
        ),
    }

    overall = interpretations.get(classification, "Classification pending.")

    # Extract key findings
    findings = []
    if row.get("perplexity_zscore", 0) > 2:
        findings.append("Perplexity significantly above baseline")
    if row.get("entropy_zscore", 0) > 2:
        findings.append("Attention entropy indicates confusion")
    if row.get("surprisal_zscore", 0) > 2:
        findings.append("High surprisal tokens detected")
    if not findings:
        findings.append("Metrics within normal range")

    return MetricInterpretation(
        filename=row["filename"],
        genre=row["genre"],
        classification=classification,
        perplexity_assessment=perp_assessment,
        attention_assessment=att_assessment,
        surprisal_assessment=surp_assessment,
        activation_assessment=act_assessment,
        overall_interpretation=overall,
        key_findings=findings
    )


def analyze_all(metrics_dir: Path, output_dir: Path | None = None) -> list[MetricInterpretation]:
    """Analyze all metrics and generate interpretations."""
    df = load_metrics(metrics_dir)

    if df.height == 0:
        print("No metrics files found")
        return []

    baseline = compute_baseline_statistics(df)
    df = compute_zscores(df, baseline)

    interpretations = []
    for row in df.to_dicts():
        interp = generate_interpretation(row)
        interpretations.append(interp)

        print(f"\n=== {interp.filename} ({interp.genre}) ===")
        print(f"Classification: {interp.classification}")
        print(f"Perplexity: {interp.perplexity_assessment}")
        print(f"Attention: {interp.attention_assessment}")
        print(f"Overall: {interp.overall_interpretation}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "interpretations.json"
        output_path.write_text(
            json.dumps([asdict(i) for i in interpretations], indent=2),
            encoding='utf-8'
        )
        print(f"\nSaved interpretations to: {output_path}")

    return interpretations


def compare_genres(df: pl.DataFrame, genre1: str, genre2: str) -> dict:
    """Compare metrics between two genres."""
    g1 = df.filter(pl.col("genre") == genre1)
    g2 = df.filter(pl.col("genre") == genre2)

    comparison = {
        "genre1": genre1,
        "genre2": genre2,
        "perplexity_diff": g2["perplexity_overall"].mean() - g1["perplexity_overall"].mean(),
        "entropy_diff": g2["attention_mean_entropy"].mean() - g1["attention_mean_entropy"].mean(),
        "surprisal_diff": g2["surprisal_mean"].mean() - g1["surprisal_mean"].mean(),
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Analyze interpretability metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run basic analysis
    python src/metrics_analyzer.py data/metrics/

    # Update all metrics files with comparative analysis
    python src/metrics_analyzer.py data/metrics/ --update-comparative

    # Show summary of comparative analysis
    python src/metrics_analyzer.py data/metrics/ --summary

    # Compare two genres
    python src/metrics_analyzer.py data/metrics/ --compare traditional experimental
        """
    )
    parser.add_argument("metrics_dir", type=Path, help="Directory containing metrics JSON files")
    parser.add_argument("--output", type=Path, help="Output directory for interpretations")
    parser.add_argument("--compare", nargs=2, metavar=("GENRE1", "GENRE2"), help="Compare two genres")
    parser.add_argument(
        "--update-comparative",
        action="store_true",
        help="Update all metrics files with comparative z-scores and classifications"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of comparative analysis across all pieces"
    )
    parser.add_argument(
        "--baseline-genre",
        type=str,
        default="traditional",
        help="Genre to use as baseline for comparative analysis (default: traditional)"
    )

    args = parser.parse_args()

    if args.update_comparative:
        updated = update_metrics_with_comparative(args.metrics_dir, args.baseline_genre)
        print(f"\nUpdated {updated} metrics files with comparative data.")
    elif args.summary:
        print_summary(args.metrics_dir)
    elif args.compare:
        df = load_metrics(args.metrics_dir)
        comparison = compare_genres(df, args.compare[0], args.compare[1])
        print(f"\nComparison: {comparison['genre1']} vs {comparison['genre2']}")
        print(f"  Perplexity difference: {comparison['perplexity_diff']:.2f}")
        print(f"  Entropy difference: {comparison['entropy_diff']:.2f}")
        print(f"  Surprisal difference: {comparison['surprisal_diff']:.2f}")
    else:
        analyze_all(args.metrics_dir, args.output)


if __name__ == "__main__":
    main()
