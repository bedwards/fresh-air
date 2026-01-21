#!/usr/bin/env python3
"""Interpretability metrics analyzer for ABC music files.

This module interprets raw metrics from model_harness and generates
human-readable descriptions and classifications.

Usage:
    python src/metrics_analyzer.py data/metrics/
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
        data = json.loads(path.read_text())

        # Extract genre from filename
        filename = data.get("filename", path.stem)
        genre = filename.split("_")[0] if "_" in filename else "unknown"

        record = {
            "filename": filename,
            "genre": genre,
            "perplexity_overall": data.get("perplexity", {}).get("overall", 0),
            "perplexity_max": data.get("perplexity", {}).get("max", 0),
            "perplexity_min": data.get("perplexity", {}).get("min", 0),
            "attention_mean_entropy": data.get("attention", {}).get("mean_entropy", 0),
            "surprisal_mean": data.get("surprisal", {}).get("mean", 0),
            "token_count": data.get("token_count", 0),
        }
        records.append(record)

    return pl.DataFrame(records)


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
    parser = argparse.ArgumentParser(description="Analyze interpretability metrics")
    parser.add_argument("metrics_dir", type=Path, help="Directory containing metrics JSON files")
    parser.add_argument("--output", type=Path, help="Output directory for interpretations")
    parser.add_argument("--compare", nargs=2, metavar=("GENRE1", "GENRE2"), help="Compare two genres")

    args = parser.parse_args()

    if args.compare:
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
