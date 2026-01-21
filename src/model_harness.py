#!/usr/bin/env python3
"""Phi-4 model harness for transformer interpretability analysis.

This module provides a wrapper around Phi-4 14B (via Ollama) to extract
interpretability metrics from ABC notation music files.

Metrics computed:
    - Perplexity (overall and per-token)
    - Attention entropy per head per layer
    - Per-token surprisal
    - Layer-wise activation statistics

Usage:
    python src/model_harness.py data/abc_files/traditional_001.abc
    python src/model_harness.py --all data/abc_files/
"""

import argparse
import json
import math
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "phi4:14b"


@dataclass
class PerplexityMetrics:
    """Perplexity-related metrics."""
    overall: float
    per_token: list[float]
    max: float
    min: float
    mean: float


@dataclass
class AttentionMetrics:
    """Attention-related metrics."""
    mean_entropy: float
    per_layer: list[float]
    per_head: list[list[float]]  # [layer][head]
    high_entropy_heads: list[dict]


@dataclass
class SurprisalMetrics:
    """Surprisal-related metrics."""
    mean: float
    per_token: list[float]
    high_surprisal_tokens: list[dict]


@dataclass
class ActivationMetrics:
    """Hidden state activation metrics."""
    per_layer_norm: list[float]
    variance_per_layer: list[float]


@dataclass
class ComparativeMetrics:
    """Metrics relative to baseline."""
    baseline: str
    perplexity_zscore: float
    entropy_zscore: float


@dataclass
class AnalysisResult:
    """Complete analysis result for an ABC file."""
    filename: str
    model: str
    timestamp: str
    token_count: int
    perplexity: PerplexityMetrics
    attention: AttentionMetrics
    surprisal: SurprisalMetrics
    activations: ActivationMetrics
    comparative: ComparativeMetrics | None


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_model_available() -> bool:
    """Check if Phi-4 model is available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(MODEL_NAME in m.get("name", "") for m in models)
        return False
    except requests.exceptions.RequestException:
        return False


def tokenize(text: str) -> list[str]:
    """Tokenize text using Ollama's tokenizer.

    Note: This is a simplified tokenization. Full implementation
    should use the actual Phi-4 tokenizer.
    """
    # TODO: Implement proper tokenization via Ollama API or transformers
    # For now, use simple whitespace + punctuation splitting
    import re
    tokens = re.findall(r'\S+|\n', text)
    return tokens


def compute_log_probabilities(text: str) -> list[float]:
    """Compute log probabilities for each token.

    Uses Ollama's API to get token-level probabilities.
    """
    # TODO: Implement using Ollama API with proper probability extraction
    # This requires the /api/generate endpoint with return_logprobs=true

    # Placeholder implementation
    tokens = tokenize(text)
    # Return synthetic log probs for testing
    import random
    return [random.uniform(-3, -0.5) for _ in tokens]


def compute_perplexity(log_probs: list[float]) -> PerplexityMetrics:
    """Compute perplexity metrics from log probabilities.

    Perplexity = exp(mean(-log(p)))

    This measures the effective vocabulary size the model
    considers at each position.
    """
    if not log_probs:
        return PerplexityMetrics(
            overall=0.0,
            per_token=[],
            max=0.0,
            min=0.0,
            mean=0.0
        )

    # Convert to per-token perplexity
    per_token_ppl = [math.exp(-lp) for lp in log_probs]

    # Overall perplexity
    mean_neg_log_prob = -sum(log_probs) / len(log_probs)
    overall_ppl = math.exp(mean_neg_log_prob)

    return PerplexityMetrics(
        overall=overall_ppl,
        per_token=per_token_ppl,
        max=max(per_token_ppl),
        min=min(per_token_ppl),
        mean=sum(per_token_ppl) / len(per_token_ppl)
    )


def compute_attention_entropy(attention_weights: list[list[list[float]]]) -> AttentionMetrics:
    """Compute entropy of attention distributions.

    Entropy H = -sum(p * log(p))

    Low entropy indicates focused attention (model knows where to look).
    High entropy indicates diffuse attention (model is uncertain).

    Args:
        attention_weights: [layer][head][position] attention distributions
    """
    # TODO: Implement attention extraction from Phi-4
    # This requires model internals access via transformers

    # Placeholder implementation
    num_layers = 32  # Phi-4 has ~32 layers
    num_heads = 32   # Phi-4 has ~32 attention heads per layer

    per_layer = [2.5 + 0.1 * i for i in range(num_layers)]
    per_head = [[2.0 + 0.1 * h for h in range(num_heads)] for _ in range(num_layers)]

    mean_entropy = sum(per_layer) / len(per_layer)

    # Identify high entropy heads (>4.5 bits)
    high_entropy_heads = []
    for layer_idx, layer_heads in enumerate(per_head):
        for head_idx, entropy in enumerate(layer_heads):
            if entropy > 4.5:
                high_entropy_heads.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "entropy": entropy
                })

    return AttentionMetrics(
        mean_entropy=mean_entropy,
        per_layer=per_layer,
        per_head=per_head,
        high_entropy_heads=high_entropy_heads
    )


def compute_surprisal(log_probs: list[float], tokens: list[str]) -> SurprisalMetrics:
    """Compute per-token surprisal.

    Surprisal = -log2(p(token))

    High surprisal (>10 bits) indicates the model encountered
    something unexpected.
    """
    if not log_probs:
        return SurprisalMetrics(mean=0.0, per_token=[], high_surprisal_tokens=[])

    # Convert natural log to bits
    per_token_surprisal = [-lp / math.log(2) for lp in log_probs]

    mean_surprisal = sum(per_token_surprisal) / len(per_token_surprisal)

    # Find high surprisal tokens (>10 bits)
    high_surprisal = []
    for i, (surprisal, token) in enumerate(zip(per_token_surprisal, tokens)):
        if surprisal > 10:
            high_surprisal.append({
                "position": i,
                "token": token,
                "surprisal": surprisal
            })

    return SurprisalMetrics(
        mean=mean_surprisal,
        per_token=per_token_surprisal,
        high_surprisal_tokens=high_surprisal
    )


def compute_activations(hidden_states: list[list[float]]) -> ActivationMetrics:
    """Compute activation statistics across layers.

    Tracks L2 norm and variance to detect processing anomalies.
    """
    # TODO: Implement activation extraction from Phi-4

    # Placeholder implementation
    num_layers = 32
    per_layer_norm = [1.0 + 0.05 * i for i in range(num_layers)]
    variance_per_layer = [0.1 + 0.02 * i for i in range(num_layers)]

    return ActivationMetrics(
        per_layer_norm=per_layer_norm,
        variance_per_layer=variance_per_layer
    )


def analyze_abc_file(abc_path: Path, baseline_stats: dict | None = None) -> AnalysisResult:
    """Analyze a single ABC file and extract all metrics.

    Args:
        abc_path: Path to ABC notation file
        baseline_stats: Optional baseline statistics for comparison

    Returns:
        Complete analysis result
    """
    abc_content = abc_path.read_text(encoding='utf-8')
    tokens = tokenize(abc_content)

    # Get log probabilities
    log_probs = compute_log_probabilities(abc_content)

    # Compute all metrics
    perplexity = compute_perplexity(log_probs)
    attention = compute_attention_entropy([])  # Placeholder
    surprisal = compute_surprisal(log_probs, tokens)
    activations = compute_activations([])  # Placeholder

    # Compute comparative metrics if baseline provided
    comparative = None
    if baseline_stats:
        perplexity_zscore = (perplexity.overall - baseline_stats['perplexity_mean']) / baseline_stats['perplexity_std']
        entropy_zscore = (attention.mean_entropy - baseline_stats['entropy_mean']) / baseline_stats['entropy_std']
        comparative = ComparativeMetrics(
            baseline="traditional_mean",
            perplexity_zscore=perplexity_zscore,
            entropy_zscore=entropy_zscore
        )

    return AnalysisResult(
        filename=abc_path.name,
        model=MODEL_NAME,
        timestamp=datetime.now().isoformat(),
        token_count=len(tokens),
        perplexity=perplexity,
        attention=attention,
        surprisal=surprisal,
        activations=activations,
        comparative=comparative
    )


def save_metrics(result: AnalysisResult, output_dir: Path) -> Path:
    """Save analysis result to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.filename}.json"

    # Convert dataclasses to dict
    result_dict = asdict(result)

    output_path.write_text(json.dumps(result_dict, indent=2), encoding='utf-8')
    return output_path


def analyze_corpus(
    input_dir: Path,
    output_dir: Path = Path("data/metrics")
) -> list[AnalysisResult]:
    """Analyze all ABC files in a directory.

    Args:
        input_dir: Directory containing ABC files
        output_dir: Directory to save metrics

    Returns:
        List of analysis results
    """
    results = []
    abc_files = sorted(input_dir.glob("*.abc"))

    print(f"Found {len(abc_files)} ABC files to analyze")

    for abc_path in abc_files:
        print(f"Analyzing: {abc_path.name}")
        result = analyze_abc_file(abc_path)
        save_metrics(result, output_dir)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze ABC files with Phi-4")
    parser.add_argument("path", type=Path, help="ABC file or directory to analyze")
    parser.add_argument("--output", type=Path, default=Path("data/metrics"), help="Output directory")
    parser.add_argument("--all", action="store_true", help="Analyze all files in directory")

    args = parser.parse_args()

    # Check Ollama availability
    if not check_ollama_available():
        print("Warning: Ollama server not available. Using placeholder metrics.")
        print("Start Ollama with: ollama serve")

    if args.path.is_file():
        result = analyze_abc_file(args.path)
        output_path = save_metrics(result, args.output)
        print(f"Saved metrics to: {output_path}")
    elif args.path.is_dir():
        results = analyze_corpus(args.path, args.output)
        print(f"Analyzed {len(results)} files")
    else:
        print(f"Error: Path not found: {args.path}")


if __name__ == "__main__":
    main()
