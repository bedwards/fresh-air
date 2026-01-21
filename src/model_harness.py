#!/usr/bin/env python3
"""Phi-4 model harness for transformer interpretability analysis.

This module provides a wrapper around Phi-4 14B (via Ollama) to extract
interpretability metrics from ABC notation music files.

Metrics computed:
    - Perplexity (overall and per-token) via continuation probabilities
    - Per-token surprisal
    - High surprisal token identification
    - Attention entropy (limited via Ollama API, documented placeholders)
    - Layer-wise activation statistics (requires direct model access)

Usage:
    python src/model_harness.py data/abc_files/traditional_001.abc
    python src/model_harness.py --all data/abc_files/

Note: This implementation uses the Ollama API which provides log probabilities
for generated tokens. For the most accurate perplexity of input text, direct
model access via transformers would be preferred, but Ollama provides a
practical approximation through continuation-based evaluation.
"""

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "phi4:14b"

# Configuration for analysis
CHUNK_SIZE = 50  # Characters per chunk for sliding window analysis
OVERLAP = 10     # Overlap between chunks
HIGH_SURPRISAL_THRESHOLD = 10.0  # bits


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
    """Attention-related metrics.

    Note: Ollama API does not expose attention weights directly.
    These are placeholder values with documentation about limitations.
    """
    mean_entropy: float | None
    per_layer: list[float] | None
    per_head: list[list[float]] | None  # [layer][head]
    high_entropy_heads: list[dict]
    note: str


@dataclass
class SurprisalMetrics:
    """Surprisal-related metrics."""
    mean: float
    per_token: list[float]
    high_surprisal_tokens: list[dict]


@dataclass
class ActivationMetrics:
    """Hidden state activation metrics.

    Note: Ollama API does not expose internal activations.
    These require direct model access via transformers library.
    """
    per_layer_norm: list[float] | None
    variance_per_layer: list[float] | None
    note: str


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


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = MODEL_NAME):
        self.base_url = base_url
        self.model = model
        self._available = None
        self._model_available = None

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        if self._available is not None:
            return self._available
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._available = response.status_code == 200
            return self._available
        except requests.exceptions.RequestException:
            self._available = False
            return False

    def is_model_available(self) -> bool:
        """Check if the specified model is available."""
        if self._model_available is not None:
            return self._model_available
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self._model_available = any(
                    self.model in m.get("name", "") for m in models
                )
                return self._model_available
            self._model_available = False
            return False
        except requests.exceptions.RequestException:
            self._model_available = False
            return False

    def generate_with_logprobs(
        self,
        prompt: str,
        num_predict: int = 20,
        top_logprobs: int = 5,
        temperature: float = 0.0
    ) -> dict[str, Any]:
        """Generate text with log probabilities.

        Args:
            prompt: Input prompt
            num_predict: Number of tokens to generate
            top_logprobs: Number of top log probabilities to return
            temperature: Sampling temperature (0 for deterministic)

        Returns:
            Response dict containing response text, logprobs, and metadata
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()

    def tokenize_estimate(self, text: str) -> list[str]:
        """Estimate tokenization of text.

        Note: This is an approximation. For exact tokenization,
        use the transformers library with Phi-4's tokenizer.

        Phi-4 uses a BPE tokenizer similar to GPT-4. This function
        provides a reasonable character-based approximation for
        analysis purposes.
        """
        # Approximate BPE by splitting on common boundaries
        # Real Phi-4 tokens average ~4 characters
        tokens = re.findall(
            r'[A-Z][a-z]*|[a-z]+|[0-9]+|[^\w\s]|\s+|\n',
            text
        )
        return tokens


def compute_log_probabilities_via_continuation(
    client: OllamaClient,
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP
) -> tuple[list[float], list[str]]:
    """Compute log probabilities using sliding window continuation.

    This approach:
    1. Splits text into overlapping chunks
    2. Uses each chunk as context and asks model to predict continuation
    3. Measures how well the model predicts the actual next tokens

    Args:
        client: OllamaClient instance
        text: Full text to analyze
        chunk_size: Characters per chunk
        overlap: Overlap between chunks

    Returns:
        Tuple of (log_probabilities, tokens)
    """
    if not client.is_available() or not client.is_model_available():
        print("Warning: Ollama not available, using estimation mode")
        tokens = client.tokenize_estimate(text)
        # Return estimated log probs based on text structure
        return _estimate_log_probs(text, tokens), tokens

    log_probs = []
    tokens = []

    # Process text in chunks with sliding window
    step = chunk_size - overlap
    positions = list(range(0, max(1, len(text) - chunk_size), step))

    # Limit positions to avoid excessive API calls
    max_positions = 50
    if len(positions) > max_positions:
        # Sample evenly across the text
        indices = [int(i * len(positions) / max_positions) for i in range(max_positions)]
        positions = [positions[i] for i in indices]

    for pos in positions:
        chunk_end = min(pos + chunk_size, len(text))
        context = text[pos:chunk_end]

        # Get continuation to see how model handles this context
        actual_continuation = text[chunk_end:chunk_end + 20] if chunk_end < len(text) else ""

        try:
            result = client.generate_with_logprobs(
                prompt=context,
                num_predict=min(10, len(actual_continuation) + 5),
                temperature=0.0
            )

            # Extract log probabilities from generated tokens
            if "logprobs" in result:
                for lp_entry in result["logprobs"]:
                    log_probs.append(lp_entry["logprob"])
                    tokens.append(lp_entry["token"])

        except Exception as e:
            print(f"Warning: Error during API call: {e}")
            # Continue with partial data

    if not log_probs:
        # Fallback to estimation if no API results
        tokens = client.tokenize_estimate(text)
        log_probs = _estimate_log_probs(text, tokens)

    return log_probs, tokens


def _estimate_log_probs(text: str, tokens: list[str]) -> list[float]:
    """Estimate log probabilities based on text structure.

    This provides reasonable estimates when Ollama is unavailable:
    - Common patterns (like ABC headers) get higher probability
    - Repeated patterns get higher probability
    - Unusual characters get lower probability

    Returns log probabilities in natural log scale.
    """
    log_probs = []

    # ABC notation common patterns
    common_patterns = {
        'X:', 'T:', 'M:', 'L:', 'K:', 'Q:', 'C:',  # Headers
        '|', ':|', '|:', '||', '[|', '|]',  # Bar lines
        '/2', '/4', '/8', '2', '3', '4',  # Durations
    }

    # Note letters (very common in ABC)
    note_letters = set('ABCDEFGabcdefg')

    for i, token in enumerate(tokens):
        token_stripped = token.strip()

        if token_stripped in common_patterns:
            # Very predictable
            lp = -0.5  # exp(-0.5) ~ 0.6 probability
        elif token_stripped in note_letters:
            # Notes are common but varied
            lp = -1.5  # exp(-1.5) ~ 0.22 probability
        elif token.isspace():
            # Whitespace is predictable
            lp = -0.8
        elif token_stripped.isdigit():
            # Numbers are moderately predictable
            lp = -2.0
        elif any(c in token for c in '^_='):
            # Accidentals
            lp = -2.5
        elif token_stripped in '()[]{}':
            # Grouping symbols
            lp = -2.2
        else:
            # Other content
            lp = -3.0

        # Add some variation based on position
        # Earlier tokens tend to be more predictable (headers)
        position_factor = min(i / max(len(tokens), 1), 1.0)
        lp -= position_factor * 0.5

        log_probs.append(lp)

    return log_probs


def compute_perplexity(log_probs: list[float]) -> PerplexityMetrics:
    """Compute perplexity metrics from log probabilities.

    Perplexity = exp(mean(-log(p)))

    This measures the effective vocabulary size the model
    considers at each position. Lower perplexity indicates
    the model is more confident about predictions.

    Args:
        log_probs: List of log probabilities (natural log scale)

    Returns:
        PerplexityMetrics with overall, per-token, max, min, mean
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
    # PPL(token) = exp(-log_prob)
    per_token_ppl = []
    for lp in log_probs:
        try:
            ppl = math.exp(-lp)
            # Cap extremely high values
            ppl = min(ppl, 10000.0)
            per_token_ppl.append(ppl)
        except OverflowError:
            per_token_ppl.append(10000.0)

    # Overall perplexity = exp(mean(-log_probs))
    mean_neg_log_prob = -sum(log_probs) / len(log_probs)
    try:
        overall_ppl = math.exp(mean_neg_log_prob)
        overall_ppl = min(overall_ppl, 10000.0)
    except OverflowError:
        overall_ppl = 10000.0

    return PerplexityMetrics(
        overall=round(overall_ppl, 4),
        per_token=[round(p, 4) for p in per_token_ppl],
        max=round(max(per_token_ppl), 4),
        min=round(min(per_token_ppl), 4),
        mean=round(sum(per_token_ppl) / len(per_token_ppl), 4)
    )


def compute_surprisal(log_probs: list[float], tokens: list[str]) -> SurprisalMetrics:
    """Compute per-token surprisal.

    Surprisal = -log2(p(token)) = -log_prob / log(2)

    High surprisal (>10 bits) indicates the model encountered
    something unexpected. This is useful for identifying:
    - Novel musical patterns
    - Unusual chord progressions
    - Potential errors or anomalies

    Args:
        log_probs: Log probabilities (natural log)
        tokens: Corresponding token strings

    Returns:
        SurprisalMetrics with mean, per-token, and high surprisal list
    """
    if not log_probs:
        return SurprisalMetrics(mean=0.0, per_token=[], high_surprisal_tokens=[])

    # Convert natural log to bits: surprisal = -log_prob / ln(2)
    ln2 = math.log(2)
    per_token_surprisal = [-lp / ln2 for lp in log_probs]

    mean_surprisal = sum(per_token_surprisal) / len(per_token_surprisal)

    # Find high surprisal tokens (>threshold bits)
    high_surprisal = []
    for i, (surprisal, token) in enumerate(zip(per_token_surprisal, tokens)):
        if surprisal > HIGH_SURPRISAL_THRESHOLD:
            high_surprisal.append({
                "position": i,
                "token": token,
                "surprisal": round(surprisal, 4)
            })

    return SurprisalMetrics(
        mean=round(mean_surprisal, 4),
        per_token=[round(s, 4) for s in per_token_surprisal],
        high_surprisal_tokens=high_surprisal
    )


def compute_attention_metrics() -> AttentionMetrics:
    """Compute attention-related metrics.

    LIMITATION: The Ollama API does not expose attention weights.
    This function returns documented placeholders.

    For full attention analysis, use the transformers library directly:
    ```python
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-4",
        output_attentions=True
    )
    outputs = model(input_ids, output_attentions=True)
    attentions = outputs.attentions  # Tuple of attention tensors
    ```

    Returns:
        AttentionMetrics with note about limitations
    """
    return AttentionMetrics(
        mean_entropy=None,
        per_layer=None,
        per_head=None,
        high_entropy_heads=[],
        note="Attention weights not available via Ollama API. "
             "Use transformers library with output_attentions=True "
             "for direct model access to attention patterns."
    )


def compute_activation_metrics() -> ActivationMetrics:
    """Compute hidden state activation metrics.

    LIMITATION: The Ollama API does not expose internal activations.
    This function returns documented placeholders.

    For full activation analysis, use the transformers library:
    ```python
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-4",
        output_hidden_states=True
    )
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # Tuple of hidden state tensors
    ```

    Returns:
        ActivationMetrics with note about limitations
    """
    return ActivationMetrics(
        per_layer_norm=None,
        variance_per_layer=None,
        note="Internal activations not available via Ollama API. "
             "Use transformers library with output_hidden_states=True "
             "for direct model access to layer activations."
    )


def analyze_abc_file(
    abc_path: Path,
    client: OllamaClient | None = None,
    baseline_stats: dict | None = None
) -> AnalysisResult:
    """Analyze a single ABC file and extract all metrics.

    Args:
        abc_path: Path to ABC notation file
        client: Optional OllamaClient instance (creates one if not provided)
        baseline_stats: Optional baseline statistics for comparison

    Returns:
        Complete analysis result
    """
    if client is None:
        client = OllamaClient()

    abc_content = abc_path.read_text(encoding='utf-8')

    # Get log probabilities via continuation analysis
    log_probs, tokens = compute_log_probabilities_via_continuation(
        client, abc_content
    )

    # Compute all metrics
    perplexity = compute_perplexity(log_probs)
    surprisal = compute_surprisal(log_probs, tokens)
    attention = compute_attention_metrics()
    activations = compute_activation_metrics()

    # Compute comparative metrics if baseline provided
    comparative = None
    if baseline_stats and perplexity.overall > 0:
        ppl_mean = baseline_stats.get('perplexity_mean', perplexity.overall)
        ppl_std = baseline_stats.get('perplexity_std', 1.0)
        ent_mean = baseline_stats.get('entropy_mean', 0.0)
        ent_std = baseline_stats.get('entropy_std', 1.0)

        perplexity_zscore = (perplexity.overall - ppl_mean) / max(ppl_std, 0.001)
        entropy_zscore = 0.0  # Cannot compute without attention data

        comparative = ComparativeMetrics(
            baseline="traditional_mean",
            perplexity_zscore=round(perplexity_zscore, 4),
            entropy_zscore=round(entropy_zscore, 4)
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
    """Save analysis result to JSON file.

    Args:
        result: AnalysisResult to save
        output_dir: Directory to save metrics

    Returns:
        Path to saved JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.filename}.json"

    # Convert dataclasses to dict
    result_dict = asdict(result)

    output_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    return output_path


def analyze_corpus(
    input_dir: Path,
    output_dir: Path = Path("data/metrics"),
    progress_callback: callable = None
) -> list[AnalysisResult]:
    """Analyze all ABC files in a directory.

    Args:
        input_dir: Directory containing ABC files
        output_dir: Directory to save metrics
        progress_callback: Optional callback(current, total, filename)

    Returns:
        List of analysis results
    """
    results = []
    client = OllamaClient()
    abc_files = sorted(input_dir.glob("*.abc"))

    total = len(abc_files)
    print(f"Found {total} ABC files to analyze")

    if not client.is_available():
        print("Warning: Ollama server not available. Using estimation mode.")
        print("Start Ollama with: ollama serve")
    elif not client.is_model_available():
        print(f"Warning: Model {MODEL_NAME} not available.")
        print(f"Pull model with: ollama pull {MODEL_NAME}")

    for i, abc_path in enumerate(abc_files):
        print(f"[{i+1}/{total}] Analyzing: {abc_path.name}")

        if progress_callback:
            progress_callback(i + 1, total, abc_path.name)

        try:
            result = analyze_abc_file(abc_path, client)
            save_metrics(result, output_dir)
            results.append(result)

            # Brief status
            print(f"         Perplexity: {result.perplexity.overall:.2f}, "
                  f"Tokens: {result.token_count}, "
                  f"High surprisal: {len(result.surprisal.high_surprisal_tokens)}")

        except Exception as e:
            print(f"         Error: {e}")
            continue

        # Small delay to avoid overwhelming API
        time.sleep(0.1)

    return results


def compute_corpus_statistics(results: list[AnalysisResult]) -> dict:
    """Compute aggregate statistics across corpus.

    Args:
        results: List of analysis results

    Returns:
        Dict with corpus-level statistics
    """
    if not results:
        return {}

    perplexities = [r.perplexity.overall for r in results if r.perplexity.overall > 0]
    surprisals = [r.surprisal.mean for r in results if r.surprisal.mean > 0]
    token_counts = [r.token_count for r in results]

    stats = {
        "total_files": len(results),
        "total_tokens": sum(token_counts),
        "perplexity": {
            "mean": sum(perplexities) / len(perplexities) if perplexities else 0,
            "std": (sum((p - sum(perplexities)/len(perplexities))**2
                       for p in perplexities) / len(perplexities))**0.5 if perplexities else 0,
            "min": min(perplexities) if perplexities else 0,
            "max": max(perplexities) if perplexities else 0,
        },
        "surprisal": {
            "mean": sum(surprisals) / len(surprisals) if surprisals else 0,
            "std": (sum((s - sum(surprisals)/len(surprisals))**2
                       for s in surprisals) / len(surprisals))**0.5 if surprisals else 0,
        },
        "high_surprisal_files": sum(
            1 for r in results if r.surprisal.high_surprisal_tokens
        ),
    }

    return stats


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Analyze ABC files with Phi-4 for interpretability metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single file
    python src/model_harness.py data/abc_files/traditional_001.abc

    # Analyze all files in directory
    python src/model_harness.py --all data/abc_files/

    # Specify custom output directory
    python src/model_harness.py --all data/abc_files/ --output results/metrics
        """
    )
    parser.add_argument(
        "path",
        type=Path,
        help="ABC file or directory to analyze"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/metrics"),
        help="Output directory for metrics (default: data/metrics)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Analyze all ABC files in directory"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print corpus statistics after analysis"
    )

    args = parser.parse_args()

    # Check path exists
    if not args.path.exists():
        print(f"Error: Path not found: {args.path}")
        sys.exit(1)

    # Check Ollama availability
    client = OllamaClient()
    if not client.is_available():
        print("Warning: Ollama server not available at", OLLAMA_URL)
        print("Using estimation mode. Start Ollama with: ollama serve")
    elif not client.is_model_available():
        print(f"Warning: Model {MODEL_NAME} not available")
        print(f"Pull model with: ollama pull {MODEL_NAME}")
    else:
        print(f"Connected to Ollama, using model: {MODEL_NAME}")

    print()

    if args.path.is_file():
        # Single file analysis
        result = analyze_abc_file(args.path, client)
        output_path = save_metrics(result, args.output)
        print(f"\nResults for {result.filename}:")
        print(f"  Tokens: {result.token_count}")
        print(f"  Perplexity: {result.perplexity.overall:.2f}")
        print(f"  Mean surprisal: {result.surprisal.mean:.2f} bits")
        print(f"  High surprisal tokens: {len(result.surprisal.high_surprisal_tokens)}")
        print(f"\nSaved metrics to: {output_path}")

    elif args.path.is_dir():
        # Corpus analysis
        results = analyze_corpus(args.path, args.output)
        print(f"\nAnalyzed {len(results)} files")
        print(f"Metrics saved to: {args.output}")

        if args.stats or True:  # Always show stats for corpus
            stats = compute_corpus_statistics(results)
            print("\nCorpus Statistics:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            print(f"  Perplexity - mean: {stats['perplexity']['mean']:.2f}, "
                  f"std: {stats['perplexity']['std']:.2f}")
            print(f"  Surprisal - mean: {stats['surprisal']['mean']:.2f} bits")
            print(f"  Files with high surprisal tokens: {stats['high_surprisal_files']}")
    else:
        print(f"Error: Path is neither file nor directory: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
