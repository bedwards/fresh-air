#!/usr/bin/env python3
"""Phi-4 model harness for transformer interpretability analysis.

This module provides a wrapper around Phi-4 14B (via Ollama or transformers) to extract
interpretability metrics from ABC notation music files.

Metrics computed:
    - Perplexity (overall and per-token) via continuation probabilities
    - Per-token surprisal
    - High surprisal token identification
    - Attention entropy (per-layer, per-head) via transformers
    - Layer-wise activation statistics (L2 norms, means, variances)
    - Head specialization detection (positional vs content heads)

Usage:
    python src/model_harness.py data/abc_files/traditional_001.abc
    python src/model_harness.py --all data/abc_files/
    python src/model_harness.py data/abc_files/traditional_001.abc --use-transformers

Note: This implementation supports both Ollama API (for basic metrics) and
direct transformers access (for full attention and hidden state analysis).
Use --use-transformers flag for full interpretability metrics.
"""

import argparse
import gc
import json
import math
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Optional imports for transformers-based inference
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None


OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "phi4:14b"
TRANSFORMERS_MODEL_NAME = "microsoft/phi-4"

# Configuration for analysis
CHUNK_SIZE = 50  # Characters per chunk for sliding window analysis
OVERLAP = 10     # Overlap between chunks
HIGH_SURPRISAL_THRESHOLD = 10.0  # bits

# Head specialization thresholds
POSITIONAL_HEAD_THRESHOLD = 0.5  # Attention mass on fixed positions
CONTENT_HEAD_ENTROPY_THRESHOLD = 2.0  # High entropy indicates content-based attention


@dataclass
class PerplexityMetrics:
    """Perplexity-related metrics."""
    overall: float
    per_token: list[float]
    max: float
    min: float
    mean: float


@dataclass
class HeadSpecialization:
    """Classification of attention head specialization."""
    layer: int
    head: int
    head_type: str  # "positional", "content", or "mixed"
    bos_attention: float  # Attention weight to BOS token
    recent_attention: float  # Attention to recent tokens (last 5)
    entropy: float  # Entropy of attention distribution
    confidence: float  # Confidence in classification


@dataclass
class AttentionMetrics:
    """Attention-related metrics.

    When using transformers: Contains full attention entropy per layer and head.
    When using Ollama: Contains placeholder values with documentation.
    """
    mean_entropy: float | None
    per_layer: list[float] | None  # Mean entropy per layer
    per_head: list[list[float]] | None  # [layer][head] entropy values
    high_entropy_heads: list[dict]  # Heads with high entropy
    low_entropy_heads: list[dict]  # Heads with low entropy (focused attention)
    head_specializations: list[dict] | None  # Head type classifications
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

    When using transformers: Contains full hidden state statistics per layer.
    When using Ollama: Contains placeholder values with documentation.
    """
    per_layer_l2_norm: list[float] | None  # L2 norm per layer
    per_layer_mean: list[float] | None  # Mean activation per layer
    per_layer_variance: list[float] | None  # Variance per layer
    per_layer_max: list[float] | None  # Max activation per layer
    per_layer_min: list[float] | None  # Min activation per layer
    embedding_norm: float | None  # L2 norm of embedding layer
    final_layer_norm: float | None  # L2 norm of final layer
    norm_growth_rate: float | None  # How norms change across layers
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


class TransformersClient:
    """Client for direct model access via transformers library.

    This provides full access to attention weights and hidden states
    for interpretability analysis.
    """

    def __init__(self, model_name: str = TRANSFORMERS_MODEL_NAME, device: str = None):
        """Initialize the transformers client.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required for TransformersClient. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._device = device

    def _get_device(self) -> str:
        """Determine the best available device."""
        if self._device and self._device != "auto":
            return self._device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> bool:
        """Load the model and tokenizer.

        Returns:
            True if successful, False otherwise
        """
        if self.model is not None:
            return True

        try:
            print(f"Loading model {self.model_name}...")
            print(f"Using device: {self._get_device()}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Determine device configuration
            device = self._get_device()

            # Load model with memory-efficient settings
            model_kwargs = {
                "trust_remote_code": True,
                "output_attentions": True,
                "output_hidden_states": True,
            }

            # Use float16 for memory efficiency
            if device in ("cuda", "mps"):
                model_kwargs["torch_dtype"] = torch.float16

            # For CPU or single GPU, load directly
            if device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                # Try to load with device_map for automatic distribution
                try:
                    model_kwargs["device_map"] = "auto"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
                except Exception as e:
                    print(f"Warning: device_map='auto' failed: {e}")
                    print("Trying direct device placement...")
                    del model_kwargs["device_map"]
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    ).to(device)

            self.model.eval()
            print(f"Model loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            return False

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def analyze_text(self, text: str) -> dict[str, Any]:
        """Analyze text and return attention weights and hidden states.

        Args:
            text: Input text to analyze

        Returns:
            Dict with 'attentions', 'hidden_states', 'logits', 'tokens', 'log_probs'
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model")

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Limit sequence length for memory
        )

        # Move inputs to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get tokens as strings for analysis
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        # Run forward pass with no gradient computation
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )

        # Compute log probabilities for each token
        # Shift logits and labels for causal LM
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        log_probs = []

        if logits.shape[0] > 1:
            # Compute log softmax over vocabulary
            log_softmax = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
            # Get log prob of actual next tokens
            for i, next_token_id in enumerate(token_ids[1:]):
                lp = log_softmax[i, next_token_id].item()
                log_probs.append(lp)

        return {
            "attentions": outputs.attentions,  # Tuple of (batch, num_heads, seq_len, seq_len)
            "hidden_states": outputs.hidden_states,  # Tuple of (batch, seq_len, hidden_dim)
            "logits": outputs.logits,
            "tokens": tokens,
            "token_ids": token_ids,
            "log_probs": log_probs
        }


def compute_attention_entropy(attentions: tuple) -> AttentionMetrics:
    """Compute attention entropy metrics from transformer attention weights.

    Entropy H = -sum(p * log(p)) measures how distributed attention is.
    - High entropy: attention spread across many positions
    - Low entropy: attention focused on few positions

    Args:
        attentions: Tuple of attention tensors, one per layer
                   Each tensor has shape (batch, num_heads, seq_len, seq_len)

    Returns:
        AttentionMetrics with per-layer and per-head entropy values
    """
    if attentions is None or len(attentions) == 0:
        return AttentionMetrics(
            mean_entropy=None,
            per_layer=None,
            per_head=None,
            high_entropy_heads=[],
            low_entropy_heads=[],
            head_specializations=None,
            note="No attention data available"
        )

    per_layer_entropy = []
    per_head_entropy = []
    all_entropies = []
    high_entropy_heads = []
    low_entropy_heads = []
    head_specializations = []

    # Small epsilon for numerical stability
    eps = 1e-10

    for layer_idx, layer_attention in enumerate(attentions):
        # layer_attention: (batch, num_heads, seq_len, seq_len)
        # Take first batch element
        attn = layer_attention[0]  # (num_heads, seq_len, seq_len)
        num_heads = attn.shape[0]
        seq_len = attn.shape[1]

        layer_head_entropies = []

        for head_idx in range(num_heads):
            # Get attention weights for this head
            head_attn = attn[head_idx]  # (seq_len, seq_len)

            # Handle NaN values in attention weights
            if torch.isnan(head_attn).any():
                head_attn = torch.nan_to_num(head_attn, nan=eps)

            # Compute entropy for each query position, then average
            # Clamp to avoid log(0) and handle edge cases
            head_attn_clamped = torch.clamp(head_attn, min=eps, max=1.0)
            # Renormalize to ensure it sums to 1 after clamping
            head_attn_normalized = head_attn_clamped / head_attn_clamped.sum(dim=-1, keepdim=True).clamp(min=eps)

            entropy_per_position = -torch.sum(
                head_attn_normalized * torch.log(head_attn_normalized.clamp(min=eps)),
                dim=-1
            )  # (seq_len,)

            # Handle any remaining NaN values and convert to bits
            entropy_per_position = torch.nan_to_num(entropy_per_position, nan=0.0)
            mean_entropy = (entropy_per_position.mean().item() / math.log(2))

            # Final NaN check
            if math.isnan(mean_entropy):
                mean_entropy = 0.0

            layer_head_entropies.append(round(mean_entropy, 4))
            all_entropies.append(mean_entropy)

            # Detect head specialization
            spec = _detect_head_specialization(
                head_attn, layer_idx, head_idx, mean_entropy
            )
            head_specializations.append(asdict(spec))

            # Track high/low entropy heads
            max_possible_entropy = math.log2(seq_len) if seq_len > 1 else 1.0
            entropy_ratio = mean_entropy / max_possible_entropy if max_possible_entropy > 0 else 0

            if entropy_ratio > 0.8:  # Very distributed attention
                high_entropy_heads.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "entropy": round(mean_entropy, 4),
                    "entropy_ratio": round(entropy_ratio, 4)
                })
            elif entropy_ratio < 0.3:  # Very focused attention
                low_entropy_heads.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "entropy": round(mean_entropy, 4),
                    "entropy_ratio": round(entropy_ratio, 4)
                })

        per_head_entropy.append(layer_head_entropies)
        per_layer_entropy.append(round(sum(layer_head_entropies) / len(layer_head_entropies), 4))

    mean_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else None

    return AttentionMetrics(
        mean_entropy=round(mean_entropy, 4) if mean_entropy else None,
        per_layer=per_layer_entropy,
        per_head=per_head_entropy,
        high_entropy_heads=high_entropy_heads[:10],  # Limit to top 10
        low_entropy_heads=low_entropy_heads[:10],
        head_specializations=head_specializations,
        note="Computed from transformers attention weights"
    )


def _detect_head_specialization(
    head_attn: "torch.Tensor",
    layer_idx: int,
    head_idx: int,
    entropy: float
) -> HeadSpecialization:
    """Detect whether an attention head is positional or content-based.

    Positional heads: Focus on fixed positions (BOS, recent tokens)
    Content heads: Attention pattern varies based on token content

    Args:
        head_attn: Attention weights for single head (seq_len, seq_len)
        layer_idx: Layer index
        head_idx: Head index
        entropy: Pre-computed entropy for this head

    Returns:
        HeadSpecialization classification
    """
    seq_len = head_attn.shape[0]

    # Compute attention to BOS token (position 0)
    # Average across all query positions
    bos_attention = head_attn[:, 0].mean().item()

    # Compute attention to recent tokens (last 5 positions relative to each query)
    # This requires looking at the causal mask pattern
    recent_attention = 0.0
    recent_window = min(5, seq_len)

    for q_pos in range(seq_len):
        # For this query, what fraction of attention goes to recent tokens?
        start_pos = max(0, q_pos - recent_window + 1)
        if q_pos >= start_pos:
            recent_attn_slice = head_attn[q_pos, start_pos:q_pos + 1]
            recent_attention += recent_attn_slice.sum().item()

    recent_attention /= seq_len  # Average across query positions

    # Classify head type
    if bos_attention > POSITIONAL_HEAD_THRESHOLD:
        head_type = "positional_bos"
        confidence = min(bos_attention / POSITIONAL_HEAD_THRESHOLD, 1.0)
    elif recent_attention > POSITIONAL_HEAD_THRESHOLD:
        head_type = "positional_recent"
        confidence = min(recent_attention / POSITIONAL_HEAD_THRESHOLD, 1.0)
    elif entropy > CONTENT_HEAD_ENTROPY_THRESHOLD:
        head_type = "content"
        confidence = min(entropy / CONTENT_HEAD_ENTROPY_THRESHOLD, 1.0)
    else:
        head_type = "mixed"
        confidence = 0.5

    return HeadSpecialization(
        layer=layer_idx,
        head=head_idx,
        head_type=head_type,
        bos_attention=round(bos_attention, 4),
        recent_attention=round(recent_attention, 4),
        entropy=round(entropy, 4),
        confidence=round(confidence, 4)
    )


def compute_hidden_state_stats(hidden_states: tuple) -> ActivationMetrics:
    """Compute hidden state statistics from transformer hidden states.

    Computes per-layer:
    - L2 norm: Overall magnitude of activations
    - Mean: Average activation value
    - Variance: Spread of activation values
    - Min/Max: Extreme values

    Args:
        hidden_states: Tuple of hidden state tensors, one per layer
                      Each tensor has shape (batch, seq_len, hidden_dim)

    Returns:
        ActivationMetrics with per-layer statistics
    """
    if hidden_states is None or len(hidden_states) == 0:
        return ActivationMetrics(
            per_layer_l2_norm=None,
            per_layer_mean=None,
            per_layer_variance=None,
            per_layer_max=None,
            per_layer_min=None,
            embedding_norm=None,
            final_layer_norm=None,
            norm_growth_rate=None,
            note="No hidden state data available"
        )

    per_layer_l2_norm = []
    per_layer_mean = []
    per_layer_variance = []
    per_layer_max = []
    per_layer_min = []

    for layer_idx, layer_hidden in enumerate(hidden_states):
        # layer_hidden: (batch, seq_len, hidden_dim)
        # Take first batch element and flatten sequence
        hidden = layer_hidden[0]  # (seq_len, hidden_dim)

        # Compute L2 norm (average across sequence positions)
        l2_norms = torch.norm(hidden, dim=-1)  # (seq_len,)
        mean_l2_norm = l2_norms.mean().item()
        per_layer_l2_norm.append(round(mean_l2_norm, 4))

        # Compute mean activation
        mean_val = hidden.mean().item()
        per_layer_mean.append(round(mean_val, 6))

        # Compute variance
        var_val = hidden.var().item()
        per_layer_variance.append(round(var_val, 6))

        # Compute min/max
        per_layer_max.append(round(hidden.max().item(), 4))
        per_layer_min.append(round(hidden.min().item(), 4))

    # Compute norm growth rate (how L2 norm changes across layers)
    embedding_norm = per_layer_l2_norm[0] if per_layer_l2_norm else None
    final_layer_norm = per_layer_l2_norm[-1] if per_layer_l2_norm else None

    norm_growth_rate = None
    if embedding_norm and final_layer_norm and len(per_layer_l2_norm) > 1:
        # Linear regression slope of norms
        n = len(per_layer_l2_norm)
        x_mean = (n - 1) / 2
        y_mean = sum(per_layer_l2_norm) / n

        numerator = sum(
            (i - x_mean) * (norm - y_mean)
            for i, norm in enumerate(per_layer_l2_norm)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0:
            norm_growth_rate = round(numerator / denominator, 6)

    return ActivationMetrics(
        per_layer_l2_norm=per_layer_l2_norm,
        per_layer_mean=per_layer_mean,
        per_layer_variance=per_layer_variance,
        per_layer_max=per_layer_max,
        per_layer_min=per_layer_min,
        embedding_norm=embedding_norm,
        final_layer_norm=final_layer_norm,
        norm_growth_rate=norm_growth_rate,
        note="Computed from transformers hidden states"
    )


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


def compute_attention_metrics_placeholder() -> AttentionMetrics:
    """Return placeholder attention metrics when using Ollama API.

    LIMITATION: The Ollama API does not expose attention weights.
    Use --use-transformers flag for full attention analysis.

    Returns:
        AttentionMetrics with note about limitations
    """
    return AttentionMetrics(
        mean_entropy=None,
        per_layer=None,
        per_head=None,
        high_entropy_heads=[],
        low_entropy_heads=[],
        head_specializations=None,
        note="Attention weights not available via Ollama API. "
             "Use --use-transformers flag for full attention analysis."
    )


def compute_activation_metrics_placeholder() -> ActivationMetrics:
    """Return placeholder activation metrics when using Ollama API.

    LIMITATION: The Ollama API does not expose internal activations.
    Use --use-transformers flag for full hidden state analysis.

    Returns:
        ActivationMetrics with note about limitations
    """
    return ActivationMetrics(
        per_layer_l2_norm=None,
        per_layer_mean=None,
        per_layer_variance=None,
        per_layer_max=None,
        per_layer_min=None,
        embedding_norm=None,
        final_layer_norm=None,
        norm_growth_rate=None,
        note="Internal activations not available via Ollama API. "
             "Use --use-transformers flag for full hidden state analysis."
    )


def analyze_abc_file(
    abc_path: Path,
    client: OllamaClient | None = None,
    baseline_stats: dict | None = None,
    use_transformers: bool = False,
    transformers_client: "TransformersClient | None" = None
) -> AnalysisResult:
    """Analyze a single ABC file and extract all metrics.

    Args:
        abc_path: Path to ABC notation file
        client: Optional OllamaClient instance (creates one if not provided)
        baseline_stats: Optional baseline statistics for comparison
        use_transformers: If True, use transformers for full attention/hidden state analysis
        transformers_client: Optional pre-loaded TransformersClient instance

    Returns:
        Complete analysis result
    """
    abc_content = abc_path.read_text(encoding='utf-8')
    model_name = MODEL_NAME

    if use_transformers:
        # Use transformers for full analysis
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required for --use-transformers. "
                "Install with: pip install transformers torch"
            )

        if transformers_client is None:
            transformers_client = TransformersClient()

        try:
            # Run transformers analysis
            analysis = transformers_client.analyze_text(abc_content)

            # Extract data
            tokens = analysis["tokens"]
            log_probs = analysis["log_probs"]
            attentions = analysis["attentions"]
            hidden_states = analysis["hidden_states"]
            model_name = TRANSFORMERS_MODEL_NAME

            # Compute metrics from transformers output
            perplexity = compute_perplexity(log_probs)
            surprisal = compute_surprisal(log_probs, tokens[1:])  # Skip first token (no log prob)
            attention = compute_attention_entropy(attentions)
            activations = compute_hidden_state_stats(hidden_states)

        except Exception as e:
            print(f"Warning: Transformers analysis failed: {e}")
            print("Falling back to Ollama mode...")
            use_transformers = False

    if not use_transformers:
        # Use Ollama for basic analysis
        if client is None:
            client = OllamaClient()

        # Get log probabilities via continuation analysis
        log_probs, tokens = compute_log_probabilities_via_continuation(
            client, abc_content
        )

        # Compute basic metrics
        perplexity = compute_perplexity(log_probs)
        surprisal = compute_surprisal(log_probs, tokens)
        attention = compute_attention_metrics_placeholder()
        activations = compute_activation_metrics_placeholder()

    # Compute comparative metrics if baseline provided
    comparative = None
    if baseline_stats and perplexity.overall > 0:
        ppl_mean = baseline_stats.get('perplexity_mean', perplexity.overall)
        ppl_std = baseline_stats.get('perplexity_std', 1.0)
        ent_mean = baseline_stats.get('entropy_mean', 0.0)
        ent_std = baseline_stats.get('entropy_std', 1.0)

        perplexity_zscore = (perplexity.overall - ppl_mean) / max(ppl_std, 0.001)

        # Compute entropy z-score if we have attention data
        entropy_zscore = 0.0
        if attention.mean_entropy is not None and ent_std > 0:
            entropy_zscore = (attention.mean_entropy - ent_mean) / ent_std

        comparative = ComparativeMetrics(
            baseline="traditional_mean",
            perplexity_zscore=round(perplexity_zscore, 4),
            entropy_zscore=round(entropy_zscore, 4)
        )

    return AnalysisResult(
        filename=abc_path.name,
        model=model_name,
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
    progress_callback: callable = None,
    use_transformers: bool = False
) -> list[AnalysisResult]:
    """Analyze all ABC files in a directory.

    Args:
        input_dir: Directory containing ABC files
        output_dir: Directory to save metrics
        progress_callback: Optional callback(current, total, filename)
        use_transformers: If True, use transformers for full analysis

    Returns:
        List of analysis results
    """
    results = []
    abc_files = sorted(input_dir.glob("*.abc"))
    total = len(abc_files)
    print(f"Found {total} ABC files to analyze")

    # Initialize the appropriate client
    client = None
    transformers_client = None

    if use_transformers:
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers not available. Install with: pip install transformers torch")
            print("Falling back to Ollama mode.")
            use_transformers = False
        else:
            print(f"Using transformers with model: {TRANSFORMERS_MODEL_NAME}")
            transformers_client = TransformersClient()
            # Pre-load model once for efficiency
            if not transformers_client.load_model():
                print("Warning: Failed to load transformers model. Falling back to Ollama.")
                use_transformers = False
                transformers_client = None

    if not use_transformers:
        client = OllamaClient()
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
            result = analyze_abc_file(
                abc_path,
                client=client,
                use_transformers=use_transformers,
                transformers_client=transformers_client
            )
            save_metrics(result, output_dir)
            results.append(result)

            # Brief status
            status_msg = f"         Perplexity: {result.perplexity.overall:.2f}, Tokens: {result.token_count}"
            if result.attention.mean_entropy is not None:
                status_msg += f", Attention entropy: {result.attention.mean_entropy:.2f}"
            status_msg += f", High surprisal: {len(result.surprisal.high_surprisal_tokens)}"
            print(status_msg)

        except Exception as e:
            print(f"         Error: {e}")
            continue

        # Small delay to avoid overwhelming API (not needed for transformers)
        if not use_transformers:
            time.sleep(0.1)

        # Clear cache between files if using transformers to manage memory
        if use_transformers and torch is not None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Cleanup transformers client
    if transformers_client is not None:
        transformers_client.unload_model()

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
    # Analyze a single file with Ollama
    python src/model_harness.py data/abc_files/traditional_001.abc

    # Analyze with full attention/hidden state metrics via transformers
    python src/model_harness.py data/abc_files/traditional_001.abc --use-transformers

    # Analyze all files in directory
    python src/model_harness.py --all data/abc_files/

    # Analyze all files with transformers
    python src/model_harness.py --all data/abc_files/ --use-transformers

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
    parser.add_argument(
        "--use-transformers", "-t",
        action="store_true",
        help="Use transformers library for full attention and hidden state analysis "
             "(requires more memory but provides complete interpretability metrics)"
    )

    args = parser.parse_args()

    # Check path exists
    if not args.path.exists():
        print(f"Error: Path not found: {args.path}")
        sys.exit(1)

    # Initialize client based on mode
    client = None
    transformers_client = None
    use_transformers = args.use_transformers

    if use_transformers:
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers and torch are required for --use-transformers.")
            print("Install with: pip install transformers torch")
            print("Falling back to Ollama mode.\n")
            use_transformers = False
        else:
            print(f"Using transformers library with model: {TRANSFORMERS_MODEL_NAME}")
            print("This will provide full attention entropy and hidden state metrics.")
            print()

    if not use_transformers:
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
        if use_transformers:
            transformers_client = TransformersClient()

        result = analyze_abc_file(
            args.path,
            client=client,
            use_transformers=use_transformers,
            transformers_client=transformers_client
        )
        output_path = save_metrics(result, args.output)

        print(f"\nResults for {result.filename}:")
        print(f"  Model: {result.model}")
        print(f"  Tokens: {result.token_count}")
        print(f"  Perplexity: {result.perplexity.overall:.2f}")
        print(f"  Mean surprisal: {result.surprisal.mean:.2f} bits")
        print(f"  High surprisal tokens: {len(result.surprisal.high_surprisal_tokens)}")

        if result.attention.mean_entropy is not None:
            print(f"\nAttention Metrics:")
            print(f"  Mean entropy: {result.attention.mean_entropy:.4f} bits")
            print(f"  Number of layers: {len(result.attention.per_layer)}")
            print(f"  High entropy heads: {len(result.attention.high_entropy_heads)}")
            print(f"  Low entropy heads: {len(result.attention.low_entropy_heads)}")

            # Count head types
            if result.attention.head_specializations:
                head_types = {}
                for spec in result.attention.head_specializations:
                    ht = spec.get('head_type', 'unknown')
                    head_types[ht] = head_types.get(ht, 0) + 1
                print(f"  Head specialization: {head_types}")

        if result.activations.per_layer_l2_norm is not None:
            print(f"\nActivation Metrics:")
            print(f"  Embedding L2 norm: {result.activations.embedding_norm:.4f}")
            print(f"  Final layer L2 norm: {result.activations.final_layer_norm:.4f}")
            print(f"  Norm growth rate: {result.activations.norm_growth_rate:.6f}")

        print(f"\nSaved metrics to: {output_path}")

        # Cleanup
        if transformers_client is not None:
            transformers_client.unload_model()

    elif args.path.is_dir():
        # Corpus analysis
        results = analyze_corpus(
            args.path,
            args.output,
            use_transformers=use_transformers
        )
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

            # Add attention statistics if available
            entropies = [r.attention.mean_entropy for r in results
                        if r.attention.mean_entropy is not None]
            if entropies:
                mean_ent = sum(entropies) / len(entropies)
                print(f"  Mean attention entropy: {mean_ent:.4f} bits")
    else:
        print(f"Error: Path is neither file nor directory: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
