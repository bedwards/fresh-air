# Interpretability Metrics Documentation

This document describes the interpretability metrics computed by the Phi-4 model harness for analyzing ABC notation music files.

## Overview

The model harness extracts several types of metrics to understand how the Phi-4 language model processes ABC notation:

1. **Perplexity** - Measures model uncertainty/confidence
2. **Surprisal** - Identifies unexpected tokens
3. **Attention** - Analyzes where the model focuses (limited via Ollama)
4. **Activations** - Tracks internal representations (limited via Ollama)

## Metrics Reference

### Perplexity

**Definition**: Perplexity measures how "surprised" the model is by the text on average. Lower perplexity indicates the model finds the text more predictable.

**Formula**:
```
Perplexity = exp(mean(-log(p)))
```

Where `p` is the probability assigned to each token.

**Per-token perplexity**:
```
PPL(token) = exp(-log_prob(token)) = 1/p(token)
```

**Interpretation**:
- **Low perplexity (< 10)**: Model is very confident; text follows expected patterns
- **Medium perplexity (10-50)**: Normal variability in predictions
- **High perplexity (> 50)**: Model finds content unusual or unpredictable

**For ABC notation**:
- Traditional tunes typically show lower perplexity due to conventional patterns
- Experimental/avant-garde pieces may show higher perplexity
- Header fields (X:, T:, K:) are usually low perplexity
- Musical content shows more variation

### Surprisal

**Definition**: Surprisal (also called "information content") measures how unexpected a specific token is, in bits.

**Formula**:
```
Surprisal = -log2(p(token))
```

Converting from natural log:
```
Surprisal (bits) = -log_prob / ln(2)
```

**Interpretation**:
- **0 bits**: Token was predicted with 100% confidence
- **1 bit**: Token had 50% probability
- **3.32 bits**: Token had 10% probability
- **6.64 bits**: Token had 1% probability
- **>10 bits**: Very unexpected (flagged as "high surprisal")

**High Surprisal Tokens**:
Tokens with surprisal > 10 bits are flagged as potentially interesting:
- Novel musical patterns
- Unusual note sequences
- Potential transcription errors
- Creative deviations from norms

### Attention Entropy (Limited)

**Definition**: Entropy of attention distributions measures how focused or diffuse the model's attention is.

**Formula**:
```
H(attention) = -sum(p * log(p))
```

Where `p` is the attention weight distribution across positions.

**Interpretation**:
- **Low entropy**: Focused attention on specific positions
- **High entropy**: Diffuse attention across many positions

**Limitation**: The Ollama API does not expose attention weights directly. Full attention analysis requires loading the model via the transformers library:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    output_attentions=True
)
outputs = model(input_ids, output_attentions=True)
attentions = outputs.attentions  # Tuple of attention tensors per layer
```

### Activation Metrics (Limited)

**Definition**: Statistics about hidden state activations across transformer layers.

**Metrics**:
- **Per-layer L2 norm**: Magnitude of activation vectors
- **Variance per layer**: Spread of activation values

**Interpretation**:
- Tracking activation norms can identify processing anomalies
- Unusual patterns may indicate the model is "working harder" on certain inputs

**Limitation**: The Ollama API does not expose internal activations. Full analysis requires:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    output_hidden_states=True
)
outputs = model(input_ids, output_hidden_states=True)
hidden_states = outputs.hidden_states  # Tuple of hidden state tensors
```

## Implementation Approach

### Ollama API Integration

The harness uses Ollama's `/api/generate` endpoint with `logprobs: true` to obtain token-level log probabilities.

**Request format**:
```json
{
  "model": "phi4:14b",
  "prompt": "...",
  "stream": false,
  "logprobs": true,
  "top_logprobs": 5,
  "options": {
    "num_predict": 10,
    "temperature": 0
  }
}
```

**Response includes**:
```json
{
  "logprobs": [
    {
      "token": "...",
      "logprob": -1.234,
      "bytes": [...]
    }
  ]
}
```

### Sliding Window Analysis

Since Ollama provides log probabilities for *generated* tokens (not input tokens), we use a continuation-based approach:

1. Split the ABC content into overlapping chunks (50 chars, 10 overlap)
2. Use each chunk as context and request continuation
3. Collect log probabilities of predicted tokens
4. Aggregate metrics across all chunks

This provides a measure of how predictable the model finds the ABC content when used as context.

### Estimation Mode

When Ollama is unavailable, the harness falls back to estimation mode that assigns log probabilities based on ABC notation structure:

| Pattern Type | Estimated Log Prob | Rationale |
|-------------|-------------------|-----------|
| Headers (X:, T:, K:) | -0.5 | Very predictable |
| Note letters (A-G, a-g) | -1.5 | Common but varied |
| Whitespace | -0.8 | Predictable |
| Numbers | -2.0 | Moderately predictable |
| Accidentals (^, _, =) | -2.5 | Less common |
| Other | -3.0 | Unpredictable |

## Output Format

Each analysis produces a JSON file in `data/metrics/`:

```json
{
  "filename": "traditional_001.abc",
  "model": "phi4:14b",
  "timestamp": "2026-01-21T10:30:45.123456",
  "token_count": 150,
  "perplexity": {
    "overall": 45.23,
    "per_token": [1.5, 2.3, ...],
    "max": 89.1,
    "min": 1.2,
    "mean": 35.4
  },
  "surprisal": {
    "mean": 5.2,
    "per_token": [0.72, 1.21, ...],
    "high_surprisal_tokens": [
      {"position": 45, "token": "^F", "surprisal": 12.3}
    ]
  },
  "attention": {
    "mean_entropy": null,
    "per_layer": null,
    "per_head": null,
    "high_entropy_heads": [],
    "note": "Attention weights not available via Ollama API..."
  },
  "activations": {
    "per_layer_norm": null,
    "variance_per_layer": null,
    "note": "Internal activations not available via Ollama API..."
  },
  "comparative": null
}
```

## Interpreting Results

### Comparing Traditional vs Experimental Music

| Metric | Traditional | Experimental |
|--------|------------|--------------|
| Overall Perplexity | Lower (10-40) | Higher (40-100+) |
| High Surprisal Tokens | Fewer | More |
| Mean Surprisal | Lower (3-5 bits) | Higher (5-8 bits) |

### Quality Indicators

**Well-formed ABC**:
- Consistent perplexity across file
- Few high surprisal tokens
- Headers have low surprisal

**Potential Issues**:
- Spike in perplexity mid-file
- Many high surprisal tokens
- Unusual patterns in header section

## Future Improvements

### With Direct Model Access

Using the transformers library would enable:

1. **True Input Perplexity**: Compute perplexity of the ABC content itself, not just continuations
2. **Attention Analysis**: Full attention weight extraction and entropy calculation
3. **Activation Analysis**: Hidden state statistics across all layers
4. **Probing Tasks**: Train classifiers on internal representations

### Recommended Setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with full outputs
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-4",
    output_attentions=True,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")

# Analyze
inputs = tokenizer(abc_content, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Access all outputs
logits = outputs.logits
attentions = outputs.attentions  # List of attention tensors
hidden_states = outputs.hidden_states  # List of hidden state tensors
```

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Perplexity in Language Models](https://huggingface.co/docs/transformers/perplexity)
- [Information Theory Basics](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Phi-4 Model Card](https://huggingface.co/microsoft/phi-4)
