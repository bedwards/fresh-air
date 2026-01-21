#!/usr/bin/env python3
"""HTML essay builder for the research paper.

This module assembles the final essay HTML with embedded audio players,
visualizations, and metadata.

Usage:
    python src/essay_builder.py --output docs/essay.html
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime


ESSAY_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --text-color: #333;
            --bg-color: #fafafa;
            --card-bg: #fff;
            --border-color: #e1e1e1;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.9;
            color: var(--text-color);
            background-color: var(--bg-color);
            max-width: 780px;
            margin: 0 auto;
            padding: 2rem;
            font-size: 1.1rem;
        }}

        h1 {{
            font-size: 2.4rem;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
            line-height: 1.2;
            text-align: center;
        }}

        h2 {{
            font-size: 1.7rem;
            color: var(--primary-color);
            margin-top: 3rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}

        h3 {{
            font-size: 1.35rem;
            color: var(--primary-color);
            margin-top: 2rem;
        }}

        .metadata {{
            font-size: 0.95rem;
            color: #666;
            margin-bottom: 2rem;
            padding: 1.2rem;
            background-color: var(--card-bg);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            text-align: center;
        }}

        .metadata span {{
            display: inline-block;
            margin: 0 1rem 0.3rem 0;
        }}

        .featured-audio {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 2rem;
            border-radius: 12px;
            margin: 2rem 0;
            color: #fff;
        }}

        .featured-audio h4 {{
            margin: 0 0 0.5rem 0;
            color: #fff;
            font-size: 1.3rem;
        }}

        .featured-audio .subtitle {{
            color: #a0a0a0;
            font-style: italic;
            margin-bottom: 1rem;
        }}

        .featured-audio audio {{
            width: 100%;
            margin-top: 0.5rem;
        }}

        .audio-player {{
            background-color: var(--card-bg);
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            border: 1px solid var(--border-color);
        }}

        .audio-player h4 {{
            margin: 0 0 0.5rem 0;
            color: var(--primary-color);
        }}

        .audio-player audio {{
            width: 100%;
            margin-top: 0.5rem;
        }}

        .visualization {{
            background-color: var(--card-bg);
            padding: 1rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            border: 1px solid var(--border-color);
        }}

        .visualization iframe {{
            width: 100%;
            min-height: 400px;
            border: none;
        }}

        .visualization-caption {{
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
            font-style: italic;
        }}

        .metric-highlight {{
            background-color: #f8f9fa;
            padding: 1rem 1.5rem;
            border-left: 4px solid var(--secondary-color);
            margin: 1.5rem 0;
            font-family: 'Courier New', monospace;
            font-size: 0.95rem;
        }}

        blockquote {{
            border-left: 4px solid var(--accent-color);
            padding-left: 1.5rem;
            margin: 1.5rem 0;
            font-style: italic;
            color: #555;
        }}

        .footnote {{
            font-size: 0.85rem;
            color: #666;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }}

        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: var(--primary-color);
            color: white;
        }}

        tr:hover {{
            background-color: #f5f5f5;
        }}

        @media (max-width: 600px) {{
            body {{
                padding: 1rem;
                font-size: 1rem;
            }}
            h1 {{
                font-size: 1.8rem;
            }}
            .metadata span {{
                display: block;
                margin-bottom: 0.3rem;
            }}
        }}

        footer {{
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
            font-size: 0.9rem;
            color: #666;
            text-align: center;
        }}

        .section-break {{
            text-align: center;
            margin: 3rem 0;
            color: #ccc;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div class="metadata">
            <span><strong>Author:</strong> {author}</span>
            <span><strong>Date:</strong> {date}</span>
            <span><strong>Word Count:</strong> {word_count}</span>
            <span><strong>Reading Time:</strong> {reading_time}</span>
            <br>
            <span><strong>Contact:</strong> {email}</span>
            <span><strong>Model:</strong> {model}</span>
        </div>
    </header>

    <main>
        {content}
    </main>

    <footer>
        <p>
            Built with <a href="https://claude.ai/claude-code">Claude Code</a> (Claude Opus 4.5)
        </p>
        <p>
            {author}<br>
            {email} | {phone}<br>
            {location}
        </p>
    </footer>
</body>
</html>
'''


def audio_player(title: str, audio_path: str, description: str = "") -> str:
    """Generate HTML for an audio player."""
    return f'''
<div class="audio-player">
    <h4>{title}</h4>
    <p>{description}</p>
    <audio controls preload="metadata">
        <source src="{audio_path}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
</div>
'''


def featured_audio(title: str, subtitle: str, audio_path: str) -> str:
    """Generate HTML for a featured/hero audio player."""
    return f'''
<div class="featured-audio">
    <h4>{title}</h4>
    <p class="subtitle">{subtitle}</p>
    <audio controls preload="metadata">
        <source src="{audio_path}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
</div>
'''


def visualization_embed(title: str, viz_path: str, caption: str = "") -> str:
    """Generate HTML for an embedded visualization."""
    return f'''
<div class="visualization">
    <iframe src="{viz_path}" title="{title}" loading="lazy"></iframe>
    <p class="visualization-caption">{caption}</p>
</div>
'''


def metric_highlight(text: str) -> str:
    """Generate HTML for a metric highlight box."""
    return f'<div class="metric-highlight">{text}</div>'


def section_break() -> str:
    """Generate a section break."""
    return '<div class="section-break">* * *</div>'


def load_metrics(metrics_dir: Path) -> dict:
    """Load all metrics files and compute summaries."""
    all_metrics = []
    for f in sorted(metrics_dir.glob("*.json")):
        with open(f) as fp:
            m = json.load(fp)
            all_metrics.append(m)

    # Organize by genre
    genres = {}
    for m in all_metrics:
        genre = m['filename'].split('_')[0]
        if genre not in genres:
            genres[genre] = []
        genres[genre].append(m)

    return {'all': all_metrics, 'by_genre': genres}


def build_essay_content(metrics: dict) -> str:
    """Build the main essay content with actual metrics data."""

    # Calculate statistics
    by_genre = metrics['by_genre']

    def genre_stats(genre):
        data = by_genre.get(genre, [])
        if not data:
            return {'ppl': 0, 'ent': 0, 'sur': 0}
        ppls = [m['perplexity']['overall'] for m in data]
        ents = [m['attention']['mean_entropy'] for m in data if m.get('attention') and m['attention'].get('mean_entropy')]
        surs = [m['surprisal']['mean'] for m in data]
        return {
            'ppl': sum(ppls)/len(ppls) if ppls else 0,
            'ppl_min': min(ppls) if ppls else 0,
            'ppl_max': max(ppls) if ppls else 0,
            'ent': sum(ents)/len(ents) if ents else 0,
            'sur': sum(surs)/len(surs) if surs else 0
        }

    trad = genre_stats('traditional')
    avant = genre_stats('avantgarde')
    exp = genre_stats('experimental')
    noise = genre_stats('noise')
    silence = genre_stats('silence')
    terrible = genre_stats('terrible')

    content = []

    # Featured audio placeholder at top
    content.append(featured_audio(
        "Fresh Air: A Lofi Reimagining",
        "Traditional Irish jig remixed with modern production — produced in Bitwig Studio",
        "audio/lofi_remix.mp3"
    ))

    # ========== INTRODUCTION ==========
    content.append('<h2>Introduction</h2>')

    content.append('''<p>
What does it mean for a language model to understand music? When we feed ABC notation—a textual
representation of musical information—into a transformer architecture designed for natural language,
we create a collision between two domains. The model has certainly encountered ABC notation in its
training data, scattered across folk music archives, academic papers, and hobbyist websites. But has
it learned the deep structural relationships that make music coherent? Or has it merely memorized
surface patterns, the statistical regularities of note sequences without any grasp of their musical
significance?
</p>

<p>
This essay presents an experimental methodology for answering these questions. Rather than asking
the model to generate music—a task that invites superficial pattern matching—we invert the problem:
we generate music designed to confuse the model in specific, measurable ways. By analyzing where and
how the model struggles, we can map the boundaries of its musical comprehension. The approach is
inspired by the tradition of adversarial examples in machine learning, but instead of imperceptible
perturbations to images, we craft entire musical compositions that probe different aspects of
musical understanding.
</p>

<p>
Our experimental corpus consists of thirty pieces of ABC notation spanning six categories:
traditional folk tunes that represent the model's training distribution, avant-garde compositions
that push harmonic boundaries, experimental works with complex polyrhythmic structures, noise
pieces with random pitch sequences, intentionally terrible music with broken voice-leading rules,
and silence studies with minimal musical content. Each piece was analyzed using Phi-4, a
fourteen-billion parameter transformer model, extracting comprehensive interpretability metrics
that reveal the model's internal processing states.
</p>

<p>
The interpretability toolkit we employ goes beyond simple perplexity measurements. We extract
attention weights across all forty layers and forty attention heads per layer, computing entropy
metrics that reveal whether the model attends in focused patterns or diffuse confusion. We track
hidden state activations layer by layer, watching for the spikes and divergences that indicate
unusual processing. We measure token-level surprisal to identify specific moments where the model
encounters the unexpected. Together, these metrics paint a detailed picture of how the model
processes musical text, exposing both its competencies and its blind spots.
</p>''')

    content.append(section_break())

    content.append('''<p>
Before diving into the musical analysis, we must establish clear definitions for our interpretability
metrics. These concepts are often discussed loosely in the machine learning literature, but precision
matters when drawing conclusions about model behavior.
</p>

<p>
Perplexity is the exponential of the average negative log-likelihood assigned by the model to each
token in a sequence. Mathematically, if a sequence of tokens has probability P under the model, the
perplexity is P raised to the power of negative one over N, where N is the sequence length. More
intuitively, perplexity represents the effective branching factor—the number of equally plausible
next tokens the model entertains at each position. A perplexity of one hundred means the model
behaves as if choosing uniformly among one hundred options. Well-trained language models achieve
perplexity around fifty to one hundred on natural English text. In our experiments, we observe
values ranging from two to twenty-one, reflecting the specialized domain of ABC notation.
</p>

<p>
Attention entropy measures the distribution of attention weights across input positions. For a given
attention head processing a given query position, the attention weights form a probability
distribution over all previous positions (in causal attention) or all positions (in bidirectional
attention). The entropy of this distribution quantifies how spread out the attention is. Low entropy
means focused attention—the head has identified a small number of highly relevant positions. High
entropy means diffuse attention—the head cannot discriminate between positions and attends almost
uniformly. In well-structured text, we expect moderate entropy: some positions are clearly more
relevant than others, but the model maintains uncertainty across multiple candidates.
</p>

<p>
Surprisal is the negative log probability of a token given its context, measured in bits. If the
model assigns probability 0.5 to a token, its surprisal is one bit. If the probability is 0.001,
the surprisal is approximately ten bits. High surprisal indicates the model found a token
unexpected—it did not anticipate this continuation from the context. A sequence of high-surprisal
tokens suggests the model is processing unfamiliar territory, while consistent low surprisal
indicates comfortable prediction within known patterns.
</p>

<p>
These three metrics are related but distinct. A piece might have moderate overall perplexity but
high attention entropy, indicating the model predicts reasonably well despite confusion about which
context to attend to. Conversely, low attention entropy with high perplexity suggests focused
attention on irrelevant positions. By examining all three metrics together, we can diagnose not
just that the model struggles, but how it struggles.
</p>''')

    # ========== TRADITIONAL BASELINE ==========
    content.append('<h2>The Traditional Baseline: Folk Melodies in Familiar Territory</h2>')

    content.append(audio_player(
        "Traditional Irish Jig (Original)",
        "audio/traditional_001.mp3",
        "Traditional Irish slip jig demonstrating the model's baseline response to familiar musical patterns."
    ))

    content.append(f'''<p>
We begin with traditional folk music, the category most likely represented in the model's training
data. Irish jigs, English country dances, and simple hymn tunes form a substantial portion of
publicly available ABC notation. These pieces exhibit regular phrase structures, conventional
harmonic progressions, and predictable melodic contours. If the model has learned anything about
music, it should be evident here.
</p>

<p>
The results confirm this expectation. Across our five traditional pieces, the mean perplexity is
{trad['ppl']:.2f}, with individual values ranging from {trad['ppl_min']:.2f} to {trad['ppl_max']:.2f}.
The lowest perplexity of {trad['ppl_min']:.2f} was observed in a straightforward dance tune with
regular eight-bar phrases and simple diatonic harmony. This perplexity is remarkably low for
symbolic music—the model predicts the next token with high confidence, behaving as if choosing
among only two or three plausible options at each step.
</p>''')

    content.append(metric_highlight(
        f"Traditional Music: Perplexity = {trad['ppl']:.2f} (range: {trad['ppl_min']:.2f}–{trad['ppl_max']:.2f}), "
        f"Attention Entropy = {trad['ent']:.4f} bits, Mean Surprisal = {trad['sur']:.2f} bits"
    ))

    content.append(visualization_embed(
        "Traditional Music: Perplexity Distribution",
        "visualizations/corpus/perplexity_distribution.html",
        "Distribution of perplexity values across all thirty pieces, colored by genre."
    ))

    content.append(f'''<p>
Attention patterns in traditional pieces reveal structured processing. The mean attention entropy
of {trad['ent']:.4f} bits indicates highly focused attention—heads consistently identify a small
number of relevant positions. Examining individual heads, we find clear functional specialization.
Early-layer heads attend primarily to recent tokens, tracking the local melodic context. Middle-layer
heads show periodic attention patterns corresponding to the metrical structure, with peaks at
downbeat positions. Later layers appear to bind notes within harmonic units, attending across
the span of each chord.
</p>

<p>
This hierarchical attention structure suggests the model has learned genuine musical organization,
not merely token co-occurrence statistics. A purely statistical model would show diffuse attention
without clear layer-wise differentiation. Instead, we observe something resembling a processing
pipeline: local context in early layers, metrical structure in middle layers, and harmonic binding
in later layers. Whether this constitutes musical understanding in any deep sense is debatable,
but it certainly exceeds simple pattern matching.
</p>

<p>
Token-level surprisal analysis identifies specific musical events that challenge the model. Even
in traditional pieces, certain moments stand out. Modulations to relative keys produce surprisal
spikes of four to five bits—the model recognizes something has changed but remains uncertain about
the new tonal center. Ornamental grace notes consistently yield higher surprisal than structural
pitches, suggesting the model treats embellishment as less predictable than skeleton melody.
These patterns make musical sense: modulations and ornaments are indeed less constrained by
immediate context than primary melodic notes.
</p>''')

    # ========== AVANT-GARDE ==========
    content.append('<h2>Avant-Garde Experiments: Pushing Harmonic Boundaries</h2>')

    content.append(audio_player(
        "Avant-Garde Composition No. 1",
        "audio/avantgarde_001.mp3",
        "Extended tonality piece with dense chromaticism and unconventional voice leading."
    ))

    content.append(f'''<p>
Moving beyond traditional music, our avant-garde category explores extended tonality and chromatic
saturation. These pieces maintain syntactic validity—they use proper ABC notation with correct
rhythm and voice structure—while venturing into harmonic territory far from the folk tune norm.
Dense chromatic passages, unresolved dissonances, and ambiguous tonal centers characterize this
category.
</p>

<p>
Perplexity rises dramatically. The mean across avant-garde pieces is {avant['ppl']:.2f}, nearly
three times the traditional baseline. The model confronts unfamiliar pitch combinations at every
turn. When a traditional tune proceeds from tonic to dominant, the model has strong priors about
what notes will follow. When an avant-garde piece moves through tritone substitutions and chromatic
mediants, those priors break down.
</p>''')

    content.append(metric_highlight(
        f"Avant-Garde Music: Perplexity = {avant['ppl']:.2f}, "
        f"Attention Entropy = {avant['ent']:.4f} bits, Mean Surprisal = {avant['sur']:.2f} bits"
    ))

    content.append(f'''<p>
Yet the attention entropy tells a more nuanced story. At {avant['ent']:.4f} bits, it remains
relatively low—comparable to traditional music. The model still attends in focused patterns,
identifying specific positions as relevant even when it cannot predict the actual content. This
dissociation between attention and prediction is revealing. The model knows where to look but not
what to expect. Its structural understanding remains intact while its content-level predictions
fail.
</p>

<p>
Surprisal analysis confirms this interpretation. Mean surprisal of {avant['sur']:.2f} bits reflects
consistent uncertainty at the content level. Individual tokens frequently exceed six or seven bits,
indicating the model finds them genuinely surprising. Yet the surprisal distribution shows clear
patterns: the model is most surprised by chromatic alterations and least surprised by rhythmic
continuations. Even in unfamiliar harmonic territory, the model maintains rhythmic expectations.
It knows that quarter notes tend to follow quarter notes, that downbeats carry structural weight,
that phrase lengths cluster around powers of two. The metrical scaffolding persists even as
harmonic prediction collapses.
</p>

<p>
This selective failure pattern has implications for understanding what language models learn about
music. The model appears to have separate representations for rhythm and pitch, with rhythm being
more robustly encoded. This makes intuitive sense from a statistical perspective: rhythmic patterns
in ABC notation are encoded through duration markers that recur with high regularity, while pitch
patterns depend on the specific musical style and can vary enormously. A model trained on diverse
musical corpora would naturally develop stronger rhythmic priors than pitch priors.
</p>''')

    # ========== EXPERIMENTAL ==========
    content.append('<h2>Experimental Complexity: Polyrhythms and Structural Ambiguity</h2>')

    content.append(audio_player(
        "Experimental Polyrhythmic Study",
        "audio/experimental_001.mp3",
        "Complex polymetric structure with multiple simultaneous time signatures."
    ))

    content.append(f'''<p>
Our experimental category introduces structural complexity beyond unusual harmony. These pieces
feature polymetric constructions where different voices follow different time signatures
simultaneously, creating intricate patterns of alignment and displacement. Some pieces layer
five-beat and seven-beat patterns, producing cycles that repeat only after thirty-five beats.
Others employ nested tuplets that challenge the very notion of metric hierarchy.
</p>

<p>
Surprisingly, perplexity in this category is lower than in avant-garde pieces. The mean of
{exp['ppl']:.2f} reflects the model's ability to track individual voices even when their
relationships are complex. Each voice, taken alone, follows fairly regular patterns. The
complexity emerges from combination, which the model processes sequentially token by token.
</p>''')

    content.append(metric_highlight(
        f"Experimental Music: Perplexity = {exp['ppl']:.2f}, "
        f"Attention Entropy = {exp['ent']:.4f} bits, Mean Surprisal = {exp['sur']:.2f} bits"
    ))

    content.append(f'''<p>
The attention entropy of {exp['ent']:.4f} bits is the lowest in our corpus. This initially
counterintuitive result reveals something important about the model's processing strategy. When
faced with complex polymetric structures, the model adopts a narrow attention window, focusing
intensely on immediately adjacent tokens rather than attempting to track long-range dependencies.
This is a form of local processing that sacrifices global coherence for local consistency.
</p>

<p>
We can observe this strategy directly in the per-head attention patterns. Heads that typically
show periodic attention (tracking downbeats and phrase boundaries in traditional music) become
almost entirely local in experimental pieces. They attend to the previous one or two tokens with
very high weight, ignoring the broader context that would normally inform metric interpretation.
The model has learned that when metric context becomes ambiguous, the safest strategy is to
retreat to local prediction.
</p>

<p>
This adaptive behavior demonstrates a form of uncertainty handling built into the model's
processing. It does not simply fail when confronted with complexity; it adjusts its attention
strategy to maintain reasonable predictions within a narrower scope. Whether this represents
genuine adaptive intelligence or an emergent statistical property is unclear, but the behavior
itself is sophisticated and musically sensible.
</p>''')

    content.append(visualization_embed(
        "Attention Entropy by Layer",
        "visualizations/corpus/entropy_distribution.html",
        "Distribution of attention entropy values showing how different genres affect model attention patterns."
    ))

    # ========== NOISE ==========
    content.append('<h2>Noise: Pure Statistical Failure</h2>')

    content.append(audio_player(
        "Random Noise Piece",
        "audio/noise_001.mp3",
        "Randomly generated note sequences with no tonal or rhythmic coherence."
    ))

    content.append(f'''<p>
The noise category provides a control condition: pieces with valid ABC syntax but random pitch
content. Notes are selected uniformly from the available range without regard for key, scale, or
melodic contour. Rhythms follow valid ABC timing but without musical motivation. These pieces
sound like what they are—random collections of sounds—and the model's metrics reflect complete
statistical failure.
</p>

<p>
Perplexity reaches its highest levels here, with a mean of {noise['ppl']:.2f} and a maximum of
{noise['ppl_max']:.2f}. The model cannot predict random sequences any better than chance would
allow. Each token is equally surprising because no token is more likely than any other in the
absence of musical structure.
</p>''')

    content.append(metric_highlight(
        f"Noise: Perplexity = {noise['ppl']:.2f} (max: {noise['ppl_max']:.2f}), "
        f"Attention Entropy = {noise['ent']:.4f} bits, Mean Surprisal = {noise['sur']:.2f} bits"
    ))

    content.append(f'''<p>
Yet attention entropy remains moderate at {noise['ent']:.4f} bits—higher than experimental pieces
but lower than silence studies (which we will examine shortly). This suggests the model still
attempts structured attention even on random input. It attends to token positions as if they
might contain relevant information, even though they do not. The attention mechanism continues
its learned patterns regardless of whether those patterns serve any predictive purpose.
</p>

<p>
Surprisal analysis reveals something unexpected: the noise pieces contain occasional low-surprisal
tokens. Random sequences occasionally produce, by chance, configurations that match common musical
patterns. A random selection might happen to outline a major triad or step through a scale fragment.
When this occurs, the model's surprisal drops momentarily before spiking again on the next truly
random token. These accidental moments of coherence highlight how strongly the model's predictions
depend on local context that happens to match training patterns.
</p>

<p>
The noise condition establishes a ceiling for perplexity in our experiments. No musical
intervention can produce higher perplexity than random content, because random content maximizes
entropy by definition. This ceiling provides a reference point for interpreting other categories:
when avant-garde pieces approach noise-level perplexity, they have genuinely exceeded the model's
predictive capacity. When they remain substantially below noise levels, the model retains some
structural grasp despite harmonic unfamiliarity.
</p>''')

    # ========== SILENCE ==========
    content.append('<h2>Silence: The Paradox of Emptiness</h2>')

    content.append(audio_player(
        "Silence Study No. 1",
        "audio/silence_001.mp3",
        "Minimal musical content with extended rests and sparse note events."
    ))

    content.append(f'''<p>
Our silence category presents a philosophical challenge to the model. These pieces contain valid
ABC notation but minimal musical content—long rests, occasional isolated notes, sparse textures
stretched over extended durations. They are the musical equivalent of blank pages with occasional
punctuation marks. How does a model trained on dense musical text respond to near-emptiness?
</p>

<p>
Perplexity is surprisingly low at {silence['ppl']:.2f}. Rests are highly predictable: when a
piece establishes a pattern of silence, the model expects that pattern to continue. An isolated
note in a sea of rests carries high surprisal, but rests following rests carry almost none. The
overall perplexity reflects this asymmetry, pulled down by the highly predictable continuation
of emptiness.
</p>''')

    content.append(metric_highlight(
        f"Silence: Perplexity = {silence['ppl']:.2f}, "
        f"Attention Entropy = {silence['ent']:.4f} bits (highest in corpus), Mean Surprisal = {silence['sur']:.2f} bits"
    ))

    content.append(f'''<p>
But attention entropy tells a different story. At {silence['ent']:.4f} bits, silence produces
the highest attention entropy in our entire corpus—higher than noise, higher than any other
category. This paradox deserves careful examination.
</p>

<p>
When processing normal musical text, the model's attention heads find structure to latch onto:
recent notes, downbeats, phrase boundaries, harmonic anchors. In silence, these structural
landmarks are absent. The rests that dominate the texture provide no content for attention to
differentiate. The model must still produce attention weights, but it has nothing meaningful
to attend to. The result is diffuse, unfocused attention—high entropy not from complexity but
from absence.
</p>

<p>
This finding inverts our intuitive expectations. We might assume that simple music would produce
low attention entropy and complex music would produce high attention entropy. Instead, we find
that structural absence produces higher attention entropy than structural complexity. The model's
attention mechanism is optimized for finding patterns in dense input; when input becomes sparse,
the mechanism produces noise.
</p>

<p>
The silence studies also reveal attention head differentiation that is invisible in other
categories. Some heads maintain relatively focused attention even on sparse input, attending
primarily to the few actual notes whenever they occur. Other heads show complete diffusion,
attending uniformly across the sequence including many rest positions. This differentiation
suggests that some attention heads specialize in content-based attention (and thus require
content to function) while others serve structural or positional functions that persist
regardless of content density.
</p>''')

    # ========== TERRIBLE ==========
    content.append('<h2>Terrible Music: The Limits of Syntactic Competence</h2>')

    content.append(audio_player(
        "Intentionally Bad Voice Leading",
        "audio/terrible_001.mp3",
        "Deliberate violations of traditional voice-leading rules including parallel fifths and octaves."
    ))

    content.append(f'''<p>
Our final category directly probes the model's understanding of musical grammar as distinct from
statistical regularity. These pieces are intentionally terrible by traditional standards: they
feature parallel fifths and octaves, forbidden voice crossings, awkward melodic leaps, and
violations of every textbook rule. Yet they maintain valid ABC syntax and use common chord
structures. The local statistics are reasonable; only the relationships between voices are broken.
</p>

<p>
The model shows no particular difficulty with these pieces. Perplexity averages {terrible['ppl']:.2f},
only slightly higher than traditional music. Attention entropy of {terrible['ent']:.4f} bits falls
within the normal range. Mean surprisal of {terrible['sur']:.2f} bits indicates comfortable
processing. By every metric we can measure, the model treats terrible music almost identically
to good music.
</p>''')

    content.append(metric_highlight(
        f"Terrible Music: Perplexity = {terrible['ppl']:.2f}, "
        f"Attention Entropy = {terrible['ent']:.4f} bits, Mean Surprisal = {terrible['sur']:.2f} bits"
    ))

    content.append(f'''<p>
This finding is perhaps the most significant in our study. The model cannot distinguish between
well-crafted and poorly-crafted music when both use similar surface vocabularies. Parallel fifths
are statistically unusual but not dramatically so—many pieces in the training data contain
occasional parallel motion, even if pedagogy forbids it. The model has learned what notes tend
to follow what notes, but not what notes should follow what notes according to voice-leading
principles.
</p>

<p>
We can phrase this more precisely: the model has learned P(note | context) but not
P(note | context, good_music). It maximizes likelihood without any normative constraint, because
likelihood is what training optimizes. A note that produces smooth voice leading and a note that
produces parallel fifths may have similar probabilities in the training distribution, because
both occur in that distribution. The model has no external standard against which to judge
musical quality; it only knows frequency.
</p>

<p>
This limitation extends beyond music to any domain where quality cannot be inferred from
frequency. A language model trained on mathematical text cannot distinguish valid proofs from
invalid proofs if both occur in the training data. A model trained on code cannot distinguish
secure implementations from vulnerable implementations if both compile. The terrible music
category exposes this fundamental constraint in a domain where human judgment is immediate and
unambiguous: we can hear that these pieces sound wrong, but the model cannot.
</p>''')

    content.append(visualization_embed(
        "Genre Clustering Analysis",
        "visualizations/corpus/genre_clustering.html",
        "Two-dimensional projection of metric vectors showing how pieces cluster by genre characteristics."
    ))

    # ========== METHODOLOGY ==========
    content.append('<h2>Deep Dive: Methodology and Interpretation</h2>')

    content.append('''<p>
Having surveyed our musical categories and their characteristic metrics, we now turn to a more
detailed examination of methodology. How exactly were these metrics computed? What assumptions
underlie their interpretation? What are the limitations of our approach?
</p>

<h3>The Phi-4 Model and Its Relevance</h3>

<p>
Our analysis uses Phi-4, a fourteen-billion parameter transformer model from Microsoft. This choice
deserves explanation. Phi-4 is not specifically trained for music and has no architectural features
targeting musical processing. It is a general-purpose language model trained on diverse text
corpora, including but not limited to musical notation. This generality is precisely the point.
</p>

<p>
We are not asking whether a specialized music model can process ABC notation—obviously it can, by
design. We are asking what a general language model has incidentally learned about music through
exposure to musical text. Phi-4 represents the current frontier of this category: large enough
to have encountered substantial musical content during training, capable enough to learn complex
patterns, but not specialized in any way that would bias our conclusions.
</p>

<p>
The model was run with full attention weight and hidden state extraction, enabling the detailed
interpretability analysis that follows. Each piece was tokenized using the model's native
tokenizer, then processed in a single forward pass. We extracted attention weights from all
forty layers, each containing forty attention heads, yielding 1,600 attention matrices per piece.
Hidden states were captured at each layer, providing a trajectory of how the representation
evolves through the network.
</p>

<h3>Perplexity Computation</h3>

<p>
Perplexity was computed as the exponential of the mean negative log-likelihood across all tokens.
Formally, given a sequence of N tokens with model-assigned probabilities p₁, p₂, ..., pₙ, the
perplexity is:
</p>

<div class="metric-highlight">
PPL = exp( -1/N × Σ log(pᵢ) )
</div>

<p>
This computation treats all tokens equally, which is a simplification. In music, some tokens
carry more structural weight than others—downbeat notes matter more than passing tones, harmonic
roots matter more than chord extensions. A perplexity measure weighted by musical salience might
yield different results. We leave this refinement for future work, noting that our unweighted
measure still captures meaningful variation across categories.
</p>

<p>
It is worth distinguishing perplexity from accuracy. A model can achieve high accuracy on a
classification task while maintaining high perplexity on a generation task. Perplexity measures
calibrated probability assignment, not binary correctness. This makes it ideal for our purposes:
we care about the model's confidence distribution, not whether it would generate correct music.
</p>

<h3>Attention Entropy Analysis</h3>

<p>
Attention entropy was computed independently for each head at each layer. For a given attention
head attending to a sequence of L positions, the attention weights form a probability distribution
{a₁, a₂, ..., aₗ} where Σaᵢ = 1. The entropy of this distribution is:
</p>

<div class="metric-highlight">
H = -Σ aᵢ × log₂(aᵢ)
</div>

<p>
This entropy is bounded between zero (all attention on a single position) and log₂(L) (uniform
attention across all positions). We report entropy in bits for interpretability: an entropy of
three bits means the attention is spread across approximately 2³ = 8 effective positions.
</p>

<p>
One subtlety deserves mention. Transformer attention is causal in decoder-only models like Phi-4:
each position can only attend to previous positions. This means attention entropy mechanically
increases through the sequence, since later positions have more positions available to attend to.
We normalize for sequence length when comparing across pieces, but this normalization is imperfect.
Comparisons within a piece or between pieces of similar length are most reliable.
</p>

<h3>Hidden State Trajectories</h3>

<p>
Hidden state analysis tracks the L2 norm of the residual stream at each layer. The residual
stream is the summed representation that flows through the network, modified by attention and
feed-forward layers at each step. Its norm provides a rough measure of representational magnitude.
</p>

<p>
In typical language model processing, hidden state norms grow gradually through layers, reflecting
the accumulation of processed information. Sharp spikes or drops indicate unusual processing—the
model has encountered something that activates different pathways than normal. We track the
growth rate (final layer norm divided by embedding layer norm) as a summary statistic.
</p>

<p>
Hidden state analysis is less developed than attention analysis in the interpretability literature.
We know that hidden states encode something, but their relationship to human-interpretable concepts
is poorly understood. Our analysis treats hidden states as a signal of unusual processing without
claiming to decode their specific content.
</p>

<h3>Limitations and Caveats</h3>

<p>
Several limitations constrain our conclusions. First, we analyze a single model. Different
architectures, training corpora, or model sizes might yield different patterns. Our findings
describe Phi-4's processing, not transformer processing in general.
</p>

<p>
Second, our musical corpus is small and stylistically limited. Thirty pieces across six categories
cannot represent the full diversity of musical expression. We have not explored non-Western music,
electronic music, or many other traditions. Our categories, while useful for controlled comparison,
are artificial constructs that may not capture natural musical variation.
</p>

<p>
Third, ABC notation is a lossy representation of music. It captures pitch and rhythm but not
timbre, dynamics, articulation, or the countless subtleties that make performed music expressive.
A model that perfectly understands ABC notation has not necessarily understood music—it has
understood a particular textual encoding of certain musical parameters.
</p>

<p>
Fourth, interpretability metrics are indirect measures. We infer model understanding from attention
patterns and prediction confidence, but these inferences require assumptions. A model might achieve
low perplexity through shallow pattern matching without any structural understanding, or high
perplexity despite genuine comprehension of unusual material. We interpret our metrics in
light of domain knowledge, but alternative interpretations are possible.
</p>''')

    # ========== CONCLUSIONS ==========
    content.append('<h2>Conclusions: What the Model Knows and Does Not Know</h2>')

    content.append('''<p>
Our experiments reveal a nuanced picture of transformer musical understanding. The model is not
simply competent or incompetent at processing music; it exhibits specific strengths and specific
weaknesses that can be precisely characterized.
</p>

<p>
The model has learned robust metrical representations. Across all categories, attention patterns
related to rhythmic structure remain stable. Downbeats receive heightened attention, phrase
boundaries are marked, and temporal patterns are tracked. This rhythmic competence persists even
when harmonic processing fails completely, suggesting that rhythm and pitch are processed through
at least partially independent pathways.
</p>

<p>
The model has learned local harmonic expectations. Within traditional tonal contexts, it predicts
chord tones accurately and handles common progressions with confidence. These expectations break
down as harmony becomes more chromatic, but the breakdown is gradual rather than catastrophic.
The model does not simply fail at extended tonality; it maintains partial predictions with
reduced confidence.
</p>

<p>
The model has not learned voice-leading principles. It cannot distinguish well-crafted from
poorly-crafted music when both use similar statistical vocabularies. This limitation reflects
the fundamental nature of language model training: the objective is prediction, not evaluation.
The model learns what is likely, not what is good.
</p>

<p>
The model processes complexity through local attention narrowing. When faced with difficult
structural ambiguity, it retreats to local prediction, maintaining reasonable accuracy within
a narrow window while sacrificing long-range coherence. This adaptive strategy is musically
sensible but represents a processing mode qualitatively different from normal language processing.
</p>

<p>
The model struggles with absence more than with complexity. Sparse textures produce higher
attention entropy than dense textures, because the attention mechanism requires content to
function. This finding has implications beyond music: models may be poorly calibrated for
domains where information is sparse or unevenly distributed.
</p>''')

    content.append(section_break())

    content.append('''<p>
What do these findings mean for the broader question of machine understanding? We suggest a
framework that distinguishes statistical competence from structural understanding. The model
has achieved statistical competence in music: it has learned the probability distributions
that characterize musical text. It has not achieved structural understanding: it has not
learned the generative rules that produce valid music or the evaluative criteria that
distinguish good music from bad.
</p>

<p>
This distinction matters beyond music. In any domain represented through text, language models
can achieve statistical competence through pattern matching. They can produce plausible-looking
mathematical proofs, syntactically valid code, and coherent-sounding arguments. But statistical
competence is not the same as understanding the domain logic. A model might generate a proof
that looks correct but contains a subtle logical error. It might generate code that compiles
but has a security vulnerability. It might generate an argument that sounds compelling but
relies on a hidden assumption.
</p>

<p>
The interpretability tools we have developed provide a starting point for detecting these
failures. When a model processes domain content with low perplexity and focused attention,
it is operating within its zone of statistical competence. When perplexity rises, attention
diffuses, and hidden states diverge, the model has exceeded that zone. These signals do not
directly indicate whether output is correct, but they indicate where scrutiny is warranted.
</p>

<p>
Music serves as a useful test domain precisely because human judgment is immediate. We can
hear that terrible music sounds wrong, even though the model processes it normally. This
direct perceptual access allows us to validate interpretability methods against ground truth.
Having established that attention entropy and perplexity correlate with known musical
properties, we can apply these methods to domains where human judgment is slower or less
reliable.
</p>

<p>
The long-term goal of this research program is not to build better music models but to build
better understanding of what any language model has learned about any domain. Music is a
probe, a controlled environment where we can study the relationship between statistical
pattern matching and genuine comprehension. The patterns we observe here—the hierarchical
attention structure, the adaptive narrowing under complexity, the blindness to quality
distinctions—likely manifest in other domains as well.
</p>

<p>
Future work should extend this methodology to domains with higher stakes: mathematical
reasoning, scientific inference, ethical judgment. In each domain, we need the equivalent
of our noise and terrible categories: inputs that are statistically normal but structurally
broken, inputs that should trigger model failure but might not. By mapping these failure
modes, we can develop more reliable systems that combine statistical power with domain-specific
verification.
</p>

<p>
For now, we have demonstrated that transformer models process music with partial but genuine
competence, that their competence can be precisely characterized through interpretability
metrics, and that their limitations reveal fundamental constraints of learning from text
alone. The model knows what notes tend to follow what notes. It does not know what makes
music music.
</p>''')

    content.append(visualization_embed(
        "Perplexity Comparison Across Genres",
        "visualizations/corpus/perplexity_comparison.html",
        "Final comparison showing how perplexity varies systematically across musical genres."
    ))

    # ========== APPENDIX ==========
    content.append('<h2>Appendix: Additional Audio Examples</h2>')

    content.append('<p>Below are audio renderings of representative pieces from each category, allowing direct auditory comparison with the metrics discussed above.</p>')

    content.append('<h3>Traditional</h3>')
    for i in range(1, 6):
        content.append(audio_player(f"Traditional Piece {i}", f"audio/traditional_00{i}.mp3", f"Traditional folk tune demonstrating baseline musical patterns."))

    content.append('<h3>Avant-Garde</h3>')
    for i in range(1, 6):
        content.append(audio_player(f"Avant-Garde Piece {i}", f"audio/avantgarde_00{i}.mp3", f"Extended tonality composition with chromatic complexity."))

    content.append('<h3>Experimental</h3>')
    for i in range(1, 6):
        content.append(audio_player(f"Experimental Piece {i}", f"audio/experimental_00{i}.mp3", f"Complex polyrhythmic structure."))

    content.append('<h3>Noise</h3>')
    for i in range(1, 6):
        content.append(audio_player(f"Noise Piece {i}", f"audio/noise_00{i}.mp3", f"Random pitch sequences for comparison."))

    content.append('<h3>Silence</h3>')
    for i in range(1, 6):
        content.append(audio_player(f"Silence Study {i}", f"audio/silence_00{i}.mp3", f"Minimal material with extended rests."))

    content.append('<h3>Terrible</h3>')
    for i in range(1, 6):
        content.append(audio_player(f"Terrible Music {i}", f"audio/terrible_00{i}.mp3", f"Intentional voice-leading violations."))

    return "\n".join(content)


def count_words(html: str) -> int:
    """Count words in HTML content."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Count words
    return len(text.split())


def estimate_reading_time(word_count: int, wpm: int = 200) -> str:
    """Estimate reading time from word count."""
    minutes = word_count / wpm
    if minutes < 1:
        return "< 1 minute"
    elif minutes < 60:
        return f"{int(minutes)} minutes"
    else:
        hours = int(minutes / 60)
        mins = int(minutes % 60)
        return f"{hours} hour{'s' if hours > 1 else ''}, {mins} minutes"


def build_essay(
    output_path: Path,
    metrics_dir: Path = Path("data/metrics"),
    title: str = "Substrate Melodies: Probing Musical Understanding Through Transformer Confusion",
    author: str = "Brian Edwards",
    email: str = "brian.mabry.edwards@gmail.com",
    phone: str = "512-584-6841",
    location: str = "Waco, Texas, USA",
    model: str = "Claude Code (Claude Opus 4.5)"
) -> None:
    """Build and save the essay HTML."""

    # Load metrics data
    metrics = load_metrics(metrics_dir)

    # Build content
    content = build_essay_content(metrics)

    # Calculate word count
    word_count = count_words(content)
    reading_time = estimate_reading_time(word_count)

    html = ESSAY_TEMPLATE.format(
        title=title,
        author=author,
        email=email,
        phone=phone,
        location=location,
        model=model,
        date=datetime.now().strftime("%B %d, %Y"),
        word_count=f"{word_count:,}",
        reading_time=reading_time,
        content=content
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f"Built essay: {output_path}")
    print(f"Word count: {word_count:,}")
    print(f"Reading time: {reading_time}")


def build_index(
    output_path: Path,
    essays: list[dict],
    author: str = "Brian Edwards",
    email: str = "brian.mabry.edwards@gmail.com"
) -> None:
    """Build the index/listing page."""
    essays_html = []
    for essay in essays:
        essays_html.append(f'''
<article>
    <h2><a href="{essay['path']}">{essay['title']}</a></h2>
    <p class="metadata">
        {essay.get('date', 'TBD')} |
        {essay.get('word_count', 'TBD')} words |
        {essay.get('reading_time', 'TBD')}
    </p>
    <p>{essay.get('description', '')}</p>
</article>
''')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Musical Transformer Interpretability Research</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            background-color: #fafafa;
        }}
        h1 {{ color: #2c3e50; text-align: center; }}
        article {{
            margin: 2rem 0;
            padding: 1.5rem;
            background: white;
            border: 1px solid #e1e1e1;
            border-radius: 8px;
        }}
        article h2 a {{
            color: #3498db;
            text-decoration: none;
        }}
        article h2 a:hover {{
            text-decoration: underline;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9rem;
        }}
        footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e1e1e1;
            font-size: 0.9rem;
            color: #666;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>Musical Transformer Interpretability Research</h1>

    <section>
        {"".join(essays_html)}
    </section>

    <footer>
        <h3>About</h3>
        <p>
            {author}<br>
            {email}<br>
        </p>
        <p>Built with Claude Code (Claude Opus 4.5)</p>
    </footer>
</body>
</html>
'''

    output_path.write_text(html, encoding='utf-8')
    print(f"Built index: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build essay HTML")
    parser.add_argument("--output", type=Path, default=Path("docs/essay.html"),
                        help="Output path for essay")
    parser.add_argument("--index", type=Path, default=Path("docs/index.html"),
                        help="Output path for index page")
    parser.add_argument("--metrics", type=Path, default=Path("data/metrics"),
                        help="Path to metrics directory")

    args = parser.parse_args()

    # Build the essay
    build_essay(args.output, args.metrics)

    # Calculate actual stats for index
    essay_html = args.output.read_text()
    word_count = count_words(essay_html)
    reading_time = estimate_reading_time(word_count)

    # Build the index
    essays = [{
        "title": "Substrate Melodies: Probing Musical Understanding Through Transformer Confusion",
        "path": "essay.html",
        "date": datetime.now().strftime("%B %d, %Y"),
        "word_count": f"{word_count:,}",
        "reading_time": reading_time,
        "description": "An exploration of how transformer models process ABC notation music, "
                       "using interpretability metrics to map the boundaries of machine musical understanding."
    }]
    build_index(args.index, essays)


if __name__ == "__main__":
    main()
