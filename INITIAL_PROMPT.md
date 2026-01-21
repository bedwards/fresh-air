# Claude Code Project Prompt: AI Music Analysis Through Transformer Interpretability

## Project Overview

You will build a complete system that generates ABC notation music files across various styles (traditional, avant-garde, experimental, noise, silence, "terrible"), analyzes them using transformer model interpretability techniques, and produces a scholarly essay with interactive audio players and visualizations. The essay will be deployed as a professional GitHub Pages site.

## Technical Stack

- **Model**: Phi-4 14B (Q4_0 quantized) via llama.cpp or Ollama
- **Analysis**: Transformer interpretability metrics (perplexity, attention entropy, surprisal, layer activations, sparse autoencoders)
- **Visualization**: BertViz, custom Polars + Altair charts
- **Output**: Professional essay with embedded audio players, deployed via GitHub Pages
- **Workflow**: Main manager Claude Code instance orchestrating background worker instances via GitHub Issues

## Architecture

### Phase 1: Infrastructure Setup (Main Instance - Interactive)

**Before starting any work:**
1. Check what's already installed: `which ollama`, `pip list | grep transformers`, `brew list`, etc.
2. Never downgrade system packages - if stuck, ask the human for help
3. Create comprehensive documentation as you build

**Initial setup tasks:**
1. Initialize Git repository with proper `.gitignore`
2. Create directory structure:
   ```
   ├── README.md                 # Project overview for developers
   ├── CLAUDE.md                 # Instructions for Claude instances
   ├── ARCHITECTURE.md           # System design documentation
   ├── scripts/
   │   ├── setup.sh             # Environment setup
   │   ├── count_words.py       # Word count utility
   │   ├── estimate_reading_time.py
   │   └── deploy.sh            # GitHub Pages deployment
   ├── src/
   │   ├── abc_generator.py     # ABC notation generation
   │   ├── model_harness.py     # Phi-4 inference wrapper
   │   ├── metrics_analyzer.py  # Interpretability metrics
   │   ├── visualizations.py    # Polars + Altair viz
   │   └── essay_builder.py     # HTML essay generation
   ├── data/
   │   ├── abc_files/           # Generated ABC notation
   │   ├── metrics/             # JSON metrics per song
   │   └── audio/               # MP3 placeholders + final audio
   ├── docs/                     # GitHub Pages site
   │   ├── index.html           # Listing page
   │   ├── essay.html           # Main essay
   │   ├── assets/              # CSS, JS, images
   │   └── visualizations/      # Embedded charts
   └── tests/
   ```

3. Install dependencies:
   - Check if Ollama installed: `which ollama`
   - If not: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Pull Phi-4: `ollama pull phi4:14b-q4_0`
   - Python packages: `transformers`, `torch`, `bertviz`, `polars`, `altair`, `numpy`, `scipy`
   - Check Python version compatibility before installing

4. Create `CLAUDE.md` with:
   - How to activate this project context
   - Where critical files live
   - Common commands
   - GitHub Issue workflow for workers

5. Set up Git worktrees strategy for parallel work:
   ```bash
   git worktree add ../music-analysis-worker-1 -b worker-1
   git worktree add ../music-analysis-worker-2 -b worker-2
   ```

### Phase 2: ABC Music Generation (GitHub Issue → Worker)

**Main instance creates GitHub Issue:**
```
Title: Generate ABC notation corpus across genres
Assignee: @worker-1
Labels: generation, abc-notation

Description:
- Generate 20+ ABC notation files covering:
  * Traditional: folk tunes, hymns, classical themes (5 files)
  * Avant-garde: unusual time signatures, microtonal hints (3 files)
  * Experimental: extreme register jumps, dense polyphony (3 files)
  * Noise: random note sequences, no tonal center (3 files)
  * Terrible: intentionally bad voice leading, parallel fifths (3 files)
  * Silence: rests, sparse notes, minimal material (3 files)

- Save to data/abc_files/ with metadata JSON:
  {
    "filename": "traditional_001.abc",
    "genre": "traditional",
    "description": "Irish jig in D major",
    "voices": 2,
    "time_signature": "6/8",
    "key": "D"
  }

- Include multi-voice pieces (up to 4 voices for polyphonic examples)
- Document generation approach in data/abc_files/README.md
```

**Worker deliverables:**
- ABC files with clear naming convention
- Metadata JSON sidecar files
- Documentation of generation strategy
- Pull request to merge back to main

### Phase 3: Model Harness Development (GitHub Issue → Worker)

**Main instance creates GitHub Issue:**
```
Title: Build Phi-4 analysis harness with interpretability metrics
Assignee: @worker-2
Labels: core-infrastructure, metrics

Description:
Develop src/model_harness.py that:

1. Loads Phi-4 14B (Q4_0) via Ollama API or llama.cpp
2. Tokenizes ABC notation text
3. Runs inference with output_attentions=True, output_hidden_states=True
4. Computes for each ABC file:
   
   A. **Perplexity Metrics**
   - Overall perplexity
   - Per-token surprisal (negative log probability)
   - Identify tokens with surprisal > 10 (high surprise)
   
   B. **Attention Analysis**
   - Attention entropy per head per layer: H = -Σ(p log p)
   - Mean attention entropy across all heads
   - Attention head specialization: which heads attend to position vs. content
   - Identify heads with anomalous entropy (>4.5 bits)
   
   C. **Hidden State Analysis**
   - Layer-wise activation statistics (mean, variance, L2 norm)
   - Track activation trajectory across layers
   - Detect sudden spikes/drops indicating unusual processing
   
   D. **Token-Level Features**
   - Per-token perplexity
   - Attention weight distributions
   - Layer activation patterns
   
   E. **Comparative Metrics**
   - Z-scores relative to traditional music baseline
   - Flag outliers (>2 standard deviations)

4. Save comprehensive metrics to data/metrics/{filename}.json:
   {
     "filename": "experimental_002.abc",
     "perplexity": 347.2,
     "mean_attention_entropy": 4.8,
     "high_surprisal_tokens": [...],
     "layer_activations": [...],
     "attention_patterns": {...},
     "comparative_z_scores": {...}
   }

5. Create src/metrics_analyzer.py for metric interpretation:
   - Classify pieces by metric profiles
   - Generate human-readable descriptions:
     * "Syntactically coherent but semantically chaotic"
     * "Low positional attention, high content confusion"
     * "Stable early layers, divergence at layer 18+"
   
6. Document metric calculations in docs/METRICS.md with formulas
7. Include unit tests in tests/test_harness.py
```

### Phase 4: Visualization System (GitHub Issue → Worker)

**Main instance creates GitHub Issue:**
```
Title: Build visualization suite for interpretability metrics
Assignee: @worker-1
Labels: visualization, analysis

Description:
Create src/visualizations.py with both library-based and custom viz:

**Using BertViz (for attention):**
1. Head view: attention flow within layers
2. Model view: bird's eye across all layers
3. Neuron view: query/key vector details
4. Export as interactive HTML widgets

**Custom Polars + Altair charts:**
1. Perplexity comparison: bar chart across all pieces
2. Attention entropy heatmap: [layer, head] grid per piece
3. Token surprisal timeline: line chart showing surprisal per token position
4. Layer activation trajectory: multi-line chart tracking norms across layers
5. Genre clustering: 2D projection (UMAP/PCA) of metric vectors
6. Comparative distributions: overlapping histograms for traditional vs. experimental

**Visualization requirements:**
- All charts must be self-contained HTML (Altair to_html())
- No external dependencies for deployed site
- Consistent color scheme (define palette in config)
- Accessible: proper alt text, colorblind-friendly
- Responsive: work on mobile and desktop
- Include data tables as fallback

**Output:**
- Save visualizations to docs/visualizations/{piece_name}/
- Each piece gets own folder with all its charts
- Create manifest JSON listing all available visualizations
- Document in docs/VISUALIZATIONS.md
```

### Phase 5: Iterative Generation (GitHub Issue → Worker)

**Main instance creates GitHub Issue:**
```
Title: Generate second-round ABC files informed by metrics
Assignee: @worker-2
Labels: generation, iteration

Description:
Analyze first-round metrics to identify:
1. What makes perplexity spike (note patterns, rhythm, structure)
2. Which attention heads get confused (specific musical features)
3. Where layer activations diverge (complexity thresholds)

Generate new ABC files that:
- **Maximize perplexity**: target >400 by design
- **Confuse syntax heads**: valid ABC syntax but unusual musical grammar
- **Overload semantic heads**: meaningful locally but incoherent globally
- **Target specific layers**: trigger divergence at predicted layer depth

Create 10 new "designed" pieces with hypothesis for each:
- Hypothesis: "Dense chromaticism will spike attention entropy in heads 8-12"
- Result: [metrics confirm/refute]

Document learnings in data/abc_files/ITERATION_LEARNINGS.md
```

### Phase 6: Essay Generation (Main Instance)

**The essay structure (all prose, no bullets in final output):**

```
Title: "Substrate Melodies: Probing Musical Understanding Through Transformer Confusion"

Author: Brian Edwards
Contact: brian.mabry.edwards@gmail.com | 512-584-6841 | Waco, Texas, USA
Date: January 21, 2026
Model: Claude Code (Claude Opus 4.5)
Word Count: [SCRIPT OUTPUT]
Reading Time: [SCRIPT OUTPUT]

=== Introduction ===
[2000-3000 words]

Begin from first principles: what does it mean for a language model to "understand" music? 
When we feed ABC notation—a textual representation of musical information—into a 
transformer architecture designed for natural language, we create a collision between 
two domains. The model has seen ABC in its training data, but has it learned the deep 
structural relationships that make music coherent? Or has it merely memorized surface 
patterns?

This essay presents an experimental methodology for answering these questions. Rather 
than asking the model to generate music, we invert the problem: we generate music 
designed to confuse the model in specific, measurable ways. By analyzing where and how 
the model struggles, we can map the boundaries of its musical comprehension.

[Explain interpretability metrics without jargon]
- Perplexity is not just "model uncertainty"—it's the effective branching factor, the 
  number of equally plausible next tokens. In English text, good models achieve perplexity 
  of fifty to one hundred. We will see values exceeding four hundred.
  
- Attention entropy measures where the model "looks" when processing each token. Low 
  entropy means focused attention (this note clearly relates to that chord). High entropy 
  means the model is lost, attending uniformly across the entire sequence because nothing 
  stands out as relevant.

- Surprisal quantifies shock: how unexpected is each token given what came before? A 
  well-trained model assigns high probability to the actual next token, yielding low 
  surprisal. When surprisal exceeds ten bits, the model is encountering something it 
  never imagined.

[Outline the experimental approach]
We generated twenty-six pieces of ABC notation spanning six categories: traditional, 
avant-garde, experimental, noise, terrible, and silence. Each piece was analyzed using 
Phi-4 14B, a fourteen-billion parameter transformer model quantized to four-bit precision. 
We extracted attention weights across all thirty-two layers and ninety-six attention 
heads, computed hidden state activations, and calculated comprehensive interpretability 
metrics.

The results reveal a nuanced picture. The model is not simply "good" or "bad" at music. 
Instead, it exhibits specific competencies and specific blind spots, which we can now 
describe with precision.

=== Traditional Baseline: "The Butterfly" ===
[1500-2000 words]

[Audio Player: butterfly.mp3]

We begin with a traditional Irish slip jig in E-flat major, chosen because it represents 
the kind of folk tune the model has certainly encountered during training. The piece 
features two voices in parallel thirds, a common texture in Celtic music, with a clear 
AABB structure and conventional harmonic rhythm.

The model's response is unremarkable—precisely the goal of this baseline. Perplexity 
sits at sixty-two, well within the normal range for coherent text. The attention patterns 
show expected behavior: positional heads in early layers track the regular 9/8 meter, 
while content heads in middle layers appear to bind notes within harmonic units. By layer 
twenty-four, the model's internal representation has stabilized, evidenced by low variance 
in activation norms.

[Visualization: attention heatmap showing regular patterns]

The attention heatmap reveals structure the model has learned to expect. Heads in layer 
eight attend strongly to downbeats, creating vertical stripes every ninth token. Heads 
in layer sixteen attend to the beginning of each phrase, marked by longer note values 
and harmonic changes. This is syntactic understanding: the model knows where phrases 
begin and end, where strong beats fall, where tensions resolve.

Token-level surprisal rarely exceeds four bits. The highest surprisal occurs at the 
modulation to the relative minor in the B section—a moment of genuine harmonic interest, 
but one the model handles gracefully. The surprisal spike to 5.2 bits reflects 
uncertainty about which chord tone will appear next, not confusion about musical grammar.

[Visualization: token surprisal timeline]

What does this baseline tell us? The model has internalized a significant amount of 
musical knowledge. It knows that certain note patterns are likely, that rhythms should 
align with meter, that phrases have beginnings and endings. This is not rote memorization—
the model generalizes these patterns to novel configurations of familiar elements.

But we can already see the limits. The model processes ABC notation token by token, 
attending to local context. It does not appear to represent global structure—the 
thirty-two-bar form, the motivic relationships between sections, the long-range 
harmonic arc. These emerge from local decisions that happen to align with musical 
convention, not from explicit structural understanding.

=== Experimental Extremes: "Chromatic Density Study" ===
[2000-2500 words]

[Audio Player: chromatic_density.mp3]

The second piece represents a controlled departure from tradition. We constructed a 
four-voice chorale where each voice moves independently in chromatic half-steps, 
avoiding any tonal center. Rhythmically, the piece remains conventional—quarter notes 
in 4/4 time. Harmonically, it is chaos.

Perplexity explodes to 418. The model is confronted with a vocabulary problem: in 
traditional tonal music, following a C-natural with a C-sharp is rare and contextually 
constrained. Here, chromaticism is relentless. Every token defies the statistical 
patterns the model learned during training.

[Visualization: layer-wise activation variance]

The activation trajectory tells a story of processing breakdown. Through the first 
ten layers, activation norms remain stable—the model is still trying to parse the 
input using familiar strategies. At layer twelve, variance spikes dramatically. The 
model has detected that normal tonal relationships do not hold, but it has no alternative 
framework. Activations in different dimensions pull in conflicting directions as the 
model attempts multiple incompatible interpretations simultaneously.

By layer twenty-two, activations have diverged wildly from the traditional baseline. 
The L-two norm is three point seven times higher, indicating the model is devoting 
substantial internal resources to processing this unexpected input. Yet perplexity 
remains elevated, suggesting these resources are not yielding confident predictions.

[Visualization: attention entropy across heads]

Attention entropy provides the most striking evidence of confusion. In traditional 
music, mean attention entropy across all heads was 2.4 bits. Here, it reaches 5.1 bits. 
Several heads exhibit near-maximal entropy, attending almost uniformly across the 
entire sequence. The model cannot identify which previous tokens are relevant to 
predicting the current token, because chromatic voice leading creates no stable 
harmonic context.

Interestingly, not all heads are equally confused. Positional heads in early layers 
maintain low entropy—they still track the regular quarter-note pulse, which remains 
intact. The confusion is concentrated in semantic heads in layers fourteen through 
twenty-six, precisely the layers where harmonic processing should occur.

[Visualization: head-specific entropy comparison]

This differential confusion reveals the model's internal architecture. Some attention 
heads have learned to track low-level features (rhythm, articulation) independent of 
higher-level structure. Other heads expect harmonic coherence and fail when it is absent. 
The model is not a monolithic entity that either understands or does not understand music. 
It is a coalition of specialized mechanisms, some of which work fine and some of which 
break catastrophically when their assumptions are violated.

Token-level analysis shows where the model is most surprised. The initial chromatic 
descent in the soprano voice registers moderate surprisal—six to seven bits per token—
because descending chromatic lines do appear in tonal music as passing motion. But when 
all four voices move chromatically simultaneously, surprisal jumps to twelve point three 
bits. The model has never encountered this configuration and has no framework for 
predicting what comes next.

=== [Continue with 8-10 more pieces, each 1500-3000 words] ===

Pieces to include:
- Silence study (mostly rests)
- Polymetric experiment (simultaneous 5/8 and 7/8)
- Microtonal approximation (quarter-tone bends encoded as grace notes)
- "Terrible" piece (parallel fifths, voice crossing, bad ranges)
- Noise piece (random notes)
- Second-iteration pieces designed from first-round learnings

=== Deep Dive: Methodology and Interpretation ===
[4000-5000 words]

[Explain technical details with precision but clarity]

How we computed perplexity. The formula itself: the exponential of the mean negative 
log-likelihood. But more importantly, what this means. When the model predicts the 
next token in the sequence, it outputs a probability distribution over its vocabulary 
of fifty thousand tokens. In an ABC notation context, only a tiny fraction of these 
are plausible—musical notes, rhythm markers, structural symbols. But the model must 
still navigate this vast space.

Perplexity collapses the entire distribution into a single number: the effective 
branching factor. A perplexity of one hundred means the model behaves as if it is 
choosing uniformly among one hundred equally likely options at each step. In reality, 
the distribution is far from uniform—some tokens have probability 0.3, others 0.001, 
most near zero—but the weighted average uncertainty is equivalent to a uniform choice 
among one hundred.

[Distinguish from related concepts]

Perplexity is not accuracy. A model can achieve perfect accuracy on a classification 
task while maintaining high perplexity on a generation task. Perplexity measures 
calibrated probability assignment, not binary correctness. This makes it ideal for 
our purposes: we do not care whether the model can compose music (it cannot), we care 
about its internal confidence when processing music.

Perplexity is also distinct from cross-entropy, though related. Cross-entropy is the 
negative log-likelihood itself, measured in bits or nats. Perplexity is the exponential 
of cross-entropy, transforming the metric into an intuitively meaningful scale. Both 
measure the same underlying quantity—model uncertainty—but perplexity has a clearer 
interpretation.

[Continue explaining other metrics]

Attention entropy. Surprisal. Layer activations. Sparse autoencoders (if we implemented 
them). Each gets the same treatment: mathematical definition, conceptual meaning, 
relationship to other metrics, caveats and limitations.

[Reflect on what we learned]

The model has learned a remarkable amount about musical structure from text alone. It 
knows that notes should align with beats, that phrases have cadences, that certain 
intervals are stable and others are tense. But its knowledge is shallow in specific ways. 
It processes music token-by-token with limited memory, missing long-range dependencies. 
It has no concept of timbre, dynamics, or performance. It treats all ABC notation equally, 
regardless of whether it represents a Bach chorale or a random sequence.

Most strikingly, the model cannot distinguish between musical coherence and statistical 
coherence. Our "terrible" piece—deliberately bad voice leading, parallel fifths, awkward 
ranges—yields moderate perplexity because the local statistics are fine. Each chord 
is a valid triad, each note is in a sensible range. Only the relationships between 
chords violate tonal grammar, and these are relationships the model does not represent.

Conversely, our avant-garde pieces—musically interesting experiments with extended 
tonality and rhythmic complexity—confuse the model because they are statistically unusual. 
The model cannot separate "unusual because experimental" from "unusual because wrong." 
Both deviate from training distribution patterns and both trigger high perplexity.

This reveals a fundamental limit of training language models on text corpora without 
grounding in the domain the text describes. The model learns correlations in symbol 
sequences, but not the rules those sequences encode. It can predict what token is likely 
to follow, but not what token should follow according to musical logic.

[Conclude with implications]

These findings have implications beyond music. Any domain represented in text—mathematics, 
chemistry, code—faces the same challenge. Language models can learn to produce plausible-
looking output by imitating statistical patterns, without understanding the underlying 
domain semantics. Our interpretability metrics provide tools for distinguishing genuine 
understanding from sophisticated mimicry.

In music, this matters less because we have human musicians who can judge coherence 
directly. In domains where human judgment is harder or where models are deployed without 
human oversight, the stakes are higher. A model that generates plausible-looking 
mathematical proofs but with subtle logical errors, or plausible-looking code with 
security vulnerabilities, fails in precisely the way our music model fails: local 
coherence without global soundness.

The solution is not to abandon language models but to recognize their specific strengths 
and limits. They are powerful pattern matchers, but pattern matching is not understanding. 
By developing interpretability tools like those presented here, we can diagnose where 
models succeed and where they fail, building more robust systems that combine statistical 
learning with domain knowledge.
```

### Phase 7: Site Deployment (Main Instance)

1. Use professional GitHub Pages theme (e.g., Minimal Mistakes, Just the Docs)
2. Create responsive layout with:
   - Clean typography for long-form reading
   - Embedded audio players with waveform visualization
   - Inline visualization iframes
   - Smooth scrolling, navigation menu
   - Mobile-friendly

3. Homepage (docs/index.html):
   ```html
   Title: "Musical Transformer Interpretability Research"
   
   Essays:
   - "Substrate Melodies: Probing Musical Understanding Through Transformer Confusion"
     Date: January 21, 2026
     Word Count: [SCRIPT OUTPUT]
     Reading Time: [SCRIPT OUTPUT]
     Tags: transformer-interpretability, music-analysis, attention-mechanisms
     
   About:
   Brian Edwards
   brian.mabry.edwards@gmail.com
   512-584-6841
   Waco, Texas, USA
   
   Built with Claude Code (Claude Opus 4.5)
   ```

4. Metadata display on each essay page
5. Deploy script:
   ```bash
   #!/bin/bash
   # scripts/deploy.sh
   
   # Run word count
   python scripts/count_words.py docs/essay.html > /tmp/wordcount.txt
   python scripts/estimate_reading_time.py docs/essay.html > /tmp/reading_time.txt
   
   # Inject metadata into HTML
   # ... 
   
   # Deploy to GitHub Pages
   git subtree push --prefix docs origin gh-pages
   ```

## Main Manager Workflow

**Your role as main interactive Claude Code instance:**

1. **Setup Phase**: Do all initial work yourself (install, configure, create structure)

2. **Delegation Phase**: 
   - Create detailed GitHub Issues for each major task
   - Monitor worker pull requests using GitHub integration
   - Read code reviews automatically generated
   - Fire off new worker instances to address review comments
   - Create issues for non-blocking problems to handle later
   - Merge PRs when satisfied
   - Monitor main branch CI/CD

3. **Minimal Context Switching**:
   - Stay in this session as long as possible
   - Use git worktrees so workers work in parallel
   - Workers clone, branch, work, PR
   - You review, merge, continue

4. **Progress Tracking**:
   - Update project board
   - Comment on closed issues with outcomes
   - Maintain PROGRESS.md with high-level status

5. **Final Assembly**:
   - You personally write the essay using worker-generated data
   - You interpret metrics and craft narrative
   - You integrate visualizations
   - You deploy the site

## Critical Constraints

1. **Never downgrade**: If package conflicts arise, ask human for help
2. **Check before install**: Always verify what's already present
3. **Document everything**: Future Claude instances (and human) need to understand your decisions
4. **Commit frequently**: Small, atomic commits with clear messages
5. **Test incrementally**: Don't build everything before testing anything

## Human's Role

- Produce audio in Bitwig after you generate ABC files
- Drop MP3 files into data/audio/ replacing placeholders
- Approve major architectural decisions if you get stuck
- Provide feedback on essay drafts

## Deliverables Checklist

- [ ] Complete codebase in well-structured repository
- [ ] All dependencies documented and installed
- [ ] 26+ ABC notation files with metadata
- [ ] Comprehensive metrics JSON for each file
- [ ] Visualization suite (BertViz + custom Altair)
- [ ] 15,000+ word essay with flowing prose
- [ ] Professional GitHub Pages site with audio players
- [ ] README.md, CLAUDE.md, ARCHITECTURE.md
- [ ] Word count and reading time scripts
- [ ] Deployment automation
- [ ] All code reviewed via GitHub integration
- [ ] Main branch passing CI

## Getting Started

Begin by:
1. Creating the repository structure
2. Writing comprehensive CLAUDE.md for future instances
3. Setting up Python environment
4. Installing Phi-4 14B via Ollama
5. Creating first GitHub Issue for ABC generation
6. Launching worker in git worktree

Remember: You are building a research artifact that must be reproducible, well-documented, and compelling to read. Every technical decision should serve the goal of revealing how transformers process musical information.

---

**Research sources to consult** (use web search):
- Anthropic Engineering blog posts on Claude Code best practices (2025-2026)
- GitHub API for programmatic issue creation
- Ollama API documentation for Phi-4 inference
- BertViz documentation and examples
- Polars + Altair visualization tutorials
- Academic papers on transformer interpretability (search "sparse autoencoders 2025", "attention entropy interpretability")
- ABC notation formal specification

Do not oversimplify. Do not avoid scope. Do not delete things willy nilly. Do not regress functionality.

Begin now. The human is ready to produce audio files once you generate the ABC notation.