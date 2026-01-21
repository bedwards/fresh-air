#!/usr/bin/env python3
"""HTML essay builder for the research paper.

This module assembles the final essay HTML with embedded audio players,
visualizations, and metadata.

Usage:
    python src/essay_builder.py --output docs/essay.html
"""

import argparse
import json
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
            line-height: 1.8;
            color: var(--text-color);
            background-color: var(--bg-color);
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
        }}

        h1 {{
            font-size: 2.5rem;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }}

        h2 {{
            font-size: 1.8rem;
            color: var(--primary-color);
            margin-top: 3rem;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}

        h3 {{
            font-size: 1.4rem;
            color: var(--primary-color);
            margin-top: 2rem;
        }}

        .metadata {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 2rem;
            padding: 1rem;
            background-color: var(--card-bg);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .metadata span {{
            display: inline-block;
            margin-right: 1.5rem;
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

        blockquote {{
            border-left: 4px solid var(--secondary-color);
            padding-left: 1.5rem;
            margin: 1.5rem 0;
            font-style: italic;
            color: #555;
        }}

        .footnote {{
            font-size: 0.85rem;
            color: #666;
        }}

        @media (max-width: 600px) {{
            body {{
                padding: 1rem;
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
    <audio controls>
        <source src="{audio_path}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
</div>
'''


def visualization_embed(title: str, viz_path: str, caption: str = "") -> str:
    """Generate HTML for an embedded visualization."""
    return f'''
<div class="visualization">
    <iframe src="{viz_path}" title="{title}"></iframe>
    <p class="visualization-caption">{caption}</p>
</div>
'''


def section(title: str, content: str, level: int = 2) -> str:
    """Generate HTML for a section."""
    tag = f"h{level}"
    return f"<{tag}>{title}</{tag}>\n{content}"


def paragraph(text: str) -> str:
    """Wrap text in paragraph tags."""
    return f"<p>{text}</p>\n"


def build_essay_content() -> str:
    """Build the main essay content.

    This is a placeholder that will be replaced with actual essay content.
    """
    content = []

    content.append(section("Introduction", """
<p>
[This section will contain 2000-3000 words exploring what it means for a language model
to "understand" music, the collision between textual representations and musical meaning,
and the methodology of this research.]
</p>

<p>
The essay will explain interpretability metrics from first principles: perplexity as
effective branching factor, attention entropy as a measure of focused vs. diffuse attention,
and surprisal as a quantification of the unexpected.
</p>
"""))

    content.append(section("Traditional Baseline: 'The Butterfly'", """
<p>
[This section will analyze a traditional Irish slip jig, establishing the baseline for
model behavior when processing familiar musical patterns.]
</p>
""" + audio_player(
    "The Butterfly",
    "audio/traditional_001.mp3",
    "Traditional Irish slip jig in E-flat major, demonstrating the model's baseline response to familiar musical patterns."
) + visualization_embed(
    "Attention Heatmap",
    "visualizations/traditional_001/attention.html",
    "Attention patterns show regular structure corresponding to the 9/8 meter."
)))

    content.append(section("Experimental Extremes: 'Chromatic Density Study'", """
<p>
[This section will examine the model's response to dense chromatic writing,
where perplexity and attention entropy spike dramatically.]
</p>
"""))

    content.append(section("Deep Dive: Methodology and Interpretation", """
<p>
[This section will provide detailed technical explanations of the metrics,
their formulas, and their interpretations, while remaining accessible to
non-specialist readers.]
</p>
"""))

    content.append(section("Conclusions", """
<p>
[This section will synthesize the findings, discuss implications for understanding
how language models process domain-specific notation, and suggest directions for
future research.]
</p>
"""))

    return "\n".join(content)


def build_essay(
    output_path: Path,
    title: str = "Substrate Melodies: Probing Musical Understanding Through Transformer Confusion",
    author: str = "Brian Edwards",
    email: str = "brian.mabry.edwards@gmail.com",
    phone: str = "512-584-6841",
    location: str = "Waco, Texas, USA",
    model: str = "Claude Code (Claude Opus 4.5)"
) -> None:
    """Build and save the essay HTML."""
    content = build_essay_content()

    # Calculate word count (placeholder)
    word_count = "TBD"
    reading_time = "TBD"

    html = ESSAY_TEMPLATE.format(
        title=title,
        author=author,
        email=email,
        phone=phone,
        location=location,
        model=model,
        date=datetime.now().strftime("%B %d, %Y"),
        word_count=word_count,
        reading_time=reading_time,
        content=content
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f"Built essay: {output_path}")


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
        }}
        h1 {{ color: #2c3e50; }}
        article {{
            margin: 2rem 0;
            padding: 1rem;
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

    args = parser.parse_args()

    # Build the essay
    build_essay(args.output)

    # Build the index
    essays = [{
        "title": "Substrate Melodies: Probing Musical Understanding Through Transformer Confusion",
        "path": "essay.html",
        "date": datetime.now().strftime("%B %d, %Y"),
        "description": "An exploration of how transformer models process ABC notation music, "
                       "using interpretability metrics to map the boundaries of machine musical understanding."
    }]
    build_index(args.index, essays)


if __name__ == "__main__":
    main()
