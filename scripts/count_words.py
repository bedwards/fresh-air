#!/usr/bin/env python3
"""Count words in an HTML file, excluding tags and code blocks."""

import sys
import re
from pathlib import Path


def count_words(html_content: str) -> int:
    """Count words in HTML content, excluding tags and code."""
    # Remove script and style blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove code blocks (pre, code tags)
    text = re.sub(r'<pre[^>]*>.*?</pre>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<code[^>]*>.*?</code>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&[a-z]+;', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Count words
    words = text.strip().split()
    return len(words)


def main():
    if len(sys.argv) < 2:
        print("Usage: count_words.py <html_file>", file=sys.stderr)
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    content = filepath.read_text(encoding='utf-8')
    word_count = count_words(content)
    print(word_count)


if __name__ == "__main__":
    main()
