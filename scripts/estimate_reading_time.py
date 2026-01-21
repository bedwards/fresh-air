#!/usr/bin/env python3
"""Estimate reading time for an HTML file."""

import sys
import re
from pathlib import Path


def count_words(html_content: str) -> int:
    """Count words in HTML content, excluding tags and code."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<pre[^>]*>.*?</pre>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<code[^>]*>.*?</code>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.strip().split()
    return len(words)


def estimate_reading_time(word_count: int, wpm: int = 200) -> str:
    """Estimate reading time based on word count.

    Args:
        word_count: Number of words
        wpm: Words per minute (default 200 for technical content)

    Returns:
        Human-readable reading time string
    """
    minutes = word_count / wpm

    if minutes < 1:
        return "less than 1 minute"
    elif minutes < 2:
        return "about 1 minute"
    else:
        rounded_minutes = round(minutes)
        return f"about {rounded_minutes} minutes"


def main():
    if len(sys.argv) < 2:
        print("Usage: estimate_reading_time.py <html_file>", file=sys.stderr)
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    content = filepath.read_text(encoding='utf-8')
    word_count = count_words(content)
    reading_time = estimate_reading_time(word_count)
    print(reading_time)


if __name__ == "__main__":
    main()
