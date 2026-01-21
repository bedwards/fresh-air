#!/bin/bash
# Deploy to GitHub Pages
# Run from project root

set -e

echo "=== Deploying to GitHub Pages ==="

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check if docs/essay.html exists
if [ ! -f "docs/essay.html" ]; then
    echo "Warning: docs/essay.html not found. Skipping metadata injection."
else
    # Update word count and reading time
    echo "Calculating word count..."
    WORD_COUNT=$(python scripts/count_words.py docs/essay.html)
    echo "Word count: $WORD_COUNT"

    echo "Calculating reading time..."
    READING_TIME=$(python scripts/estimate_reading_time.py docs/essay.html)
    echo "Reading time: $READING_TIME"

    # Save metadata
    echo "$WORD_COUNT" > docs/assets/wordcount.txt
    echo "$READING_TIME" > docs/assets/reading_time.txt
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain docs/)" ]; then
    echo "Committing docs changes..."
    git add docs/
    git commit -m "Update docs for deployment

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
fi

# Deploy to gh-pages branch
echo "Deploying to gh-pages..."
git subtree push --prefix docs origin gh-pages

echo ""
echo "=== Deployment Complete ==="
echo "Site will be available at: https://bedwards.github.io/fresh-air/"
echo ""
