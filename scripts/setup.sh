#!/bin/bash
# Setup script for Music Analysis project
# Run this script to initialize the development environment

set -e

echo "=== Music Analysis Project Setup ==="

# Check Python version
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "Found Python $PYTHON_VERSION"
else
    echo "Error: Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check for virtual environment
echo "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment"
else
    echo "Virtual environment already exists"
fi

# Activate and install dependencies
source .venv/bin/activate
echo "Activated virtual environment"

if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

# Check Ollama
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "Ollama is installed"

    # Check for phi4 model
    if ollama list | grep -q "phi4"; then
        echo "Phi-4 model is available"
    else
        echo "Phi-4 model not found. Installing..."
        ollama pull phi4:14b
    fi
else
    echo "Warning: Ollama not found. Install from https://ollama.ai"
fi

# Create directory structure
echo "Ensuring directory structure..."
mkdir -p data/abc_files data/metrics data/audio
mkdir -p docs/assets docs/visualizations
mkdir -p tests

echo ""
echo "=== Setup Complete ==="
echo "To activate the environment: source .venv/bin/activate"
echo "To start Ollama: ollama serve"
echo ""
