#!/usr/bin/env python3
"""Generate images for essays using Google Gemini Imagen."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from google import genai
from google.genai import types


def generate_image(prompt: str, output_path: Path) -> bool:
    """Generate an image using Gemini Imagen.

    Args:
        prompt: Description of the image to generate
        output_path: Where to save the image

    Returns:
        True if successful, False otherwise
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return False

    try:
        client = genai.Client(api_key=api_key)

        # Use Imagen 4 model
        response = client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9",
            )
        )

        if response.generated_images:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the image
            image = response.generated_images[0]
            image.image.save(str(output_path))
            print(f"Generated: {output_path}")
            return True
        else:
            print(f"No images generated for: {prompt[:50]}...")
            return False

    except Exception as e:
        print(f"Error generating image: {e}")
        return False


# Image prompts for each essay
PERPLEXITY_IMAGES = [
    ("perplexity_concept.png",
     "Scientific illustration of perplexity in language models: a branching decision tree with probability distributions at each node, showing how a model weighs multiple possible next tokens. Clean, minimalist style with blue and white color scheme. Educational diagram aesthetic."),
    ("perplexity_formula.png",
     "Mathematical visualization of perplexity formula: exp(-1/N * sum(log(p))). Show the exponential function transforming log probabilities into an intuitive branching factor. Clean typography, educational poster style."),
    ("perplexity_comparison.png",
     "Comparison chart showing perplexity values: low perplexity (focused beam of light), medium perplexity (moderate spread), high perplexity (diffuse scatter). Abstract visualization of prediction confidence."),
    ("perplexity_music.png",
     "Abstract visualization of a language model processing music notation: streams of musical notes flowing through a neural network, with probability distributions shown as colored halos around predicted notes. Digital art style."),
]

ENTROPY_IMAGES = [
    ("entropy_concept.png",
     "Scientific illustration of attention entropy: a transformer attention head visualized as a spotlight that can be focused (low entropy, narrow beam) or diffuse (high entropy, wide spread). Clean technical diagram."),
    ("entropy_heatmap.png",
     "Stylized attention heatmap visualization: a grid of colored cells showing where a model attends, with bright spots indicating focused attention and uniform color indicating high entropy confusion. Data visualization aesthetic."),
    ("entropy_heads.png",
     "Diagram showing different attention head behaviors: positional heads attending to fixed positions, content heads attending based on meaning, and confused heads with diffuse attention. Technical illustration style."),
    ("entropy_layers.png",
     "Cross-section of a transformer model showing attention entropy changing across layers: early layers with structured patterns, middle layers with complex attention, late layers with focused output attention. Architectural diagram style."),
]

SURPRISAL_IMAGES = [
    ("surprisal_concept.png",
     "Scientific illustration of surprisal in information theory: a scale balancing expected vs unexpected events, with rare events weighted heavily. Clean educational diagram with mathematical notation."),
    ("surprisal_bits.png",
     "Visualization of surprisal measured in bits: probability 0.5 = 1 bit, probability 0.25 = 2 bits, probability 0.125 = 3 bits. Logarithmic scale shown as stacked information units. Infographic style."),
    ("surprisal_timeline.png",
     "Time series visualization of surprisal: a line graph showing surprisal spiking at unexpected tokens in a sequence, with annotations marking high-surprisal events. Data visualization style."),
    ("surprisal_music.png",
     "Abstract visualization of musical surprisal: common chord progressions shown as smooth paths, unexpected harmonies shown as jarring discontinuities. Notes flowing through probability space. Artistic data visualization."),
]


def main():
    """Generate all images for the essays."""
    output_dir = Path("docs/images")

    all_images = [
        ("perplexity", PERPLEXITY_IMAGES),
        ("entropy", ENTROPY_IMAGES),
        ("surprisal", SURPRISAL_IMAGES),
    ]

    for essay_name, images in all_images:
        print(f"\n=== Generating images for {essay_name} essay ===")
        essay_dir = output_dir / essay_name

        for filename, prompt in images:
            output_path = essay_dir / filename
            if output_path.exists():
                print(f"Skipping (exists): {output_path}")
                continue
            generate_image(prompt, output_path)


if __name__ == "__main__":
    main()
