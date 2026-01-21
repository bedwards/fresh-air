#!/usr/bin/env python3
"""Generate images for music concepts essay using Google Gemini Imagen."""

import os
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


# Image prompts for music concepts essay
MUSIC_CONCEPTS_IMAGES = [
    # Fundamentals
    ("abc_notation.png",
     "Educational diagram showing ABC notation music format: a simple melody written as text letters like 'C D E F G A B c d e' with rhythm markers. Clean typography on cream paper background, vintage sheet music aesthetic. Show the direct correspondence between letters and musical notes."),

    ("staff_vs_abc.png",
     "Side-by-side comparison diagram: traditional five-line musical staff notation on the left showing a simple melody, and the same melody written in ABC text notation on the right. Educational poster style with clean arrows showing correspondence. Blue and gold color scheme."),

    ("scales_major_minor.png",
     "Musical illustration of major and minor scales: two parallel piano keyboards showing the C major scale (all white keys, bright sunny visualization) and C minor scale (with black keys for Eb, Ab, Bb, darker moody visualization). Clear step pattern indicators showing whole and half steps."),

    ("melodic_contour.png",
     "Artistic visualization of melodic contour: a flowing line graph showing the rise and fall of a melody over time, with musical notes positioned along the curve. The line arches up for ascending passages and curves down for descending ones. Watercolor style in blues and purples."),

    # Rhythm and Meter
    ("rhythm_basics.png",
     "Educational diagram of musical rhythm: whole notes, half notes, quarter notes, and eighth notes shown in decreasing size from left to right, each with their duration labeled. Below, a visual representation of how they divide time into equal parts. Clean infographic style."),

    ("time_signatures.png",
     "Comparison of common time signatures: 4/4 shown as four equal beats with emphasis on beat 1, 3/4 as a waltz pattern with three beats, 6/8 as compound meter with two groups of three. Each visualized as colored blocks representing the beat patterns. Music education poster style."),

    ("phrase_structure.png",
     "Diagram of musical phrase structure: an 8-bar phrase shown as two 4-bar sections (antecedent and consequent), with melodic arcs drawn above showing tension building and resolving. Color-coded sections showing call-and-response structure. Architectural diagram aesthetic."),

    # Harmony
    ("harmony_triads.png",
     "Illustration of musical triads: three-note chords stacked in thirds shown on piano keys and as notes on a staff. Major triad (bright, sunny colors), minor triad (cooler, blue tones), diminished (tense, angular), augmented (unstable, wavering). Educational music theory style."),

    ("chord_progressions.png",
     "Visual representation of common chord progressions: the I-IV-V-I progression shown as a circular journey with Roman numerals. Arrows show the movement from tonic to subdominant to dominant and back to tonic. Include sense of tension building and resolution. Clean modern infographic."),

    ("diatonic_chromatic.png",
     "Comparison of diatonic and chromatic music: a piano keyboard with diatonic notes (white keys in C major) glowing softly versus chromatic notes (all 12 keys including black keys) creating a dense, colorful pattern. Show how chromaticism adds color and complexity."),

    # Voice Leading
    ("voice_leading.png",
     "Educational diagram of good voice leading: four-part vocal harmony (soprano, alto, tenor, bass) moving smoothly between chords, with lines showing contrary and oblique motion. Arrows indicate small, stepwise movements. Elegant manuscript paper background."),

    ("parallel_fifths.png",
     "Illustration of why parallel fifths are traditionally avoided: two voices moving in parallel perfect fifths shown with red warning indicators, contrasted with good voice leading showing contrary motion in green. Musical staff with clear annotations. Theory textbook style."),

    ("cadences.png",
     "Visual guide to musical cadences: authentic cadence (V-I) shown as a strong arrival home, half cadence (ending on V) as a question mark, plagal cadence (IV-I) as a gentle 'amen', deceptive cadence (V-vi) as a surprising detour. Each with emotional color coding."),

    # Advanced Concepts
    ("modulation.png",
     "Artistic visualization of musical modulation (key change): a landscape transitioning from one color palette (representing one key) to another through a pivot point. Show the sense of shifting tonal center, like moving from daylight to golden hour. Abstract but musical."),

    ("ornamentation.png",
     "Detailed illustration of musical ornaments: grace notes shown as tiny notes before main notes, trills as rapid alternations, mordents as quick turns, and rolls (in Irish music) as rapid note groupings. Each ornament labeled with its notation symbol. Celtic manuscript style."),

    ("polyrhythm.png",
     "Visualization of polyrhythm: 3 against 2 pattern shown as interlocking gears or woven threads, where three beats in one voice align differently with two beats in another. The pattern creates a cycle. Mathematical precision meets musical flow. Geometric art style."),

    # Musical Form and Tradition
    ("musical_form.png",
     "Diagram of common musical forms: binary form (AB) as two connected sections, ternary form (ABA) as departure and return, rondo (ABACA) as a journey with recurring home. Color-coded sections with connecting paths. Architectural blueprint aesthetic."),

    ("folk_tradition.png",
     "Evocative illustration of folk music tradition: Irish musicians playing fiddle and tin whistle in a warm pub setting, with ABC notation floating ethereally in the background. The music passes from old hands to young. Warm sepia tones with golden highlights. Nostalgic but vibrant."),
]


def main():
    """Generate all images for the music concepts essay."""
    output_dir = Path("docs/images/music_concepts")

    print(f"=== Generating {len(MUSIC_CONCEPTS_IMAGES)} images for music concepts essay ===\n")

    success_count = 0
    for filename, prompt in MUSIC_CONCEPTS_IMAGES:
        output_path = output_dir / filename
        if output_path.exists():
            print(f"Skipping (exists): {output_path}")
            success_count += 1
            continue
        if generate_image(prompt, output_path):
            success_count += 1

    print(f"\n=== Complete: {success_count}/{len(MUSIC_CONCEPTS_IMAGES)} images generated ===")


if __name__ == "__main__":
    main()
