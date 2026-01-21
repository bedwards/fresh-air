#!/usr/bin/env python3
"""ABC notation music generator for transformer analysis.

This module generates ABC notation files across various musical genres
for analysis with transformer interpretability techniques.

Genres:
    - traditional: Folk tunes, hymns, classical themes
    - avant-garde: Unusual time signatures, extended harmony
    - experimental: Extreme registers, dense polyphony
    - noise: Random sequences, no tonal center
    - terrible: Intentionally bad voice leading
    - silence: Sparse notes, extended rests

Usage:
    python src/abc_generator.py --genre traditional --count 5
    python src/abc_generator.py --all
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

Genre = Literal["traditional", "avant-garde", "experimental", "noise", "terrible", "silence"]

# ABC notation building blocks
NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
OCTAVE_MARKS = [",,", ",", "", "'", "''"]  # low to high
ACCIDENTALS = ['', '^', '_', '=']  # natural, sharp, flat, explicit natural
DURATIONS = ['', '2', '3', '4', '6', '8', '/2', '/4']
RESTS = ['z', 'z2', 'z4', 'z8']


@dataclass
class ABCMetadata:
    """Metadata for an ABC notation file."""
    filename: str
    genre: Genre
    subgenre: str
    description: str
    voices: int
    measures: int
    time_signature: str
    key: str
    tempo: int
    generation_params: dict


def generate_traditional(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a traditional-style ABC tune.

    Traditional pieces feature:
    - Diatonic melodies
    - Regular phrase structures
    - Common time signatures (4/4, 3/4, 6/8)
    - Conventional harmonic progressions
    """
    if seed is not None:
        random.seed(seed)

    # TODO: Implement traditional tune generation
    # This is a placeholder for the worker to implement

    abc_content = f"""X:{index}
T:Traditional Tune {index:03d}
M:4/4
L:1/8
K:C
|: CDEF GABc | cBAG FEDC :|
"""

    metadata = ABCMetadata(
        filename=f"traditional_{index:03d}.abc",
        genre="traditional",
        subgenre="placeholder",
        description="Placeholder traditional tune",
        voices=1,
        measures=2,
        time_signature="4/4",
        key="C",
        tempo=120,
        generation_params={"seed": seed, "algorithm": "placeholder"}
    )

    return abc_content, metadata


def generate_avantgarde(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate an avant-garde ABC tune.

    Avant-garde pieces feature:
    - Unusual time signatures (5/8, 7/8, mixed meter)
    - Extended harmony and chromaticism
    - Wide melodic leaps
    - Complex rhythmic patterns
    """
    if seed is not None:
        random.seed(seed)

    # TODO: Implement avant-garde tune generation

    abc_content = f"""X:{index}
T:Avant-garde Study {index:03d}
M:7/8
L:1/16
K:Cmix
|: ^C2_D2E2^F G2=A2B2c2 | _d4^e4f4 :|
"""

    metadata = ABCMetadata(
        filename=f"avantgarde_{index:03d}.abc",
        genre="avant-garde",
        subgenre="placeholder",
        description="Placeholder avant-garde piece",
        voices=1,
        measures=2,
        time_signature="7/8",
        key="Cmix",
        tempo=100,
        generation_params={"seed": seed, "algorithm": "placeholder"}
    )

    return abc_content, metadata


def generate_experimental(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate an experimental ABC piece.

    Experimental pieces feature:
    - Extreme register changes
    - Dense polyphonic textures (4 voices)
    - Simultaneous contrasting materials
    - Unpredictable structures
    """
    if seed is not None:
        random.seed(seed)

    # TODO: Implement experimental piece generation

    abc_content = f"""X:{index}
T:Experimental {index:03d}
M:4/4
L:1/16
K:none
V:1
C,,,4 c''''4 | C,,,4 c''''4 |
V:2
E,,,4 e''''4 | E,,,4 e''''4 |
"""

    metadata = ABCMetadata(
        filename=f"experimental_{index:03d}.abc",
        genre="experimental",
        subgenre="placeholder",
        description="Placeholder experimental piece",
        voices=2,
        measures=2,
        time_signature="4/4",
        key="none",
        tempo=60,
        generation_params={"seed": seed, "algorithm": "placeholder"}
    )

    return abc_content, metadata


def generate_noise(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a noise/random ABC piece.

    Noise pieces feature:
    - Random pitch sequences
    - No tonal center
    - Erratic rhythms
    - No structural patterns
    """
    if seed is not None:
        random.seed(seed)

    # Generate random notes
    notes = []
    for _ in range(32):
        note = random.choice(NOTES)
        octave = random.choice(OCTAVE_MARKS)
        accidental = random.choice(ACCIDENTALS)
        duration = random.choice(DURATIONS)
        notes.append(f"{accidental}{note}{octave}{duration}")

    notes_str = ' '.join(notes[:16]) + ' | ' + ' '.join(notes[16:]) + ' |'

    abc_content = f"""X:{index}
T:Noise Sequence {index:03d}
M:4/4
L:1/8
K:none
{notes_str}
"""

    metadata = ABCMetadata(
        filename=f"noise_{index:03d}.abc",
        genre="noise",
        subgenre="random",
        description="Random noise sequence with no tonal center",
        voices=1,
        measures=2,
        time_signature="4/4",
        key="none",
        tempo=120,
        generation_params={"seed": seed, "algorithm": "random"}
    )

    return abc_content, metadata


def generate_terrible(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate an intentionally bad ABC piece.

    Terrible pieces feature:
    - Parallel fifths and octaves
    - Voice crossing
    - Awkward ranges
    - Bad voice leading
    - Unresolved dissonances
    """
    if seed is not None:
        random.seed(seed)

    # TODO: Implement terrible piece generation

    abc_content = f"""X:{index}
T:Terrible Harmony {index:03d}
M:4/4
L:1/4
K:C
V:1
C G C G | C G C G |
V:2
G d G d | G d G d |
% Parallel fifths throughout - intentionally bad
"""

    metadata = ABCMetadata(
        filename=f"terrible_{index:03d}.abc",
        genre="terrible",
        subgenre="parallel_fifths",
        description="Intentionally bad harmony with parallel fifths",
        voices=2,
        measures=2,
        time_signature="4/4",
        key="C",
        tempo=100,
        generation_params={"seed": seed, "algorithm": "bad_voice_leading"}
    )

    return abc_content, metadata


def generate_silence(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a sparse/silent ABC piece.

    Silence pieces feature:
    - Extended rests
    - Sparse note material
    - Minimal melodic content
    - Long pauses
    """
    if seed is not None:
        random.seed(seed)

    # TODO: Implement silence piece generation

    abc_content = f"""X:{index}
T:Silence Study {index:03d}
M:4/4
L:1/4
K:C
z4 | z4 | C z3 | z4 | z4 | z2 E z | z4 | z4 |
"""

    metadata = ABCMetadata(
        filename=f"silence_{index:03d}.abc",
        genre="silence",
        subgenre="sparse",
        description="Sparse piece with extended silences",
        voices=1,
        measures=8,
        time_signature="4/4",
        key="C",
        tempo=60,
        generation_params={"seed": seed, "algorithm": "sparse"}
    )

    return abc_content, metadata


GENERATORS = {
    "traditional": generate_traditional,
    "avant-garde": generate_avantgarde,
    "experimental": generate_experimental,
    "noise": generate_noise,
    "terrible": generate_terrible,
    "silence": generate_silence,
}


def save_abc(abc_content: str, metadata: ABCMetadata, output_dir: Path) -> None:
    """Save ABC content and metadata to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ABC file
    abc_path = output_dir / metadata.filename
    abc_path.write_text(abc_content, encoding='utf-8')

    # Save metadata JSON
    json_path = output_dir / f"{metadata.filename.replace('.abc', '.json')}"
    json_path.write_text(json.dumps(asdict(metadata), indent=2), encoding='utf-8')

    print(f"Generated: {abc_path}")


def generate_corpus(
    genre: Genre | None = None,
    count: int = 5,
    output_dir: Path = Path("data/abc_files"),
    seed: int | None = None
) -> list[ABCMetadata]:
    """Generate a corpus of ABC files.

    Args:
        genre: Specific genre to generate, or None for all genres
        count: Number of files per genre
        output_dir: Directory to save files
        seed: Random seed for reproducibility

    Returns:
        List of metadata for generated files
    """
    all_metadata = []

    if genre:
        genres = [genre]
    else:
        genres = list(GENERATORS.keys())

    for g in genres:
        generator = GENERATORS[g]
        for i in range(1, count + 1):
            file_seed = seed + i if seed else None
            abc_content, metadata = generator(i, file_seed)
            save_abc(abc_content, metadata, output_dir)
            all_metadata.append(metadata)

    return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Generate ABC notation files")
    parser.add_argument("--genre", choices=list(GENERATORS.keys()), help="Genre to generate")
    parser.add_argument("--count", type=int, default=5, help="Number of files per genre")
    parser.add_argument("--output", type=Path, default=Path("data/abc_files"), help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--all", action="store_true", help="Generate all genres")

    args = parser.parse_args()

    if args.all:
        generate_corpus(genre=None, count=args.count, output_dir=args.output, seed=args.seed)
    elif args.genre:
        generate_corpus(genre=args.genre, count=args.count, output_dir=args.output, seed=args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
