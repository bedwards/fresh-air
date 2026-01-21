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
CHROMATIC_NOTES = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']
OCTAVE_MARKS = [",,", ",", "", "'", "''"]  # low to high
ACCIDENTALS = ['', '^', '_', '=']  # natural, sharp, flat, explicit natural
DURATIONS = ['', '2', '3', '4', '6', '8', '/2', '/4']
RESTS = ['z', 'z2', 'z4', 'z8']

# Scales and modes for traditional music
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]  # Semitone intervals from root
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
DORIAN_MODE = [0, 2, 3, 5, 7, 9, 10]
MIXOLYDIAN_MODE = [0, 2, 4, 5, 7, 9, 10]

# Irish traditional music patterns
JIG_RHYTHMS = ['3', '3', '', '3', '3', '']  # 6/8 compound duple
REEL_RHYTHMS = ['', '', '', '', '', '', '', '']  # 4/4 eighth notes
SLIP_JIG_RHYTHMS = ['3', '3', '3', '', '', '']  # 9/8 compound triple

# Common traditional cadential patterns (scale degrees, 1-indexed)
TRADITIONAL_CADENCES = [
    [5, 1],  # V-I perfect cadence
    [4, 1],  # IV-I plagal cadence
    [2, 1],  # ii-I
    [5, 3, 1],  # V-iii-I
]

# Keys for traditional music
TRADITIONAL_KEYS = ['G', 'D', 'A', 'C', 'F', 'Gm', 'Dm', 'Am', 'Em']


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


def scale_degree_to_note(degree: int, key: str = 'C', octave: int = 0) -> str:
    """Convert a scale degree (1-7) to an ABC note in the given key.

    Args:
        degree: Scale degree (1-7)
        key: Key signature (e.g., 'G', 'Dm')
        octave: Octave modifier (-2 to 2)

    Returns:
        ABC note string
    """
    key_offsets = {'C': 0, 'G': 7, 'D': 2, 'A': 9, 'E': 4, 'B': 11, 'F': 5}

    # Handle minor keys
    is_minor = key.endswith('m')
    base_key = key[:-1] if is_minor else key

    scale = MINOR_SCALE if is_minor else MAJOR_SCALE

    # Get the semitone offset for this scale degree
    semitone = scale[(degree - 1) % 7]

    # Convert to absolute note
    key_offset = key_offsets.get(base_key, 0)
    absolute_semitone = (key_offset + semitone) % 12

    # Map to note name
    note_names = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
    note_sharps = ['', '^', '', '^', '', '', '^', '', '^', '', '^', '']

    note = note_names[absolute_semitone]
    accidental = note_sharps[absolute_semitone]

    # Handle octave
    if octave < 0:
        note = note.upper() + ',' * (-octave)
    elif octave > 0:
        note = note.lower() + "'" * (octave - 1)
    else:
        note = note.upper()

    return accidental + note


def generate_irish_jig(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate an Irish jig in 6/8 time."""
    if seed is not None:
        random.seed(seed)

    key = random.choice(['G', 'D', 'A', 'Em', 'Am'])
    tempo = random.randint(100, 132)

    # Jig patterns - groups of 3 eighth notes
    def generate_jig_phrase(length: int = 8) -> str:
        """Generate a phrase of jig patterns."""
        notes = []
        scale_degrees = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 = octave

        for bar in range(length):
            # Two groups of 3 eighth notes per bar
            for group in range(2):
                # Stepwise motion with occasional leaps
                if random.random() < 0.7:
                    # Stepwise pattern
                    start = random.choice(scale_degrees[:5])
                    direction = random.choice([1, -1])
                    pattern = [start, start + direction, start + 2*direction]
                else:
                    # Leap and return
                    base = random.choice(scale_degrees[:5])
                    leap = random.choice([3, 4, 5])
                    pattern = [base, base + leap, base]

                for deg in pattern:
                    deg = max(1, min(8, deg))
                    octave = 0 if deg <= 7 else 1
                    actual_deg = deg if deg <= 7 else deg - 7
                    note = scale_degree_to_note(actual_deg, key.replace('m', ''), octave)
                    notes.append(note)

            notes.append('|')

        return ' '.join(notes)

    # Generate A and B parts (8 bars each)
    a_part = generate_jig_phrase(8)
    b_part = generate_jig_phrase(8)

    abc_content = f"""X:{index}
T:Irish Jig {index:03d}
C:Generated
M:6/8
L:1/8
Q:1/4={tempo}
K:{key}
|: {a_part} :|
|: {b_part} :|
"""

    metadata = ABCMetadata(
        filename=f"traditional_{index:03d}.abc",
        genre="traditional",
        subgenre="irish_jig",
        description=f"Irish jig in {key} with typical compound duple meter patterns",
        voices=1,
        measures=16,
        time_signature="6/8",
        key=key,
        tempo=tempo,
        generation_params={"seed": seed, "algorithm": "irish_jig", "structure": "AABB"}
    )

    return abc_content, metadata


def generate_slip_jig(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a slip jig in 9/8 time."""
    if seed is not None:
        random.seed(seed)

    key = random.choice(['G', 'D', 'Em', 'Am'])
    tempo = random.randint(90, 116)

    def generate_slip_phrase(length: int = 8) -> str:
        """Generate a phrase in 9/8."""
        notes = []

        for bar in range(length):
            # Three groups of 3 eighth notes per bar
            for group in range(3):
                start = random.randint(1, 5)
                direction = random.choice([1, -1, 0])

                for i in range(3):
                    deg = max(1, min(7, start + i * direction))
                    note = scale_degree_to_note(deg, key.replace('m', ''), 0)
                    notes.append(note)

            notes.append('|')

        return ' '.join(notes)

    a_part = generate_slip_phrase(8)
    b_part = generate_slip_phrase(8)

    abc_content = f"""X:{index}
T:Slip Jig {index:03d}
C:Generated
M:9/8
L:1/8
Q:3/8={tempo}
K:{key}
|: {a_part} :|
|: {b_part} :|
"""

    metadata = ABCMetadata(
        filename=f"traditional_{index:03d}.abc",
        genre="traditional",
        subgenre="slip_jig",
        description=f"Slip jig in {key} with compound triple meter (9/8)",
        voices=1,
        measures=16,
        time_signature="9/8",
        key=key,
        tempo=tempo,
        generation_params={"seed": seed, "algorithm": "slip_jig", "structure": "AABB"}
    )

    return abc_content, metadata


def generate_reel(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a reel in 4/4 time."""
    if seed is not None:
        random.seed(seed)

    key = random.choice(['G', 'D', 'A', 'Em', 'Bm'])
    tempo = random.randint(100, 132)

    def generate_reel_phrase(length: int = 8) -> str:
        """Generate a reel phrase with eighth note runs."""
        notes = []

        for bar in range(length):
            # 8 eighth notes per bar
            start = random.randint(1, 5)
            direction = random.choice([1, -1])

            for i in range(8):
                # Mix of stepwise and small leaps
                if i % 4 == 0:
                    start = random.randint(1, 6)
                    direction = random.choice([1, -1])

                deg = max(1, min(8, start + (i % 4) * direction))
                octave = 0 if deg <= 7 else 1
                actual_deg = deg if deg <= 7 else deg - 7
                note = scale_degree_to_note(actual_deg, key.replace('m', ''), octave)
                notes.append(note)

            notes.append('|')

        return ' '.join(notes)

    a_part = generate_reel_phrase(8)
    b_part = generate_reel_phrase(8)

    abc_content = f"""X:{index}
T:Reel {index:03d}
C:Generated
M:4/4
L:1/8
Q:1/4={tempo}
K:{key}
|: {a_part} :|
|: {b_part} :|
"""

    metadata = ABCMetadata(
        filename=f"traditional_{index:03d}.abc",
        genre="traditional",
        subgenre="reel",
        description=f"Reel in {key} with continuous eighth note patterns",
        voices=1,
        measures=16,
        time_signature="4/4",
        key=key,
        tempo=tempo,
        generation_params={"seed": seed, "algorithm": "reel", "structure": "AABB"}
    )

    return abc_content, metadata


def generate_hymn(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a 4-part hymn in chorale style."""
    if seed is not None:
        random.seed(seed)

    key = random.choice(['C', 'G', 'F', 'D'])
    tempo = random.randint(60, 80)

    # Generate simple chord progression
    progressions = [
        [1, 4, 5, 1],  # I-IV-V-I
        [1, 6, 4, 5],  # I-vi-IV-V
        [1, 5, 6, 4],  # I-V-vi-IV
        [1, 4, 1, 5],  # I-IV-I-V
    ]

    def chord_to_notes(root_degree: int, key: str) -> list[str]:
        """Convert a chord root to 4-part voicing."""
        # Simple triadic voicing with doubling
        bass = scale_degree_to_note(root_degree, key, -1)
        tenor = scale_degree_to_note(((root_degree + 4) % 7) or 7, key, 0)
        alto = scale_degree_to_note(((root_degree + 2) % 7) or 7, key, 0)
        soprano = scale_degree_to_note(root_degree, key, 1)
        return [soprano, alto, tenor, bass]

    measures = 8
    progression = random.choice(progressions)

    # Build each voice line
    soprano_line = []
    alto_line = []
    tenor_line = []
    bass_line = []

    for m in range(measures):
        chord_root = progression[m % len(progression)]
        voices = chord_to_notes(chord_root, key)

        # Whole notes for hymn style
        soprano_line.append(voices[0] + '4')
        alto_line.append(voices[1] + '4')
        tenor_line.append(voices[2] + '4')
        bass_line.append(voices[3] + '4')

        soprano_line.append('|')
        alto_line.append('|')
        tenor_line.append('|')
        bass_line.append('|')

    abc_content = f"""X:{index}
T:Hymn {index:03d}
C:Generated
M:4/4
L:1/4
Q:1/4={tempo}
K:{key}
V:S clef=treble name="Soprano"
{' '.join(soprano_line)}
V:A clef=treble name="Alto"
{' '.join(alto_line)}
V:T clef=treble-8 name="Tenor"
{' '.join(tenor_line)}
V:B clef=bass name="Bass"
{' '.join(bass_line)}
"""

    metadata = ABCMetadata(
        filename=f"traditional_{index:03d}.abc",
        genre="traditional",
        subgenre="hymn",
        description=f"Four-part hymn in {key} with conventional chord progressions",
        voices=4,
        measures=measures,
        time_signature="4/4",
        key=key,
        tempo=tempo,
        generation_params={"seed": seed, "algorithm": "hymn_chorale", "progression": progression}
    )

    return abc_content, metadata


def generate_traditional(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a traditional-style ABC tune.

    Traditional pieces feature:
    - Diatonic melodies
    - Regular phrase structures
    - Common time signatures (4/4, 3/4, 6/8)
    - Conventional harmonic progressions

    Cycles through different traditional forms based on index.
    """
    if seed is not None:
        random.seed(seed)

    # Cycle through traditional types
    traditional_types = [
        generate_irish_jig,
        generate_slip_jig,
        generate_reel,
        generate_hymn,
        generate_irish_jig,  # More jigs (common)
    ]

    generator = traditional_types[(index - 1) % len(traditional_types)]
    abc_content, metadata = generator(index, seed)

    # Update filename to match expected pattern
    metadata.filename = f"traditional_{index:03d}.abc"

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

    # Unusual time signatures
    time_sigs = ['5/8', '7/8', '11/8', '5/4', '7/4']
    time_sig = random.choice(time_sigs)
    beats = int(time_sig.split('/')[0])

    # Extended modes
    modes = ['Cmix', 'Ddor', 'Elyd', 'Floc', 'Gphr']
    mode = random.choice(modes)

    tempo = random.randint(60, 100)
    measures = random.randint(12, 20)

    # Generate with wide leaps and chromaticism
    def generate_avant_phrase(length: int) -> str:
        notes = []
        current_pitch = random.randint(0, 11)  # Chromatic pitch class

        for bar in range(length):
            bar_notes = []
            remaining_beats = beats

            while remaining_beats > 0:
                # Random duration
                dur_choices = [1, 2, 3] if remaining_beats >= 3 else [1, 2][:remaining_beats]
                dur = random.choice(dur_choices)
                remaining_beats -= dur

                # Wide leap (tritone, 7th, 9th) or chromatic step
                leap = random.choice([-11, -7, -6, -5, -1, 1, 5, 6, 7, 11])
                current_pitch = (current_pitch + leap) % 12

                # Determine octave (vary widely)
                octave = random.choice([-1, 0, 0, 1, 1])

                # Build note
                note_name = CHROMATIC_NOTES[current_pitch]
                if octave < 0:
                    if note_name.startswith('^'):
                        note_str = note_name[0] + note_name[1].upper() + ',' * (-octave)
                    else:
                        note_str = note_name.upper() + ',' * (-octave)
                elif octave > 0:
                    if note_name.startswith('^'):
                        note_str = note_name[0] + note_name[1].lower() + "'" * (octave - 1)
                    else:
                        note_str = note_name.lower() + "'" * (octave - 1)
                else:
                    note_str = note_name

                # Add duration
                if dur > 1:
                    note_str += str(dur)

                bar_notes.append(note_str)

            notes.append(' '.join(bar_notes))
            notes.append('|')

        return ' '.join(notes)

    content = generate_avant_phrase(measures)

    abc_content = f"""X:{index}
T:Avant-garde Study {index:03d}
C:Generated
M:{time_sig}
L:1/8
Q:1/8={tempo}
K:{mode}
{content}
"""

    metadata = ABCMetadata(
        filename=f"avantgarde_{index:03d}.abc",
        genre="avant-garde",
        subgenre="extended_meter",
        description=f"Avant-garde piece in {time_sig} with wide melodic leaps and chromatic harmony",
        voices=1,
        measures=measures,
        time_signature=time_sig,
        key=mode,
        tempo=tempo,
        generation_params={"seed": seed, "algorithm": "avant_garde_leaps", "chromatic": True}
    )

    return abc_content, metadata


def generate_experimental(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate an experimental ABC piece with 4 independent voices.

    Experimental pieces feature:
    - Extreme register changes
    - Dense polyphonic textures (4 voices)
    - Independent chromatic voice leading
    - Unpredictable structures
    """
    if seed is not None:
        random.seed(seed)

    tempo = random.randint(40, 72)
    measures = random.randint(8, 16)

    def generate_chromatic_voice(name: str, base_octave: int, measures: int) -> str:
        """Generate a single chromatic voice line."""
        notes = []
        current_pitch = random.randint(0, 11)

        for bar in range(measures):
            bar_notes = []
            remaining_beats = 16  # 16 sixteenth notes per bar

            while remaining_beats > 0:
                # Varied durations
                dur = random.choice([1, 2, 3, 4, 6, 8])
                dur = min(dur, remaining_beats)
                remaining_beats -= dur

                # Sometimes rest
                if random.random() < 0.15:
                    if dur == 1:
                        bar_notes.append('z')
                    else:
                        bar_notes.append(f'z{dur}')
                    continue

                # Chromatic movement with occasional octave leaps
                if random.random() < 0.2:
                    # Octave leap
                    leap = random.choice([-12, 12])
                else:
                    # Chromatic or semitone movement
                    leap = random.choice([-3, -2, -1, 1, 2, 3])

                current_pitch = (current_pitch + leap) % 12

                # Vary octave dramatically
                octave = base_octave + random.choice([-2, -1, 0, 0, 1, 2])

                note_name = CHROMATIC_NOTES[current_pitch]

                # Build octave notation
                if octave <= -2:
                    if note_name.startswith('^'):
                        note_str = note_name[0] + note_name[1].upper() + ',' * (-octave)
                    else:
                        note_str = note_name.upper() + ',' * (-octave)
                elif octave >= 1:
                    if note_name.startswith('^'):
                        note_str = note_name[0] + note_name[1].lower() + "'" * octave
                    else:
                        note_str = note_name.lower() + "'" * max(0, octave - 1)
                else:
                    note_str = note_name

                if dur > 1:
                    note_str += str(dur)

                bar_notes.append(note_str)

            notes.append(' '.join(bar_notes))
            notes.append('|')

        return f"V:{name}\n" + ' '.join(notes)

    voice1 = generate_chromatic_voice('1', 1, measures)
    voice2 = generate_chromatic_voice('2', 0, measures)
    voice3 = generate_chromatic_voice('3', 0, measures)
    voice4 = generate_chromatic_voice('4', -1, measures)

    abc_content = f"""X:{index}
T:Experimental {index:03d}
C:Generated
M:4/4
L:1/16
Q:1/4={tempo}
K:none
V:1 clef=treble name="Voice 1"
V:2 clef=treble name="Voice 2"
V:3 clef=bass name="Voice 3"
V:4 clef=bass name="Voice 4"
{voice1}
{voice2}
{voice3}
{voice4}
"""

    metadata = ABCMetadata(
        filename=f"experimental_{index:03d}.abc",
        genre="experimental",
        subgenre="chromatic_polyphony",
        description="Dense 4-voice chromatic piece with extreme register changes and independent lines",
        voices=4,
        measures=measures,
        time_signature="4/4",
        key="none",
        tempo=tempo,
        generation_params={"seed": seed, "algorithm": "chromatic_4voice", "independent_voices": True}
    )

    return abc_content, metadata


def generate_noise(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a noise/random ABC piece.

    Noise pieces feature:
    - Truly random pitch sequences
    - No tonal center
    - Erratic rhythms with no pattern
    - No structural coherence
    """
    if seed is not None:
        random.seed(seed)

    # Random time signature
    time_sigs = ['4/4', '5/4', '7/8', '3/4', '6/8']
    time_sig = random.choice(time_sigs)

    measures = random.randint(8, 16)

    # Generate completely random notes
    def random_note() -> str:
        """Generate a completely random note."""
        # Random note name
        note = random.choice(NOTES)

        # Random accidental
        accidental = random.choice(['', '^', '_', '^^', '__', '='])

        # Random octave (extreme range)
        octave_mark = random.choice([',,', ',', '', '', "'", "''"])

        # Random duration
        duration = random.choice(['', '2', '3', '4', '/2', '/4', '6', '8', '/3'])

        # Determine case based on octave
        if octave_mark in ["'", "''"]:
            note = note.lower()
            octave_str = octave_mark
        elif octave_mark in [',', ',,']:
            note = note.upper()
            octave_str = octave_mark
        else:
            note = random.choice([note.upper(), note.lower()])
            octave_str = ''

        return f"{accidental}{note}{octave_str}{duration}"

    # Generate chaotic content
    notes = []
    for bar in range(measures):
        bar_notes = []
        # Random number of events per bar
        num_events = random.randint(3, 12)

        for _ in range(num_events):
            if random.random() < 0.1:
                # Occasional rest
                bar_notes.append(random.choice(['z', 'z2', 'z4', 'z/2']))
            else:
                bar_notes.append(random_note())

        notes.append(' '.join(bar_notes))
        notes.append('|')

    content = ' '.join(notes)

    abc_content = f"""X:{index}
T:Noise Sequence {index:03d}
C:Generated (Random)
M:{time_sig}
L:1/8
K:none
% Deliberately random - no tonal center or coherent rhythm
{content}
"""

    metadata = ABCMetadata(
        filename=f"noise_{index:03d}.abc",
        genre="noise",
        subgenre="pure_random",
        description="Completely random pitch and rhythm sequence with no musical coherence",
        voices=1,
        measures=measures,
        time_signature=time_sig,
        key="none",
        tempo=120,
        generation_params={"seed": seed, "algorithm": "pure_random", "tonal_center": None}
    )

    return abc_content, metadata


def generate_terrible(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate an intentionally bad ABC piece.

    Terrible pieces feature:
    - Parallel fifths and octaves (forbidden in voice leading)
    - Voice crossing
    - Awkward ranges (too high/low)
    - Unresolved dissonances
    - Bad voice leading
    """
    if seed is not None:
        random.seed(seed)

    key = random.choice(['C', 'G', 'F'])
    tempo = random.randint(80, 120)
    measures = random.randint(8, 12)

    # Deliberately bad voice leading patterns
    bad_patterns = [
        'parallel_fifths',
        'parallel_octaves',
        'voice_crossing',
        'awkward_range',
        'unresolved_dissonance'
    ]
    pattern = random.choice(bad_patterns)

    if pattern == 'parallel_fifths':
        # Generate parallel fifths between two voices
        soprano = []
        bass = []

        for m in range(measures):
            # Move in parallel fifths (bad!)
            bass_note = random.randint(1, 5)
            soprano_note = bass_note + 4  # Perfect 5th above

            s_note = scale_degree_to_note(soprano_note if soprano_note <= 7 else soprano_note - 7, key, 1)
            b_note = scale_degree_to_note(bass_note, key, -1)

            soprano.append(f"{s_note}4 |")
            bass.append(f"{b_note}4 |")

        abc_content = f"""X:{index}
T:Terrible Harmony {index:03d} - Parallel Fifths
C:Generated (Intentionally Bad)
M:4/4
L:1/4
Q:1/4={tempo}
K:{key}
V:S clef=treble
{' '.join(soprano)}
V:B clef=bass
{' '.join(bass)}
% WARNING: Contains deliberate parallel fifths throughout
"""
        subgenre = "parallel_fifths"

    elif pattern == 'parallel_octaves':
        # Parallel octaves
        soprano = []
        bass = []

        for m in range(measures):
            note_deg = random.randint(1, 7)
            s_note = scale_degree_to_note(note_deg, key, 1)
            b_note = scale_degree_to_note(note_deg, key, -1)

            soprano.append(f"{s_note}4 |")
            bass.append(f"{b_note}4 |")

        abc_content = f"""X:{index}
T:Terrible Harmony {index:03d} - Parallel Octaves
C:Generated (Intentionally Bad)
M:4/4
L:1/4
Q:1/4={tempo}
K:{key}
V:S clef=treble
{' '.join(soprano)}
V:B clef=bass
{' '.join(bass)}
% WARNING: Contains deliberate parallel octaves - voices double at octave
"""
        subgenre = "parallel_octaves"

    elif pattern == 'voice_crossing':
        # Voices that cross each other
        high_voice = []
        low_voice = []

        for m in range(measures):
            if m % 2 == 0:
                # Normal position
                high_voice.append(scale_degree_to_note(5, key, 1) + "4 |")
                low_voice.append(scale_degree_to_note(1, key, 0) + "4 |")
            else:
                # Crossed position - low voice goes higher than high voice
                high_voice.append(scale_degree_to_note(1, key, -1) + "4 |")
                low_voice.append(scale_degree_to_note(5, key, 1) + "4 |")

        abc_content = f"""X:{index}
T:Terrible Harmony {index:03d} - Voice Crossing
C:Generated (Intentionally Bad)
M:4/4
L:1/4
Q:1/4={tempo}
K:{key}
V:1 clef=treble name="High Voice"
{' '.join(high_voice)}
V:2 clef=bass name="Low Voice"
{' '.join(low_voice)}
% WARNING: Voices cross each other repeatedly
"""
        subgenre = "voice_crossing"

    elif pattern == 'awkward_range':
        # Notes in extreme, awkward ranges
        melody = []

        for m in range(measures):
            if random.random() < 0.5:
                # Extremely high
                note = random.choice(['c', 'd', 'e', 'f', 'g'])
                melody.append(f"{note}'''4 {note}'''4 |")
            else:
                # Extremely low
                note = random.choice(['C', 'D', 'E', 'F', 'G'])
                melody.append(f"{note},,,4 {note},,,4 |")

        abc_content = f"""X:{index}
T:Terrible Harmony {index:03d} - Awkward Range
C:Generated (Intentionally Bad)
M:4/4
L:1/4
Q:1/4={tempo}
K:{key}
{' '.join(melody)}
% WARNING: Notes in extremely awkward, unplayable ranges
"""
        subgenre = "awkward_range"

    else:  # unresolved_dissonance
        # Major 7ths and minor 2nds left hanging
        voice1 = []
        voice2 = []

        for m in range(measures):
            # Create dissonance that never resolves
            v1_deg = random.randint(1, 7)
            v2_deg = v1_deg + 7  # Minor 2nd / Major 7th
            if v2_deg > 7:
                v2_deg -= 7

            v1_note = scale_degree_to_note(v1_deg, key, 0)
            v2_note = scale_degree_to_note(v2_deg, key, 1)

            voice1.append(f"{v1_note}4 |")
            voice2.append(f"{v2_note}4 |")

        abc_content = f"""X:{index}
T:Terrible Harmony {index:03d} - Unresolved Dissonance
C:Generated (Intentionally Bad)
M:4/4
L:1/4
Q:1/4={tempo}
K:{key}
V:1 clef=treble
{' '.join(voice1)}
V:2 clef=treble
{' '.join(voice2)}
% WARNING: Contains unresolved major 7ths and minor 2nds
"""
        subgenre = "unresolved_dissonance"

    metadata = ABCMetadata(
        filename=f"terrible_{index:03d}.abc",
        genre="terrible",
        subgenre=subgenre,
        description=f"Intentionally bad music with {subgenre.replace('_', ' ')}",
        voices=2,
        measures=measures,
        time_signature="4/4",
        key=key,
        tempo=tempo,
        generation_params={"seed": seed, "algorithm": "bad_voice_leading", "error_type": subgenre}
    )

    return abc_content, metadata


def generate_silence(index: int, seed: int | None = None) -> tuple[str, ABCMetadata]:
    """Generate a sparse/silent ABC piece.

    Silence pieces feature:
    - Extended rests
    - Very sparse note material
    - Isolated notes with long pauses
    - Minimal melodic content
    """
    if seed is not None:
        random.seed(seed)

    key = random.choice(['C', 'G', 'Am', 'Em'])
    tempo = random.randint(40, 60)
    measures = random.randint(16, 32)

    # Generate very sparse content
    content = []
    notes_placed = 0

    for m in range(measures):
        # Most measures are just rests
        if random.random() < 0.15:  # Only 15% chance of a note
            note_deg = random.randint(1, 7)
            note = scale_degree_to_note(note_deg, key.replace('m', ''), 0)

            # Random short duration followed by rest
            dur = random.choice(['', '2'])
            rest_dur = random.choice(['2', '3', ''])

            content.append(f"z2 {note}{dur} z{rest_dur} |")
            notes_placed += 1
        else:
            # Full measure of rest
            content.append("z4 |")

    # Ensure at least a few notes
    if notes_placed < 3:
        # Replace some rests with sparse notes
        indices = random.sample(range(len(content)), min(3, len(content)))
        for i in indices:
            note_deg = random.randint(1, 7)
            note = scale_degree_to_note(note_deg, key.replace('m', ''), 0)
            content[i] = f"z2 {note} z |"

    abc_content = f"""X:{index}
T:Silence Study {index:03d}
C:Generated
M:4/4
L:1/4
Q:1/4={tempo}
K:{key}
% Sparse, meditative piece with extended silences
{' '.join(content)}
"""

    metadata = ABCMetadata(
        filename=f"silence_{index:03d}.abc",
        genre="silence",
        subgenre="sparse",
        description="Sparse piece with extended silences and isolated notes",
        voices=1,
        measures=measures,
        time_signature="4/4",
        key=key,
        tempo=tempo,
        generation_params={"seed": seed, "algorithm": "sparse_silence", "note_density": 0.15}
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
