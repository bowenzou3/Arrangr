import warnings
import os
from pathlib import Path
from music21 import stream, note, chord, tempo, clef, harmony, key, meter, metadata, duration
import librosa
import numpy as np
from scipy.ndimage import median_filter
from requests.exceptions import RequestsDependencyWarning

# suppress known requests dependency-version warning when environment has newer urllib3/charset-normalizer
warnings.filterwarnings('ignore', category=RequestsDependencyWarning)

# ===========================
# SATB Arrangement Logic
# ===========================
# Voice ranges (MIDI numbers)
RANGES = {
    "soprano": (60, 81),  # C4–A5
    "alto": (55, 74),     # G3–D5
    "tenor": (48, 67),    # C3–G4
    "bass": (40, 60),     # E2–C4
}

# Vocal and guitar performance ranges
VOCAL_RANGE = (60, 76)   # D4–E5 (comfortable for many pop vocal leads)
GUITAR_RANGE = (40, 64)  # E2–E4 (open strings and bar chords)

def fit_to_range(pitch, low, high):
    """Adjusts a MIDI pitch to fit within a specified range, transposing by octaves"""
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return pitch

def voice_chord(ch):
    """Convert a music21 chord into SATB notes, prioritizing root, third, fifth"""
    # Ensure chord has at least 3 distinct pitches for basic triad
    # Use normalized int pitches and avoid unhashable values from weird pitch objects
    pitch_values = []
    for p in ch.pitches:
        try:
            pitch_values.append(int(p.midi))
        except Exception:
            continue

    unique_pitches = []
    for pv in pitch_values:
        if pv not in unique_pitches:
            unique_pitches.append(pv)

    unique_pitches = sorted(unique_pitches)

    if len(unique_pitches) < 3: # Need at least root, third, fifth
        # Fallback for simple chords or arpeggios: just use original notes or a simplified voicing
        if not unique_pitches:
            return None
        # Try to create a basic triad from available notes, or just duplicate
        if len(unique_pitches) == 1: # Only root
            root_midi = unique_pitches[0]
            pitches = [root_midi, root_midi + 4, root_midi + 7]  # Assume major triad
        elif len(unique_pitches) == 2: # Root and one other (e.g., C, G)
            root_midi = unique_pitches[0]
            second_midi = unique_pitches[1]
            if (second_midi - root_midi) % 12 == 7: # Root and fifth
                pitches = [root_midi, root_midi + 4, second_midi]  # Add major third
            else: # Arbitrary other interval, try to complete triad
                pitches = sorted([root_midi, second_midi, root_midi + 7])  # Add fifth
        else:
            pitches = unique_pitches # Should be covered by initial check, but for safety
    else:
        # Simple inversion detection and voicing for triads
        # This prioritizes a basic R-3-5 structure
        # A more advanced voicings would consider inversions more thoroughly
        pitches = unique_pitches # Take the lowest three for root, third, fifth for basic voicing

    # Assign parts: Bass gets root, Tenor gets root/third, Alto gets third/fifth, Soprano gets fifth/higher
    # This is a very basic voicing. For true SATB, proper voice leading and counterpoint rules are critical.
    # We are simplifying to ensure all voices have a note.

    # Try to extract root, third, fifth from the chosen pitches
    root_val = pitches[0]
    third_val = pitches[1] if len(pitches) > 1 else root_val + 4 # Default to major third
    fifth_val = pitches[2] if len(pitches) > 2 else root_val + 7 # Default to perfect fifth

    s_midi = fit_to_range(fifth_val, *RANGES['soprano'])
    a_midi = fit_to_range(third_val, *RANGES['alto'])
    t_midi = fit_to_range(root_val, *RANGES['tenor'])
    b_midi = fit_to_range(root_val - 12, *RANGES['bass'])  # Bass often an octave lower

    # Ensure notes are within range and prevent excessive doubling on small chords
    # Simple check to avoid all voices playing the same note
    if s_midi == a_midi == t_midi == b_midi:
        s_midi += 12 # Spread out if all notes are the same

    return {
        "soprano": note.Note(s_midi),
        "alto": note.Note(a_midi),
        "tenor": note.Note(t_midi),
        "bass": note.Note(b_midi),
    }

def get_diatonic_chord_tables(root_pc, mode='major'):
    """Return a list of triad PC sets for a key (I-ii-iii-IV-V-vi-viiº)."""
    if mode == 'minor':
        # Natural minor triads
        intervals = [0, 2, 3, 5, 7, 8, 10]
        chord_templates = [
            [0, 3, 7], [2, 5, 9], [3, 7, 10], [5, 8, 0],
            [7, 10, 2], [8, 0, 3], [10, 2, 5]
        ]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]
        chord_templates = [
            [0, 4, 7], [2, 5, 9], [4, 7, 11], [5, 9, 0],
            [7, 11, 2], [9, 0, 4], [11, 2, 5]
        ]

    chords = []
    for template in chord_templates:
        chords.append({(root_pc + interval) % 12 for interval in template})

    return chords


def choose_diatonic_chord_note(pitch_class, key_root_pc, mode='major'):
    """Choose a diatonic triad containing the melody pitch class, fallback to I/IV/V."""
    candidate_chords = get_diatonic_chord_tables(key_root_pc, mode)

    # First pass: find any chord that contains the melody note pitch class.
    for i, chord_pcs in enumerate(candidate_chords):
        if pitch_class in chord_pcs:
            return i, chord_pcs

    # Fallback: V, IV, I
    for deg in [4, 3, 0]:
        if 0 <= deg < len(candidate_chords):
            return deg, candidate_chords[deg]

    return 0, candidate_chords[0]


def chord_presents_as_guitar_chord(chord_obj):
    """Construct a guitar-friendly voicing from a chord object."""
    if not chord_obj or not chord_obj.pitches:
        return None

    # Use root of the chord and build a standard open/dropped voicing in mid-register.
    root_name = chord_obj.root().name
    try:
        base = note.Note(root_name + '2')  # E2-E4 region
    except Exception:
        base = note.Note('E2')

    root_midi = base.pitch.midi

    # Use root, third, fifth stacked in the guitar mid-range.
    # Ensure we stay in a playable range by spreading within a 2-octave window.
    triad = chord_obj.commonName if chord_obj.commonName else None
    pitch_classes = [p.pitchClass for p in chord_obj.pitches]

    if len(pitch_classes) < 3:
        # fallback to major third/fifth built from root
        third = root_midi + 4
        fifth = root_midi + 7
    else:
        # keep diatonic chord structure
        sorted_pcs = sorted({p for p in pitch_classes})
        third = root_midi + 4 if (root_midi + 4) % 12 in sorted_pcs else root_midi + 3
        fifth = root_midi + 7 if (root_midi + 7) % 12 in sorted_pcs else root_midi + 8

    # Make sure these notes remain in a guitar-friendly span
    notes = [note.Note(root_midi), note.Note(third), note.Note(fifth)]
    for n in notes:
        n.pitch.midi = fit_to_range(n.pitch.midi, *GUITAR_RANGE)

    return chord.Chord(notes)


def transpose_to_vocal_range(mel_note):
    """Transpose melody note to fit comfortably in vocal range."""
    midi_val = mel_note.pitch.midi
    target = fit_to_range(midi_val, *VOCAL_RANGE)
    n = note.Note(target, duration=mel_note.duration)
    return n


def separate_audio(input_file_path, output_dir='spleeter_output'):
    """Separate vocals and accompaniment using spleeter (2 stems)."""
    try:
        from spleeter.separator import Separator
    except ImportError:
        raise ImportError('spleeter is required for source separation; run pip install spleeter')

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    separator = Separator('spleeter:2stems')
    separator.separate_to_file(str(input_file_path), str(output_base))

    source_dir = output_base / Path(input_file_path).stem
    vocals_path = source_dir / 'vocals.wav'
    accompaniment_path = source_dir / 'accompaniment.wav'

    if not vocals_path.exists() or not accompaniment_path.exists():
        raise FileNotFoundError('Spleeter output files not found.')

    return str(vocals_path), str(accompaniment_path)


def quantize_duration_simple(q):
    """Force simpler rhythmic resolution: quarter/half/whole (minimal 16th spam)."""
    if q <= 0:
        return 0.0
    q = float(q)
    q = max(q, 0.25)
    if q >= 3.5:
        return round(q / 4.0) * 4.0
    elif q >= 1.75:
        return round(q / 2.0) * 2.0
    else:
        return round(q * 4.0) / 4.0


def extract_melody(vocal_file_path, tempo_bpm=120):
    """Extract melody line using librosa piptrack from isolated vocals."""
    y, sr = librosa.load(vocal_file_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    hop_length = 512
    frame_duration = hop_length / float(sr)
    quarter_per_frame = frame_duration * tempo_bpm / 60.0

    melody_notes = []
    prev_pitch = None
    prev_duration = 0.0

    for t in range(pitches.shape[1]):
        mag = np.max(magnitudes[:, t])
        if mag < 0.01:
            pitch_val = None
        else:
            idx = np.argmax(magnitudes[:, t])
            pitch_hz = pitches[idx, t]
            pitch_val = round(librosa.hz_to_midi(pitch_hz)) if pitch_hz > 0 else None

        if pitch_val == prev_pitch:
            prev_duration += quarter_per_frame
        else:
            if prev_pitch is not None:
                dur = quantize_duration_simple(prev_duration)
                if dur > 0:
                    if prev_pitch is None:
                        melody_notes.append(note.Rest(quarterLength=dur))
                    else:
                        n = note.Note(int(prev_pitch), quarterLength=dur)
                        melody_notes.append(n)
            prev_pitch = pitch_val
            prev_duration = quarter_per_frame

    if prev_pitch is not None:
        dur = quantize_duration_simple(prev_duration)
        if dur > 0:
            if prev_pitch is None:
                melody_notes.append(note.Rest(quarterLength=dur))
            else:
                melody_notes.append(note.Note(int(prev_pitch), quarterLength=dur))

    # If melody is empty, fallback to one quarter note rest.
    if not melody_notes:
        melody_notes.append(note.Rest(quarterLength=1.0))

    # Smooth rhythm so repeated adjacent notes are merged into longer values
    melody_notes = smooth_rhythm(melody_notes)
    return melody_notes


def detect_chords(accompaniment_file_path, tempo_bpm=120):
    """Detect chord roots from accompaniment and produce simple triads."""
    y, sr = librosa.load(accompaniment_file_path, sr=None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Smooth over time to reduce jumpy chord detection
    chroma_smooth = median_filter(chroma, size=(1, 9))
    root_trace = np.argmax(chroma_smooth, axis=0)

    # Group frames into chord units (approx 1/4 note per block)
    hop_length = 512
    frame_time = hop_length / float(sr)
    quarter_frames = max(1, int(round((60.0 / tempo_bpm) / frame_time)))

    chord_list = []
    n_frames = root_trace.shape[0]

    for start in range(0, n_frames, quarter_frames):
        end = min(n_frames, start + quarter_frames)
        block_roots = root_trace[start:end]
        block = chroma_smooth[:, start:end]
        if block_roots.size == 0 or block.size == 0:
            continue
        root_pc = int(np.bincount(block_roots).argmax())

        block_avg = np.mean(block, axis=1)
        major_score = block_avg[root_pc] + block_avg[(root_pc + 4) % 12] + block_avg[(root_pc + 7) % 12]
        minor_score = block_avg[root_pc] + block_avg[(root_pc + 3) % 12] + block_avg[(root_pc + 7) % 12]
        mode = 'major' if major_score >= minor_score else 'minor'

        if mode == 'major':
            degrees = [0, 4, 7]
        else:
            degrees = [0, 3, 7]

        chord_notes = [note.Note((root_pc + d) % 12 + 60) for d in degrees]
        c = chord.Chord(chord_notes)
        c.duration = duration.Duration(1.0)
        chord_list.append(c)

    if not chord_list:
        chord_list = [chord.Chord([note.Note('C4'), note.Note('E4'), note.Note('G4')], quarterLength=1.0)]
        root_pc = 0
        mode = 'major'
    else:
        root_pc = int(chord_list[0].root().pitchClass)
        mode = 'major'

    return chord_list, root_pc, mode


# Syllable palettes by voice role and section type
SYLLABLES = {
    "intro": {
        "soprano": ["ah", "oh", "ee", "oo"],
        "alto": ["oh", "mm", "ah"],
        "tenor": ["ah", "oo"],
        "bass": ["oh", "mm"],
        "percussion": ["boot", "tss", "kah"]
    },
    "verse": {
        "soprano": ["ah", "ee", "oh", "oo"],
        "alto": ["la", "oh", "na"],
        "tenor": ["doo", "la", "dum"],
        "bass": ["bum", "oh", "doo"],
        "percussion": ["tss", "boot", "psh"]
    },
    "chorus": {
        "soprano": ["ah", "ee", "oo"],
        "alto": ["la", "na", "shoo"],
        "tenor": ["duh", "la", "yeah"],
        "bass": ["bah", "doo", "dum"],
        "percussion": ["boot", "tss", "kah", "psh"]
    },
    "bridge": {
        "soprano": ["ooh", "aah"],
        "alto": ["mm", "oh"],
        "tenor": ["vee", "doo"],
        "bass": ["mm", "oh"],
        "percussion": ["tss", "psh"]
    },
    "outro": {
        "soprano": ["ah", "mm"],
        "alto": ["oh", "mm"],
        "tenor": ["oo", "ah"],
        "bass": ["mm", "oh"],
        "percussion": ["boot", "tss"]
    }
}

def detect_song_sections(onset_env, sr, duration_seconds):
    """Detect song sections (intro, verse, chorus, etc) from onset strength."""
    # Simplified heuristic: divide song into sections based on total duration
    total_measures = int((duration_seconds * 120) / 60 / 4)  # approx quarters
    
    sections = []
    if total_measures < 16:
        sections = [{"name": "intro", "start": 0, "measures": max(4, total_measures // 3)}]
        sections.append({"name": "verse", "start": sections[0]["measures"], "measures": total_measures - sections[0]["measures"]})
    elif total_measures < 48:
        intro_m = 4
        verse_m = (total_measures - intro_m) // 3
        sections = [
            {"name": "intro", "start": 0, "measures": intro_m},
            {"name": "verse", "start": intro_m, "measures": verse_m},
            {"name": "chorus", "start": intro_m + verse_m, "measures": total_measures - intro_m - verse_m}
        ]
    else:
        intro_m, verse_m, chorus_m, bridge_m = 4, 8, 8, 4
        outro_m = total_measures - intro_m - 2*verse_m - 2*chorus_m - bridge_m
        outro_m = max(4, outro_m)
        sections = [
            {"name": "intro", "start": 0, "measures": intro_m},
            {"name": "verse", "start": intro_m, "measures": verse_m},
            {"name": "chorus", "start": intro_m + verse_m, "measures": chorus_m},
            {"name": "verse", "start": intro_m + verse_m + chorus_m, "measures": verse_m},
            {"name": "chorus", "start": intro_m + 2*verse_m + chorus_m, "measures": chorus_m},
            {"name": "bridge", "start": intro_m + 2*verse_m + 2*chorus_m, "measures": bridge_m},
            {"name": "outro", "start": intro_m + 2*verse_m + 2*chorus_m + bridge_m, "measures": outro_m}
        ]
    return sections

def get_syllable(voice, section_name, event_index):
    """Select appropriate syllable for a voice in a section."""
    syllable_set = SYLLABLES.get(section_name, SYLLABLES["verse"]).get(voice, ["ah"])
    return syllable_set[event_index % len(syllable_set)]

def voice_chord_smooth(ch, prev_voicing=None):
    """Voice chord with proper SATB rules: smooth voice leading, avoid parallel 5ths/octaves."""
    if not ch or not ch.pitches:
        return None
    
    pitch_values = sorted(list({int(p.midi) for p in ch.pitches if isinstance(p.midi, (int, float))}))
    
    if len(pitch_values) < 3:
        if len(pitch_values) == 1:
            root_midi = pitch_values[0]
            pitches = [root_midi, root_midi + 4, root_midi + 7]
        elif len(pitch_values) == 2:
            root_midi, second = pitch_values[0], pitch_values[1]
            if (second - root_midi) % 12 == 7:
                pitches = [root_midi, root_midi + 4, second]
            else:
                pitches = [root_midi, second, root_midi + 7]
        else:
            pitches = pitch_values
    else:
        pitches = pitch_values[:3]
    
    root_val, third_val, fifth_val = pitches[0], pitches[1], pitches[2]
    
    # Smooth voice leading: if previous voicing exists, move by step when possible
    if prev_voicing:
        s_midi = fit_to_range(fifth_val, *RANGES['soprano'])
        a_midi = fit_to_range(third_val, *RANGES['alto'])
        t_midi = fit_to_range(root_val, *RANGES['tenor'])
        b_midi = fit_to_range(root_val - 12, *RANGES['bass'])
        
        # Prefer closest motion from previous voicing
        candidates = {
            "S": [fit_to_range(fifth_val + o*12, *RANGES['soprano']) for o in [-1, 0, 1]],
            "A": [fit_to_range(third_val + o*12, *RANGES['alto']) for o in [-1, 0, 1]],
            "T": [fit_to_range(root_val + o*12, *RANGES['tenor']) for o in [-1, 0, 1]],
            "B": [fit_to_range(root_val - 12 + o*12, *RANGES['bass']) for o in [-1, 0, 1]]
        }
        
        if prev_voicing.get("S"):
            s_midi = min(candidates["S"], key=lambda x: abs(x - prev_voicing["S"].pitch.midi))
        if prev_voicing.get("A"):
            a_midi = min(candidates["A"], key=lambda x: abs(x - prev_voicing["A"].pitch.midi))
        if prev_voicing.get("T"):
            t_midi = min(candidates["T"], key=lambda x: abs(x - prev_voicing["T"].pitch.midi))
        if prev_voicing.get("B"):
            b_midi = min(candidates["B"], key=lambda x: abs(x - prev_voicing["B"].pitch.midi))
    else:
        s_midi = fit_to_range(fifth_val, *RANGES['soprano'])
        a_midi = fit_to_range(third_val, *RANGES['alto'])
        t_midi = fit_to_range(root_val, *RANGES['tenor'])
        b_midi = fit_to_range(root_val - 12, *RANGES['bass'])
    
    return {
        "S": note.Note(s_midi),
        "A": note.Note(a_midi),
        "T": note.Note(t_midi),
        "B": note.Note(b_midi),
    }


def apply_spacing_strategy(voiced, spacing='staggered'):
    """Apply spacing rules to a voiced SATB chord."""
    if not voiced:
        return voiced

    if spacing == 'staggered':
        b = fit_to_range(voiced['B'].pitch.midi, *RANGES['bass'])
        t = max(fit_to_range(voiced['T'].pitch.midi, *RANGES['tenor']), b + 5)
        a = max(fit_to_range(voiced['A'].pitch.midi, *RANGES['alto']), t + 3)
        s = max(fit_to_range(voiced['S'].pitch.midi, *RANGES['soprano']), a + 3)

        voiced = {
            'B': note.Note(b),
            'T': note.Note(t),
            'A': note.Note(a),
            'S': note.Note(s)
        }
    elif spacing == 'block':
        # Keep block spacing but respect ranges
        voiced = {
            'B': note.Note(fit_to_range(voiced['B'].pitch.midi, *RANGES['bass'])),
            'T': note.Note(fit_to_range(voiced['T'].pitch.midi, *RANGES['tenor'])),
            'A': note.Note(fit_to_range(voiced['A'].pitch.midi, *RANGES['alto'])),
            'S': note.Note(fit_to_range(voiced['S'].pitch.midi, *RANGES['soprano']))
        }
    return voiced


def interval(n1, n2):
    return abs(n1 - n2) % 12


def is_parallel(interval1, interval2):
    return interval1 == interval2 and interval1 in (7, 12)


def fix_voice_leading(prev_voicing, new_voicing):
    """Avoid robotic parallel fifths/octaves by slight upper voice adjustment."""
    if not prev_voicing or not new_voicing:
        return new_voicing

    result = {}
    for v in ['S','A','T','B']:
        if new_voicing.get(v):
            result[v] = note.Note(new_voicing[v].pitch.midi)
        else:
            result[v] = None

    for i, v1 in enumerate(['S', 'A', 'T', 'B']):
        for v2 in ['A', 'T', 'B'][i:]:
            if prev_voicing.get(v1) and prev_voicing.get(v2) and result.get(v1) and result.get(v2):
                prev_int = interval(prev_voicing[v1].pitch.midi, prev_voicing[v2].pitch.midi)
                new_int = interval(result[v1].pitch.midi, result[v2].pitch.midi)
                if is_parallel(prev_int, new_int):
                    result[v1].pitch.midi = fit_to_range(result[v1].pitch.midi + 1, *RANGES['soprano'])

    return result


def smooth_rhythm(note_sequence):
    """Merge successive same notes/rests into longer durations."""
    if not note_sequence:
        return note_sequence

    merged = [note_sequence[0]]
    for item in note_sequence[1:]:
        last = merged[-1]
        same = False

        if isinstance(last, note.Rest) and isinstance(item, note.Rest):
            same = True
        elif isinstance(last, note.Note) and isinstance(item, note.Note):
            same = last.pitch.midi == item.pitch.midi

        if same:
            last.duration.quarterLength += item.duration.quarterLength
        else:
            merged.append(item)

    return merged


def arrange(chord_progression, melody_notes, title='Arrangr Acapella',
            key_signature=0, time_signature='4/4', tempo_bpm=120):
    """
    Create professional SATB acapella arrangement with proper voice leading.
    Solo line appears ONLY when melody is active (has pitch content).
    SATB provides harmonic backup with appropriate syllables per section.
    """
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title

    # Global markers
    score.insert(0, key.KeySignature(key_signature))
    score.insert(0, meter.TimeSignature(time_signature))
    score.insert(0, tempo.MetronomeMark(number=tempo_bpm))

    # Create parts
    solo = stream.Part()
    soprano = stream.Part()
    alto = stream.Part()
    tenor = stream.Part()
    bass = stream.Part()

    solo.partName = 'Solo'
    soprano.partName = 'Soprano'
    alto.partName = 'Alto'
    tenor.partName = 'Tenor'
    bass.partName = 'Bass'

    # Assign clefs and metadata
    for part, c in [(solo, clef.TrebleClef()), (soprano, clef.TrebleClef()),
                    (alto, clef.TrebleClef()), (tenor, clef.TrebleClef()),
                    (bass, clef.BassClef())]:
        part.insert(0, c)
        part.insert(0, tempo.MetronomeMark(number=tempo_bpm))
        part.insert(0, key.KeySignature(key_signature))
        part.insert(0, meter.TimeSignature(time_signature))

    # Estimate song duration and detect sections
    total_events = max(len(melody_notes), len(chord_progression))
    duration_seconds = sum([n.duration.quarterLength for n in melody_notes if n]) * (60 / tempo_bpm) if melody_notes else 30
    sections = analyze_song_sections(max(1, total_events // 4))
    
    # Map measures to section definitions for strategy-based voice activity
    measures_per_section = {}
    cumulative = 0
    for sec in sections:
        for m in range(cumulative, cumulative + sec['measures']):
            measures_per_section[m] = sec
        cumulative += sec['measures']

    prev_voicing = None

    for i in range(total_events):
        mel_note = melody_notes[i] if i < len(melody_notes) else None
        current_chord = chord_progression[i] if i < len(chord_progression) else None
        
        # Determine section for this event
        section_idx = i // 4  # Roughly 4 events per measure
        section_def = measures_per_section.get(section_idx, sections[-1]) if measures_per_section else {'name': 'verse', 'texture': 'medium'}
        section_name = section_def['name']
        soloist_active = section_name in ['verse', 'chorus', 'bridge', 'outro']
        strategy = get_section_arrangement_strategy(section_name, section_def.get('texture', 'medium'), soloist_active)

        voice_active = {
            'S': strategy['voices'].get('S', 'active') == 'active',
            'A': strategy['voices'].get('A', 'active') == 'active',
            'T': strategy['voices'].get('T', 'active') == 'active',
            'B': strategy['voices'].get('B', 'active') == 'active'
        }

        # Check if melody has actual pitch content
        has_melody = mel_note is not None and hasattr(mel_note, 'pitch') and mel_note.pitch is not None
        
        if has_melody:
            event_duration = round(mel_note.duration.quarterLength * 4) / 4
            event_duration = max(event_duration, 0.25)
        elif current_chord is not None and hasattr(current_chord, 'duration'):
            event_duration = round(current_chord.duration.quarterLength * 4) / 4
            event_duration = max(event_duration, 0.25)
        else:
            event_duration = 1.0

        # SOLO: Add only if melody is active (has pitch)
        if has_melody:
            solo_note = transpose_to_vocal_range(mel_note)
            solo_note.lyric = mel_note.lyric if hasattr(mel_note, 'lyric') and mel_note.lyric else 'ah'
            solo.append(solo_note)
        else:
            solo.append(note.Rest(quarterLength=event_duration))

        # SATB BACKUP: discretized, sparse, phrase-aware voicing
        if current_chord is not None and hasattr(current_chord, 'pitches') and current_chord.pitches:
            base_voicing = voice_chord_smooth(current_chord, prev_voicing)
            base_voicing = apply_spacing_strategy(base_voicing, strategy.get('spacing', 'staggered'))

            controlled = {"S": None, "A": None, "T": None, "B": None}

            if has_melody:
                # Soft choir support during solo
                if i % 4 == 0:
                    controlled['A'] = base_voicing.get('A')
                if i % 2 == 0:
                    controlled['B'] = base_voicing.get('B')
            else:
                # Choir section (no solo) with structured entry
                root_midi = current_chord.root().pitch.midi if current_chord.root() else (base_voicing['T'].pitch.midi if base_voicing and base_voicing.get('T') else 60)
                controlled['S'] = note.Note(fit_to_range(root_midi + 7, *RANGES['soprano']))
                controlled['A'] = note.Note(fit_to_range(root_midi + 4, *RANGES['alto'])) if i % 2 == 0 else None
                controlled['T'] = note.Note(fit_to_range(root_midi, *RANGES['tenor']))
                controlled['B'] = note.Note(fit_to_range(root_midi - 12, *RANGES['bass']))

            controlled = fix_voice_leading(prev_voicing, controlled)
            prev_voicing = controlled

            for voice_name, part in [('S', soprano), ('A', alto), ('T', tenor), ('B', bass)]:
                if not voice_active.get(voice_name, True):
                    part.append(note.Rest(quarterLength=event_duration))
                    continue

                vnote = controlled.get(voice_name)
                if vnote is None:
                    part.append(note.Rest(quarterLength=event_duration))
                else:
                    vnote.duration = duration.Duration(event_duration)
                    vnote.lyric = get_syllable({'S':'soprano','A':'alto','T':'tenor','B':'bass'}[voice_name], section_name, i)
                    part.append(vnote)
        else:
            for part in [soprano, alto, tenor, bass]:
                part.append(note.Rest(quarterLength=event_duration))

    # Insert parts in score (solo first)
    score.insert(0, bass)
    score.insert(0, tenor)
    score.insert(0, alto)
    score.insert(0, soprano)
    score.insert(0, solo)

    return score

# ===========================
# Audio Processing (MP3/WAV)
# ===========================
def audio_to_chords_and_melody(audio_file_path):
    """Convert audio (wav or mp3) to chord progression and melody using librosa + optional spleeter."""
    print(f"Analyzing audio file: {audio_file_path}")

    try:
        vocals_file, accompaniment_file = separate_audio(audio_file_path)
    except Exception as e:
        print('Warning: source separation unavailable or failed, using full mix. Error:', e)
        vocals_file, accompaniment_file = audio_file_path, audio_file_path

    # Estimate tempo from accompaniment track
    y_acc, sr_acc = librosa.load(accompaniment_file, sr=None)
    onset_env = librosa.onset.onset_strength(y=y_acc, sr=sr_acc)
    if len(onset_env) >= 2:
        tempo_bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_acc)
    else:
        tempo_bpm, beats = 120.0, np.array([], dtype=int)

    try:
        tempo_bpm = float(tempo_bpm)
    except Exception:
        tempo_bpm = 120.0

    # Melody (solo) from vocals
    melody_list = extract_melody(vocals_file, tempo_bpm=tempo_bpm)

    # Chords from accompaniment track
    chord_list, root_pc, mode = detect_chords(accompaniment_file, tempo_bpm=tempo_bpm)

    # Align and trim chord/melody lists
    if len(chord_list) < len(melody_list) and len(chord_list) > 0:
        repeat_factor = int(np.ceil(len(melody_list) / len(chord_list)))
        chord_list = (chord_list * repeat_factor)[:len(melody_list)]
    elif len(chord_list) > len(melody_list):
        chord_list = chord_list[:len(melody_list)]

    if not melody_list:
        melody_list = [note.Rest(quarterLength=1.0)]
    if not chord_list:
        chord_list = [chord.Chord([note.Note('C4'), note.Note('E4'), note.Note('G4')], quarterLength=1.0)]
        root_pc = 0

    return chord_list, melody_list, int(tempo_bpm), root_pc

# ===========================
# SIMPLE SATB ARRANGEMENT (per basic rules)
# ===========================
def analyze_song_sections(total_measures):
    """
    Detect and structure song sections for smart arrangement decisions.
    Returns list of section definitions with measures and arrangement strategy.
    """
    sections = []
    
    if total_measures <= 8:
        # Short song: intro + main
        sections = [
            {"name": "intro", "measures": min(2, total_measures // 4), "texture": "thin"},
            {"name": "main", "measures": total_measures - min(2, total_measures // 4), "texture": "full"}
        ]
    elif total_measures <= 32:
        # Standard pop: intro + verse + chorus
        intro_m = 2
        verse_m = (total_measures - intro_m - 8) // 2
        sections = [
            {"name": "intro", "measures": intro_m, "texture": "thin"},
            {"name": "verse", "measures": verse_m, "texture": "medium"},
            {"name": "chorus", "measures": 8, "texture": "full"}
        ]
    else:
        # Full song: intro + verse + chorus + verse + chorus + bridge + outro
        intro_m = 4
        verse_m = 8
        chorus_m = 8
        bridge_m = 4
        outro_m = total_measures - (intro_m + 2*verse_m + 2*chorus_m + bridge_m)
        outro_m = max(4, outro_m)
        
        sections = [
            {"name": "intro", "measures": intro_m, "texture": "thin"},
            {"name": "verse", "measures": verse_m, "texture": "medium"},
            {"name": "chorus", "measures": chorus_m, "texture": "full"},
            {"name": "verse", "measures": verse_m, "texture": "medium"},
            {"name": "chorus", "measures": chorus_m, "texture": "full"},
            {"name": "bridge", "measures": bridge_m, "texture": "medium"},
            {"name": "outro", "measures": outro_m, "texture": "full"}
        ]
    
    return sections


def get_section_arrangement_strategy(section_name, texture, soloist_active):
    """
    Define how voices should behave in each section type.
    Returns dict with voice activity and arrangement approach.
    """
    strategies = {
        "intro": {
            "thin": {
                "voices": {"S": "rest", "A": "active", "T": "active", "B": "rest"},
                "spacing": "staggered",
                "rhythm": "half/whole notes",
                "dynamics": "p",
                "notes": "Light, intimate start with just middle voices sustained on 'oo'"
            },
            "medium": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "rest"},
                "spacing": "staggered",
                "rhythm": "half/whole notes",
                "dynamics": "mp",
                "notes": "Build with three voices, Bass enters late or rests"
            },
            "full": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "active"},
                "spacing": "block",
                "rhythm": "quarter/half notes",
                "dynamics": "mf",
                "notes": "Full SATB block chord foundation, steady and clean"
            }
        },
        "verse": {
            "thin": {
                "voices": {"S": "rest", "A": "active", "T": "active", "B": "active"},
                "spacing": "staggered",
                "rhythm": "half/whole notes",
                "dynamics": "mp",
                "notes": "Soloist shines: choir sustains on 'oo', let melody breathe. Bass enters with half notes."
            },
            "medium": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "active"},
                "spacing": "staggered",
                "rhythm": "quarter/half notes",
                "dynamics": "mf",
                "notes": "All voices active but at different rhythmic layers. S/A sustain while T/B track harmony."
            },
            "full": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "active"},
                "spacing": "block",
                "rhythm": "quarter/half notes",
                "dynamics": "f",
                "notes": "SATB block chords under soloist, rhythmically unified but clean"
            }
        },
        "chorus": {
            "thin": {
                "voices": {"S": "active", "A": "active", "T": "rest", "B": "rest"},
                "spacing": "staggered",
                "rhythm": "quarter notes",
                "dynamics": "mf",
                "notes": "Soprano + Alto harmony, light texture, lets soloist shine"
            },
            "medium": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "rest"},
                "spacing": "staggered",
                "rhythm": "quarter/half notes",
                "dynamics": "mf",
                "notes": "Three-voice harmony, Tenor adds warmth, Bass rests to keep space for melody"
            },
            "full": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "active"},
                "spacing": "block",
                "rhythm": "quarter/half notes",
                "dynamics": "f",
                "notes": "Full SATB, rhythmically unified, punchy and supportive"
            }
        },
        "bridge": {
            "thin": {
                "voices": {"S": "rest", "A": "active", "T": "active", "B": "active"},
                "spacing": "staggered",
                "rhythm": "half notes",
                "dynamics": "mp",
                "notes": "Stripped down, vulnerable texture; focus on melody with minimal support"
            },
            "medium": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "rest"},
                "spacing": "staggered",
                "rhythm": "half notes",
                "dynamics": "mp",
                "notes": "Three voices only, create contrast before final chorus. Sustained, sparse"
            },
            "full": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "active"},
                "spacing": "block",
                "rhythm": "quarter/half notes",
                "dynamics": "mf",
                "notes": "Building section: full harmony to transition back to chorus energy"
            }
        },
        "outro": {
            "thin": {
                "voices": {"S": "rest", "A": "active", "T": "rest", "B": "active"},
                "spacing": "sparse",
                "rhythm": "whole notes",
                "dynamics": "p",
                "notes": "Fade out: keep only outer voices for intimate close"
            },
            "medium": {
                "voices": {"S": "rest", "A": "active", "T": "active", "B": "active"},
                "spacing": "staggered",
                "rhythm": "whole/half notes",
                "dynamics": "p",
                "notes": "Three voices diminish, sustained final chord, ritardando feel"
            },
            "full": {
                "voices": {"S": "active", "A": "active", "T": "active", "B": "active"},
                "spacing": "block",
                "rhythm": "whole notes",
                "dynamics": "p",
                "notes": "Full SATB final chord held, serene conclusion"
            }
        }
    }
    
    # If soloist is active, scale back choir slightly
    strategy = strategies.get(section_name, strategies["verse"]).get(texture, strategies["verse"]["medium"]).copy()
    
    if soloist_active:
        # When soloist is active, choir plays supporting role
        if strategy["voices"]["S"] == "active":
            strategy["dynamics"] = "mp" if strategy["dynamics"] == "mf" else "p"
    
    return strategy


def create_professional_arrangement(melody_notes, chord_progression, title='Arrangement'):
    """
    Create a PROFESSIONAL, MUSICALLY INTELLIGENT SATB + SOLOIST arrangement.
    
    Key principles:
    - Soloist carries the melody
    - Choir provides intelligent, layered support
    - Use space and silence
    - Build texture across sections
    - Think in phrases, not beats
    
    Returns JSON dict with complete arrangement metadata.
    """
    
    # Estimate total measures
    total_measures = max(len(melody_notes) if melody_notes else 0, len(chord_progression) if chord_progression else 0) // 4
    total_measures = max(8, total_measures)  # Minimum 8 measures
    
    # Analyze structure
    section_list = analyze_song_sections(total_measures)
    
    # Build arrangement sections
    sections = []
    measure_counter = 0
    
    for section_def in section_list:
        section_name = section_def["name"]
        section_measures = section_def["measures"]
        base_texture = section_def["texture"]
        
        # Determine if soloist is active (mostly in verses/choruses)
        soloist_active = section_name in ["verse", "chorus", "bridge", "outro"]
        if section_name == "intro":
            soloist_active = False
        
        # Get arrangement strategy for this section
        strategy = get_section_arrangement_strategy(section_name, base_texture, soloist_active)
        
        # Create arrangement section
        arrangement_section = {
            "section_name": section_name,
            "texture": base_texture,
            "solo_active": soloist_active,
            
            "voices": strategy["voices"].copy(),
            
            "rhythm_style": strategy["rhythm"],
            "spacing_strategy": strategy["spacing"],
            "notes_density": "low" if base_texture == "thin" else ("medium" if base_texture == "medium" else "high"),
            "dynamics": strategy["dynamics"],
            
            "voice_behavior": {
                "S": "silence" if strategy["voices"]["S"] == "rest" else ("upper pad/harmony" if soloist_active else "melody line"),
                "A": "sustained harmony on 'oo'" if strategy["voices"]["A"] == "active" else "silence",
                "T": "inner harmony, slower rhythm" if strategy["voices"]["T"] == "active" else "silence",
                "B": "root movement, long notes" if strategy["voices"]["B"] == "active" else "silence",
                "solo": "main melody" if soloist_active else "silence"
            },
            
            "arrangement_notes": strategy.get("notes", ""),
            "measures": section_measures,
            "measure_span": f"{measure_counter + 1}-{measure_counter + section_measures}"
        }
        
        sections.append(arrangement_section)
        measure_counter += section_measures
    
    # Return professional arrangement JSON
    arrangement_json = {
        "clefs": {
            "S": "treble",
            "A": "treble",
            "T": "treble",
            "B": "bass",
            "solo": "treble"
        },
        "sections": sections,
        "title": title,
        "total_measures": total_measures,
        "arrangement_philosophy": "Soloist-centric with intelligent choir support. Emphasizes space, phrase structure, and singability."
    }
    
    return arrangement_json


def stretch_notes(note_list, factor=2.0):
    """Stretch durations in a music21 note/rest list."""
    for n in note_list:
        if isinstance(n, (note.Note, note.Rest)) and n.duration is not None:
            n.duration.quarterLength = max(0.25, n.duration.quarterLength * factor)
    return note_list


def classical_style(score_or_arrangement):
    """Apply classical-style texture to either a music21 score or arrangement dict."""
    if isinstance(score_or_arrangement, stream.Score):
        for part in score_or_arrangement.parts:
            stretch_notes([n for n in part.recurse().notesAndRests], factor=1.8)
        # Optionally set lyrics/syllables for each part
        for part in score_or_arrangement.parts:
            for n in part.recurse().notes:
                if n.lyric is None or n.lyric.strip() == "":
                    n.lyric = 'ah'
        return score_or_arrangement

    if isinstance(score_or_arrangement, dict):
        score_or_arrangement['style'] = 'classical'
        score_or_arrangement['vowels'] = 'ah/oo'
        score_or_arrangement['dynamics'] = 'mp-mf'
        return score_or_arrangement

    return score_or_arrangement


def contemporary_style(score_or_arrangement):
    """Apply contemporary (Pentatonix-like) style texture."""
    if isinstance(score_or_arrangement, stream.Score):
        # Make bass more rhythmic by splitting long notes in bass part
        bass_part = score_or_arrangement.parts['Bass'] if 'Bass' in score_or_arrangement.parts else None
        if bass_part:
            for n in list(bass_part.recurse().notesAndRests):
                if isinstance(n, note.Note) and n.duration.quarterLength >= 1.0:
                    # turn one long note into 2 staccato hits
                    n.duration.quarterLength = n.duration.quarterLength / 2
                    n.lyric = 'dum'
            for part_name in ['Soprano', 'Alto', 'Tenor']:
                prt = score_or_arrangement.parts[part_name] if part_name in score_or_arrangement.parts else None
                if prt:
                    for n in prt.recurse().notes:
                        if n.lyric is None or n.lyric.strip() == '':
                            n.lyric = np.random.choice(['doo', 'bap', 'tsk'])
        return score_or_arrangement

    if isinstance(score_or_arrangement, dict):
        score_or_arrangement['style'] = 'contemporary'
        score_or_arrangement['vowels'] = ['doo', 'bap', 'tsk']
        score_or_arrangement['dynamics'] = 'mf-f'
        return score_or_arrangement

    return score_or_arrangement


def apply_style(arrangement_or_score, style='classical'):
    """Dispatch style transfer rule-set."""
    if style == 'classical':
        return classical_style(arrangement_or_score)
    elif style == 'contemporary':
        return contemporary_style(arrangement_or_score)
    return arrangement_or_score


def align_lyrics(melody_notes, lyrics):
    """Align words to melody notes for realistic lyric attachment."""
    words = [w for w in lyrics.strip().split() if w]
    result = []
    idx = 0

    for n in melody_notes:
        if isinstance(n, note.Rest) or n is None:
            result.append(None)
        else:
            if idx < len(words):
                result.append(words[idx])
                idx += 1
            else:
                result.append('_')

    return result


def export_midi(score, output_path='output/choir.mid'):
    """Export music21 score to MIDI using pretty_midi for choir soundfont path usage."""
    try:
        import pretty_midi
    except ImportError:
        raise ImportError('pretty_midi is required: pip install pretty_midi')

    pm = pretty_midi.PrettyMIDI()
    program = pretty_midi.instrument_name_to_program('Choir Aahs') if hasattr(pretty_midi, 'instrument_name_to_program') else 0

    tempo_bpm = 120.0
    mm = next(score.recurse().getElementsByClass(tempo.MetronomeMark), None)
    if mm is not None and mm.number:
        tempo_bpm = float(mm.number)

    sec_per_quarter = 60.0 / tempo_bpm

    for part in score.parts:
        instr = pretty_midi.Instrument(program=program, name=part.partName)
        time_cursor = 0.0
        for element in part.recurse().notesAndRests:
            dur_q = float(element.duration.quarterLength) if element.duration else 1.0
            dur = dur_q * sec_per_quarter
            if isinstance(element, note.Note):
                pmn = pretty_midi.Note(velocity=80, pitch=int(element.pitch.midi), start=time_cursor, end=time_cursor + dur)
                instr.notes.append(pmn)
            time_cursor += dur
        pm.instruments.append(instr)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(output_path)
    return output_path


# ===========================================================================
# MP3 → Simple SATB + Soloist A Cappella Arrangement Pipeline
# ===========================================================================
# Rules:
# - Soloist = melody only
# - SATB = sustained harmony (whole/half notes only)
# - Bass = root notes, bass clef
# - Syllables: S/A→"oo"/"ah", T→"oh", B→"doo"/"dum", Solo→lyrics or "ah"

# ---------------------------------------------------------------------------
# Voice ranges (MIDI note numbers) for the MP3 pipeline
# ---------------------------------------------------------------------------
VOICE_RANGES = {
    "solo": (60, 84),   # C4–C6  (soprano soloist)
    "S":    (60, 81),   # C4–A5
    "A":    (53, 69),   # F3–A4
    "T":    (48, 67),   # C3–G4
    "B":    (40, 60),   # E2–C4
}

# Chord-tone voicing templates per quality (semitone offsets from root)
CHORD_VOICINGS = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
}

# Simple per-voice syllable defaults for the MP3 pipeline
SATB_SYLLABLES = {
    "solo": "ah",
    "S":    "oo",
    "A":    "ah",
    "T":    "oh",
    "B":    "doo",
}


# ---------------------------------------------------------------------------
# MP3 Pipeline Step 1 — Load Audio
# ---------------------------------------------------------------------------
def load_audio(path: str, sr: int = 22050):
    """Load MP3/WAV; return (y, sr)."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


# ---------------------------------------------------------------------------
# MP3 Pipeline Step 2 — Extract Tempo + Beat Grid
# ---------------------------------------------------------------------------
def extract_tempo_beats(y: np.ndarray, sr: int):
    """Return (bpm_float, beat_times_array)."""
    tempo_est, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(np.round(tempo_est, 1))
    return bpm, beat_times


# ---------------------------------------------------------------------------
# MP3 Pipeline Step 3 — Extract Melody via pyin (quantised to beats)
# ---------------------------------------------------------------------------
def pyin_extract_melody(y: np.ndarray, sr: int, beat_times: np.ndarray) -> list:
    """
    Use librosa pyin to track F0 and quantise to beat grid.
    Returns list of {beat_index, midi_pitch (or None), duration_beats}.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C3"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )
    times = librosa.times_like(f0, sr=sr)

    melody_beats = []
    for i, bt in enumerate(beat_times):
        t_start = bt
        t_end   = beat_times[i + 1] if i + 1 < len(beat_times) else bt + 0.5

        mask = (times >= t_start) & (times < t_end) & voiced_flag
        if mask.sum() > 0:
            median_f0 = float(np.median(f0[mask]))
            midi = int(np.round(librosa.hz_to_midi(median_f0)))
            midi = int(np.clip(midi, *VOICE_RANGES["solo"]))
        else:
            midi = None  # rest

        melody_beats.append({"beat": i, "midi": midi})

    return melody_beats


# ---------------------------------------------------------------------------
# MP3 Pipeline Step 4 — Detect Chords via chromagram (one per measure)
# ---------------------------------------------------------------------------
def chromagram_detect_chords(y: np.ndarray, sr: int, beat_times: np.ndarray,
                              beats_per_measure: int = 4) -> list:
    """
    Chromagram → one chord label per measure.
    Returns list of {measure, root_midi, quality, pitches}.
    """
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]),
                                          sr=sr, hop_length=hop_length)

    num_measures = max(1, len(beat_times) // beats_per_measure)
    chords_out   = []

    chord_templates = {
        "maj":  np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float),
        "min":  np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float),
        "dom7": np.array([1,0,0,0,1,0,0,1,0,0,1,0], dtype=float),
        "maj7": np.array([1,0,0,0,1,0,0,1,0,0,0,1], dtype=float),
        "min7": np.array([1,0,0,1,0,0,0,1,0,0,1,0], dtype=float),
    }
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    for m in range(num_measures):
        b_start = m * beats_per_measure
        b_end   = min(b_start + beats_per_measure, len(beat_times))

        t_start = beat_times[b_start]
        t_end   = beat_times[b_end - 1] + 0.01

        mask = (frame_times >= t_start) & (frame_times < t_end)
        if mask.sum() == 0:
            chords_out.append({"measure": m, "root_midi": 48,
                                "root_name": "C", "quality": "maj",
                                "pitches": [48, 52, 55]})
            continue

        mean_chroma = chroma[:, mask].mean(axis=1)
        mean_chroma /= (mean_chroma.max() + 1e-9)

        best_score  = -1
        best_root   = 0
        best_qual   = "maj"

        for root in range(12):
            rotated = np.roll(mean_chroma, -root)
            for qual, tmpl in chord_templates.items():
                score = float(np.dot(rotated, tmpl))
                if score > best_score:
                    best_score = score
                    best_root  = root
                    best_qual  = qual

        offsets   = CHORD_VOICINGS.get(best_qual, [0, 4, 7])
        root_midi = 48 + best_root          # root anchored in octave 3
        pitches   = [(root_midi + o) for o in offsets]

        chords_out.append({
            "measure":    m,
            "root_midi":  root_midi,
            "root_name":  note_names[best_root],
            "quality":    best_qual,
            "pitches":    pitches,
        })

    return chords_out


# ---------------------------------------------------------------------------
# MP3 Pipeline Step 5 — Simplify Rhythm → whole / half notes only
# ---------------------------------------------------------------------------
def beat_simplify_rhythm(melody_beats: list,
                         beats_per_measure: int = 4) -> list:
    """
    Collapse consecutive same-pitch beats into whole (4) or half (2) notes.
    Returns list of {start_beat, midi, duration_beats}.
    """
    if not melody_beats:
        return []

    simplified = []
    i = 0
    while i < len(melody_beats):
        current_midi = melody_beats[i]["midi"]
        run = 1
        while (i + run < len(melody_beats) and
               melody_beats[i + run]["midi"] == current_midi and
               run < beats_per_measure):
            run += 1

        dur = 4 if run >= 3 else 2

        simplified.append({
            "start_beat":     melody_beats[i]["beat"],
            "midi":           current_midi,
            "duration_beats": dur,
        })
        i += run

    return simplified


# ---------------------------------------------------------------------------
# MP3 Pipeline Step 6 — Assign SATB Voices
# ---------------------------------------------------------------------------
def _fit_to_range(midi: int, lo: int, hi: int) -> int:
    """Octave-shift midi into [lo, hi]."""
    while midi < lo:
        midi += 12
    while midi > hi:
        midi -= 12
    return int(np.clip(midi, lo, hi))


def _closest_in_range(pitches: list, lo: int, hi: int,
                       prefer: str = "mid") -> int:
    """Pick chord tone (octave-adjusted) closest to range midpoint."""
    candidates = list({_fit_to_range(p, lo, hi) for p in pitches})
    if not candidates:
        return (lo + hi) // 2
    mid = (lo + hi) // 2
    if prefer == "high":
        return max(candidates)
    if prefer == "low":
        return min(candidates)
    return min(candidates, key=lambda x: abs(x - mid))


def assign_satb(simplified_melody: list,
                chords: list,
                beats_per_measure: int = 4) -> dict:
    """
    Distribute melody + chord data across Solo/S/A/T/B parts.
    Returns dict with keys solo/S/A/T/B, each a list of
    {start_beat, midi (None=rest), duration_beats}.
    """
    parts = {v: [] for v in ("solo", "S", "A", "T", "B")}

    if simplified_melody:
        last = simplified_melody[-1]
        total_beats = last["start_beat"] + last["duration_beats"]
    else:
        total_beats = len(chords) * beats_per_measure

    chord_map = {c["measure"]: c for c in chords}

    # Solo line
    for seg in simplified_melody:
        parts["solo"].append({
            "start_beat":     seg["start_beat"],
            "midi":           seg["midi"],
            "duration_beats": seg["duration_beats"],
        })

    num_measures = (total_beats + beats_per_measure - 1) // beats_per_measure

    for m in range(num_measures):
        beat_start = m * beats_per_measure
        c = chord_map.get(m, {"pitches": [48, 52, 55, 59], "root_midi": 48})
        pitches = c["pitches"]
        root    = c["root_midi"]

        solo_singing = any(
            seg["midi"] is not None and
            beat_start < seg["start_beat"] + seg["duration_beats"] and
            seg["start_beat"] < beat_start + beats_per_measure
            for seg in simplified_melody
        )

        dur = beats_per_measure  # whole note per measure

        if solo_singing:
            # Soprano rests; A/T/B sustain softly
            parts["S"].append({"start_beat": beat_start, "midi": None, "duration_beats": dur})
            parts["A"].append({"start_beat": beat_start,
                                "midi": _closest_in_range(pitches, *VOICE_RANGES["A"]),
                                "duration_beats": dur})
            parts["T"].append({"start_beat": beat_start,
                                "midi": _closest_in_range(pitches, *VOICE_RANGES["T"]),
                                "duration_beats": dur})
            parts["B"].append({"start_beat": beat_start,
                                "midi": _fit_to_range(root, *VOICE_RANGES["B"]),
                                "duration_beats": dur})
        else:
            # Full SATB carries chord when solo rests
            parts["S"].append({"start_beat": beat_start,
                                "midi": _closest_in_range(pitches, *VOICE_RANGES["S"], prefer="high"),
                                "duration_beats": dur})
            parts["A"].append({"start_beat": beat_start,
                                "midi": _closest_in_range(pitches, *VOICE_RANGES["A"]),
                                "duration_beats": dur})
            parts["T"].append({"start_beat": beat_start,
                                "midi": _closest_in_range(pitches, *VOICE_RANGES["T"]),
                                "duration_beats": dur})
            parts["B"].append({"start_beat": beat_start,
                                "midi": _fit_to_range(root, *VOICE_RANGES["B"]),
                                "duration_beats": dur})
            parts["solo"].append({"start_beat": beat_start, "midi": None, "duration_beats": dur})

    return parts


# ---------------------------------------------------------------------------
# MP3 Pipeline Step 7 — Assign Syllables
# ---------------------------------------------------------------------------
def assign_syllables_satb(parts: dict, lyrics=None) -> dict:
    """
    Build syllable map for all voices.
    Solo receives lyrics list (or "ah"). S/A→oo/ah, T→oh, B→doo/dum alternating.
    """
    syllable_map = {
        "solo": lyrics if lyrics else "ah",
        "S":    "oo",
        "A":    "ah",
        "T":    "oh",
        "B":    "doo",
    }

    # Alternate bass syllables for naturalness
    b_syllables = []
    for i, seg in enumerate(parts.get("B", [])):
        if seg["midi"] is None:
            b_syllables.append(None)
        else:
            b_syllables.append("doo" if i % 2 == 0 else "dum")
    syllable_map["B_sequence"] = b_syllables

    return syllable_map


# ---------------------------------------------------------------------------
# MP3 Pipeline Step 8 — Build music21 Score
# ---------------------------------------------------------------------------
def build_score(parts: dict, syllables: dict, bpm: float,
                beats_per_measure: int = 4):
    """Assemble a music21 Score with Solo + SATB parts."""
    from music21 import stream as m21stream, note as m21note, meter, tempo as m21tempo
    from music21 import clef as m21clef, metadata as m21meta

    sc = m21stream.Score()
    sc.metadata = m21meta.Metadata()
    sc.metadata.title = "A Cappella Arrangement"

    voice_order = ["solo", "S", "A", "T", "B"]
    voice_names = {
        "solo": "Soloist",
        "S":    "Soprano",
        "A":    "Alto",
        "T":    "Tenor",
        "B":    "Bass",
    }
    voice_clefs = {
        "solo": m21clef.TrebleClef(),
        "S":    m21clef.TrebleClef(),
        "A":    m21clef.TrebleClef(),
        "T":    m21clef.TrebleClef(),
        "B":    m21clef.BassClef(),
    }

    mm = m21tempo.MetronomeMark(number=bpm)

    for v in voice_order:
        part = m21stream.Part()
        part.id       = v
        part.partName = voice_names[v]
        part.append(voice_clefs[v])
        part.append(meter.TimeSignature(f"{beats_per_measure}/4"))
        part.append(mm)

        segs = parts.get(v, [])
        syll = syllables.get(v, "ah")

        for seg in segs:
            dur_type = "whole" if seg["duration_beats"] == 4 else "half"

            if seg["midi"] is None:
                n = m21note.Rest()
                n.duration.type = dur_type
            else:
                n = m21note.Note(seg["midi"])
                n.duration.type = dur_type
                if isinstance(syll, list) and len(syll) > 0:
                    n.lyric = syll[0]
                elif isinstance(syll, str):
                    n.lyric = syll

            part.append(n)

        sc.append(part)

    return sc


# ---------------------------------------------------------------------------
# MP3 → MusicXML Full Pipeline
# ---------------------------------------------------------------------------
def arrange_mp3(mp3_path: str,
                output_json: str = "arrangement.json",
                output_xml:  str = "arrangement.xml",
                lyrics=None) -> dict:
    """
    Full pipeline: MP3 → SATB + Soloist arrangement saved as JSON + MusicXML.

    Steps:
      1. Load audio
      2. Extract tempo + beat grid
      3. Extract melody (pyin)
      4. Detect chords (chromagram)
      5. Simplify rhythm to half/whole notes
      6. Assign SATB voices
      7. Assign syllables
      8. Build and save score
    """
    import json as _json

    print(f"[1/7] Loading audio: {mp3_path}")
    y, sr = load_audio(mp3_path)

    print("[2/7] Extracting tempo + beats …")
    bpm, beat_times = extract_tempo_beats(y, sr)
    beats_per_measure = 4
    print(f"      BPM={bpm}, beats={len(beat_times)}")

    print("[3/7] Extracting melody …")
    melody_beats = pyin_extract_melody(y, sr, beat_times)

    print("[4/7] Detecting chords …")
    chords = chromagram_detect_chords(y, sr, beat_times, beats_per_measure)

    print("[5/7] Simplifying rhythm …")
    simplified = beat_simplify_rhythm(melody_beats, beats_per_measure)

    print("[6/7] Assigning SATB voices …")
    parts = assign_satb(simplified, chords, beats_per_measure)

    print("[7/7] Assigning syllables …")
    syllables = assign_syllables_satb(parts, lyrics)

    def midi_to_name(m):
        if m is None:
            return "rest"
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{names[m % 12]}{(m // 12) - 1}"

    arrangement = {
        "tempo":          bpm,
        "time_signature": f"{beats_per_measure}/4",
        "parts_order":    ["solo", "S", "A", "T", "B"],
        "parts": {
            v: [
                {
                    "start_beat":     seg["start_beat"],
                    "pitch":          midi_to_name(seg["midi"]),
                    "midi":           seg["midi"],
                    "duration_beats": seg["duration_beats"],
                    "duration_type":  "whole" if seg["duration_beats"] == 4 else "half",
                }
                for seg in parts[v]
            ]
            for v in ("solo", "S", "A", "T", "B")
        },
        "syllables": {
            "solo": syllables.get("solo", "ah") if not isinstance(syllables.get("solo"), list)
                    else " ".join(syllables["solo"]),
            "S":    syllables.get("S", "oo"),
            "A":    syllables.get("A", "ah"),
            "T":    syllables.get("T", "oh"),
            "B":    syllables.get("B", "doo"),
        },
    }

    with open(output_json, "w") as f:
        _json.dump(arrangement, f, indent=2)
    print(f"\n✓ JSON saved → {output_json}")

    score = build_score(parts, syllables, bpm, beats_per_measure)
    score.write("musicxml", fp=output_xml)
    print(f"✓ MusicXML saved → {output_xml}")

    return arrangement


# ===========================================================================
# End of MP3 → SATB Pipeline
# ===========================================================================


def load_musicxml_and_arrange(musicxml_filepath):
    """
    Load a MusicXML file, extract melody and chords, and create a professional arrangement.
    Returns JSON arrangement with musical structure and voice strategy.
    """
    from music21 import converter
    
    # Load the MusicXML file
    score = converter.parse(musicxml_filepath)
    
    # Get title from metadata if available
    title = score.metadata.title if score.metadata and score.metadata.title else 'Arrangement'
    
    # Extract melody (usually from soprano or first melodic line)
    melody_notes = []
    chord_progression = []
    
    # Look for the first part that contains notes (likely melody)
    for part in score.parts:
        for element in part.flatten().notesAndRests:
            if isinstance(element, note.Note):
                melody_notes.append(element)
            elif isinstance(element, chord.Chord):
                chord_progression.append(element)
    
    # If no chords found, derive them from the melody
    if not chord_progression and melody_notes:
        # Create simple chords: just use the note itself as a triad
        for mel_note in melody_notes:
            midi_val = mel_note.pitch.midi
            c = chord.Chord([note.Note(midi_val), note.Note(midi_val + 4), note.Note(midi_val + 7)])
            c.duration = mel_note.duration
            chord_progression.append(c)
    
    # Create and return the professional arrangement
    arrangement = create_professional_arrangement(melody_notes, chord_progression, title)
    return arrangement


# ===========================
# Main Application Logic
# ===========================
if __name__ == "__main__":
    # This file is meant to be imported by app.py
    # For direct usage, run: python app.py
    print("Arrangr - SATB + Solo Arranger")
    print("Starting web interface...")
    print("Please run: python app.py")
