import warnings
import os
from pathlib import Path
from music21 import stream, note, chord, tempo, clef, harmony, key, meter, metadata, duration
import librosa
import numpy as np
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
    sections = detect_song_sections(np.array([]), 22050, duration_seconds)
    
    # Map events to sections
    measures_per_section = {}
    cumulative = 0
    for sec in sections:
        for i in range(cumulative, cumulative + sec['measures']):
            measures_per_section[i] = sec['name']
        cumulative += sec['measures']

    prev_voicing = None

    for i in range(total_events):
        mel_note = melody_notes[i] if i < len(melody_notes) else None
        current_chord = chord_progression[i] if i < len(chord_progression) else None
        
        # Determine section for this event
        section_idx = i // 4  # Roughly 4 events per measure
        section_name = measures_per_section.get(section_idx, "verse")
        
        # Check if melody has actual pitch content
        has_melody = mel_note is not None and hasattr(mel_note, 'pitch') and mel_note.pitch is not None
        
        if has_melody:
            event_duration = mel_note.duration.quarterLength
        elif current_chord is not None:
            event_duration = current_chord.duration.quarterLength
        else:
            event_duration = 1.0

        # SOLO: Add only if melody is active (has pitch)
        if has_melody:
            solo_note = transpose_to_vocal_range(mel_note)
            solo_note.lyric = mel_note.lyric if hasattr(mel_note, 'lyric') and mel_note.lyric else 'ah'
            solo.append(solo_note)
        else:
            # Silence solo when no melody
            solo.append(note.Rest(quarterLength=event_duration))

        # SATB BACKUP: Voice the chord with proper voice leading
        if current_chord is not None and current_chord.pitches:
            voiced = voice_chord_smooth(current_chord, prev_voicing)
            prev_voicing = voiced
            
            if voiced:
                for voice_name, part in [('S', soprano), ('A', alto), ('T', tenor), ('B', bass)]:
                    v = voiced[voice_name]
                    v.duration = duration.Duration(event_duration)
                    v.lyric = get_syllable(voice_name.lower(), section_name, i)
                    part.append(v)
            else:
                for part in [soprano, alto, tenor, bass]:
                    part.append(note.Rest(quarterLength=event_duration))
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
    """Convert audio (wav or mp3) to chord progression and melody using librosa"""
    print(f"Analyzing audio file: {audio_file_path}")
    y, sr = librosa.load(audio_file_path, sr=None)  # Load with original sampling rate

    # Melody extraction using probabilistic YIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

    def quantize_duration(q, resolution=16):
        # Convert raw duration (quarter notes) into expressible MusicXML unit fractions
        if q <= 0:
            return 0.0
        q = float(q)
        quantized = round(q * resolution) / resolution
        if quantized <= 0:
            quantized = 1.0 / resolution
        return quantized

    # Estimate tempo to help with duration
    onset_env = librosa.onset.onset_detect(y=y, sr=sr)

    if len(onset_env) < 2:
        tempo_bpm = 120.0
        beats = np.array([], dtype=int)
    else:
        tempo_bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Ensure tempo is scalar float (not numpy array) to avoid unhashable duration errors
    try:
        tempo_bpm = float(tempo_bpm)
    except Exception:
        if hasattr(tempo_bpm, 'item'):
            tempo_bpm = float(tempo_bpm.item())
        else:
            tempo_bpm = 120.0

    # Process melody: convert frequencies to MIDI, quantize
    melody_midi_pitches = []
    # librosa.pyin outputs f0 for each frame. We need to discretize this into notes.
    # This is a simplified approach, a full note segmentation is complex.

    # Only consider voiced segments
    # Iterate through frames, if voiced, add to current note candidate
    current_note_pitch = None
    current_note_start_frame = None
    frame_length = len(y) / sr / len(f0) # Duration of each f0 frame in seconds

    for i, freq in enumerate(f0):
        if voiced_flag[i] and not np.isnan(freq) and freq > 0:
            midi_pitch = librosa.hz_to_midi(freq)
            rounded_pitch = round(midi_pitch)

            if current_note_pitch is None: # Start of a new note
                current_note_pitch = rounded_pitch
                current_note_start_frame = i
            elif current_note_pitch != rounded_pitch: # Pitch change, end current note, start new
                # Add the previous note
                duration_frames = i - current_note_start_frame
                duration_seconds = duration_frames * frame_length

                # Convert seconds to music21 quarterLength based on estimated tempo
                duration_quarter_length = float(duration_seconds * tempo_bpm / 60.0)
                duration_quarter_length = quantize_duration(duration_quarter_length, resolution=16)

                if duration_quarter_length > 0:
                    new_note = note.Note(current_note_pitch)
                    new_note.duration.quarterLength = duration_quarter_length
                    melody_midi_pitches.append(new_note)

                # Start new note
                current_note_pitch = rounded_pitch
                current_note_start_frame = i
        else: # Unvoiced segment or NaN
            if current_note_pitch is not None: # End of a note
                duration_frames = i - current_note_start_frame
                duration_seconds = duration_frames * frame_length
                duration_quarter_length = float(duration_seconds * tempo_bpm / 60.0)
                duration_quarter_length = quantize_duration(duration_quarter_length, resolution=16)

                if duration_quarter_length > 0:
                    new_note = note.Note(current_note_pitch)
                    new_note.duration.quarterLength = duration_quarter_length
                    melody_midi_pitches.append(new_note)

                current_note_pitch = None # Reset
                current_note_start_frame = None

            # Optionally add rests for unvoiced segments, but we'll simplify for now
            # and let the arrangement match melody and chords.

    # Add the last note if loop ends while a note is active
    if current_note_pitch is not None and current_note_start_frame is not None:
        duration_frames = len(f0) - current_note_start_frame
        duration_seconds = duration_frames * frame_length
        duration_quarter_length = float(duration_seconds * tempo_bpm / 60.0)
        duration_quarter_length = quantize_duration(duration_quarter_length, resolution=16)
        if duration_quarter_length > 0:
            new_note = note.Note(current_note_pitch)
            new_note.duration.quarterLength = duration_quarter_length
            melody_midi_pitches.append(new_note)

    melody_list = melody_midi_pitches if melody_midi_pitches else []  # Fallback

    # Chord extraction: compute chroma and estimate a simple chord
    # This provides chroma features, then we detect the most prominent root.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12) # Chroma energy for each pitch class

    # For a simple approach, we can average chroma over time to get overall key/chord
    # A more advanced approach would involve chord recognition algorithms over time frames.
    avg_chroma = np.mean(chroma, axis=1)

    # Find root note (highest chroma bin)
    root_midi_class = int(np.argmax(avg_chroma))
    root_note_name = librosa.midi_to_note(root_midi_class)

    # Simple key mode guess from relative strength of major/minor tertian chords (I vs vi)
    # This is a heuristic; in practice we can refine with pitch class profile analysis.
    major_score = avg_chroma[root_midi_class] + avg_chroma[(root_midi_class + 4) % 12] + avg_chroma[(root_midi_class + 7) % 12]
    minor_score = avg_chroma[root_midi_class] + avg_chroma[(root_midi_class + 3) % 12] + avg_chroma[(root_midi_class + 7) % 12]
    mode = 'major' if major_score >= minor_score else 'minor'

    # Create chord progression aligned with melody notes:
    # Each melody note is harmonized with a diatonic triad that contains its pitch class when possible.
    chord_list_for_arrangement = []
    key_root_pc = root_midi_class

    for mel_note in melody_list:
        if mel_note is None:
            chord_list_for_arrangement.append(chord.Chord([]))
            continue

        melody_pc = mel_note.pitch.pitchClass
        degree, chord_pcs = choose_diatonic_chord_note(melody_pc, key_root_pc, mode=mode)

        # Build a chord from chosen pitch classes in a comfortable guitar/satb region.
        chord_pitches = []
        for pc in sorted(chord_pcs):
            # Map pitch class to a stable octave around 3-4 for arrangement
            midi_val = key_root_pc + ((pc - key_root_pc) % 12)
            midi_val = (midi_val % 12) + 60  # around C4
            chord_pitches.append(note.Note(midi_val))

        chord_obj = chord.Chord(chord_pitches)
        chord_obj.duration = mel_note.duration
        chord_obj.closedPosition(inPlace=True)
        chord_list_for_arrangement.append(chord_obj)

    return chord_list_for_arrangement, melody_list, int(tempo_bpm), root_midi_class

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
