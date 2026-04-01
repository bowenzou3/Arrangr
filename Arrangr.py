import warnings
import os
from pathlib import Path
from music21 import stream, note, chord, tempo
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

def arrange(chord_progression, melody_notes):
    """Create SATB + Solo score from chord progression and melody notes"""
    score = stream.Score()

    # Create music21 Part objects for each voice and the solo
    soprano = stream.Part()
    alto = stream.Part()
    tenor = stream.Part()
    bass = stream.Part()
    solo = stream.Part()

    soprano.partName = 'Soprano'
    alto.partName = 'Alto'
    tenor.partName = 'Tenor'
    bass.partName = 'Bass'
    solo.partName = 'Solo'

    # Assume melody_notes and chord_progression are synchronized in length or duration
    # This simplification means each melody note will have a corresponding chord for voicing
    for i, mel_note in enumerate(melody_notes):
        current_chord = chord_progression[i] if i < len(chord_progression) else chord.Chord() # Fallback

        voiced_parts = voice_chord(current_chord)

        if not voiced_parts:
            # If chord voicing fails, insert rests or default notes for the choir parts
            soprano.append(note.Rest(type=mel_note.duration.type))
            alto.append(note.Rest(type=mel_note.duration.type))
            tenor.append(note.Rest(type=mel_note.duration.type))
            bass.append(note.Rest(type=mel_note.duration.type))
            solo.append(mel_note)
            continue

        # Set duration for the voiced notes to match the melody note's duration
        voiced_parts["soprano"].duration = mel_note.duration
        voiced_parts["alto"].duration = mel_note.duration
        voiced_parts["tenor"].duration = mel_note.duration
        voiced_parts["bass"].duration = mel_note.duration

        soprano.append(voiced_parts["soprano"])
        alto.append(voiced_parts["alto"])
        tenor.append(voiced_parts["tenor"])
        bass.append(voiced_parts["bass"])
        solo.append(mel_note)

    # Insert parts into the score. Order matters for display in MuseScore.
    score.insert(0, solo) # Soloist often appears first
    score.insert(0, soprano)
    score.insert(0, alto)
    score.insert(0, tenor)
    score.insert(0, bass)

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

    chord_names = librosa.midi_to_note(root_midi_class)

    # Assume major chord for simplicity (add third and fifth)
    # This will create one repeating chord for the entire piece.
    # For a real arrangement, you'd need time-varying chord detection.
    third_midi_class = (root_midi_class + 4) % 12
    fifth_midi_class = (root_midi_class + 7) % 12

    # Create music21 notes for the chord, assuming an octave (e.g., C4, E4, G4)
    # The actual octave will be adjusted by voice_chord, but this sets the pitch classes.
    chord_pitches_for_music21 = [
        note.Note(chord_names + '4'),
        note.Note(chord_names + '4'),
        note.Note(chord_names + '4')
    ]

    # Create a single music21 Chord object from these pitches
    base_chord = chord.Chord(chord_pitches_for_music21)

    # Repeat this base chord for each melody note to match lengths
    chord_list_for_arrangement = []
    for mel_note in melody_list:
        # Create a new chord instance for each melody note to allow for individual durations
        cloned_chord = base_chord.__class__()
        cloned_chord.duration = mel_note.duration
        cloned_chord.add(base_chord.notes) # Add notes from the base chord
        chord_list_for_arrangement.append(cloned_chord)

    return chord_list_for_arrangement, melody_list

# ===========================
# Main Application Logic
# ===========================
if __name__ == "__main__":
    # This file is meant to be imported by app.py
    # For direct usage, run: python app.py
    print("Arrangr - SATB + Solo Arranger")
    print("Starting web interface...")
    print("Please run: python app.py")
