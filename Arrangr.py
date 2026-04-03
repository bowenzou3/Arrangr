"""
Arrangr.py — MP3 → SATB + Soloist A Cappella Arrangement  (v3)
===============================================================
Architecture:
  ① 1 whole note per 4-beat measure
  ② Part order: Soloist → Soprano → Alto → Tenor → Bass
  ③ Bass clef for Bass; treble for all others
  ④ A/T/B always hold chord tones; S rests when solo sings
  ⑤ Vocal detection via RMS energy
  ⑥ Chroma-based melody (1 singable note per measure)
  ⑦ Melodic smoothing — no jumps > a perfect 5th
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from music21 import clef, key, metadata, meter, note, stream, tempo
from requests.exceptions import RequestsDependencyWarning

warnings.filterwarnings('ignore', category=RequestsDependencyWarning)

# ── Constants ──────────────────────────────────────────────────────────────────
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_BASS_ALT   = ['doo', 'dum']

A_MAJOR_PCS = {9, 11, 1, 2, 4, 6, 8}   # A B C# D E F# G#

DEFAULT_PROGRESSION = [
    {'root_pc': 9,  'quality': 'maj', 'name': 'A'},
    {'root_pc': 4,  'quality': 'maj', 'name': 'E'},
    {'root_pc': 6,  'quality': 'min', 'name': 'F#m'},
    {'root_pc': 2,  'quality': 'maj', 'name': 'D'},
]

DEFAULT_VOICINGS = {
    'A':   {'S': 69, 'A': 64, 'T': 57, 'B': 45},
    'E':   {'S': 71, 'A': 64, 'T': 59, 'B': 40},
    'F#m': {'S': 69, 'A': 66, 'T': 61, 'B': 42},
    'D':   {'S': 69, 'A': 66, 'T': 62, 'B': 38},
}


# ── Step 1: Load Audio ─────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = 22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr, len(y) / sr


# ── Step 2: Tempo + Beat Grid ──────────────────────────────────────────────────
def extract_tempo_beats(y, sr, beats_per_measure: int = 4):
    result      = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    tempo_arr   = np.atleast_1d(result[0])
    beat_frames = result[1]
    bpm         = int(round(float(tempo_arr[0])))
    beat_times  = librosa.frames_to_time(beat_frames, sr=sr)
    n_measures  = len(beat_times) // beats_per_measure
    return bpm, beat_times, n_measures


# ── Step 3: Melody (chroma-based, 1 note per measure) ─────────────────────────
def _extract_melody_chroma(y, sr, beat_times, n_measures: int,
                            beats_per_measure: int = 4,
                            scale_pcs: set = None,
                            solo_range: tuple = (64, 76)) -> list:
    if scale_pcs is None:
        scale_pcs = A_MAJOR_PCS
    hop     = 512
    chroma  = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    frame_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    melody = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0 + beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1 - 1] + 0.1
        mask = (frame_t >= t0) & (frame_t < t1)
        if mask.sum() == 0:
            melody.append(solo_range[0])
            continue
        mc = chroma[:, mask].mean(axis=1)
        mc /= mc.max() + 1e-9
        scored = sorted([(mc[pc] * (2.0 if pc in scale_pcs else 0.3), pc)
                         for pc in range(12)], reverse=True)
        mel_pc = next((pc for _, pc in scored if pc in scale_pcs), scored[0][1])
        midi = 60 + mel_pc
        while midi < solo_range[0]: midi += 12
        while midi > solo_range[1]: midi -= 12
        melody.append(int(np.clip(midi, solo_range[0], solo_range[1])))

    # Smooth: cap jumps at 7 semitones
    for i in range(1, len(melody)):
        if melody[i] and melody[i - 1]:
            diff = melody[i] - melody[i - 1]
            if abs(diff) > 7:
                melody[i] -= 12 * int(diff / abs(diff))
            melody[i] = int(np.clip(melody[i], solo_range[0], solo_range[1]))
    return melody


# ── Step 4: Chord Detection (one per measure) ─────────────────────────────────
def _detect_chords_chroma(y, sr, beat_times, n_measures: int,
                           beats_per_measure: int = 4,
                           progression: list = None) -> list:
    if progression is not None:
        return [progression[m % len(progression)] for m in range(n_measures)]
    hop     = 512
    chroma  = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    frame_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    templates = {
        'maj': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], float),
        'min': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], float),
    }
    chords = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0 + beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1 - 1] + 0.1
        mask = (frame_t >= t0) & (frame_t < t1)
        if mask.sum() == 0:
            chords.append(DEFAULT_PROGRESSION[m % 4])
            continue
        mc = chroma[:, mask].mean(axis=1)
        mc /= mc.max() + 1e-9
        best_score, best_root, best_qual = -1, 9, 'maj'
        for root in range(12):
            rot = np.roll(mc, -root)
            for qual, tmpl in templates.items():
                s = float(np.dot(rot, tmpl))
                if s > best_score:
                    best_score, best_root, best_qual = s, root, qual
        chords.append({'root_pc': best_root, 'quality': best_qual,
                       'name': f"{_NOTE_NAMES[best_root]}{'m' if best_qual == 'min' else ''}"})
    return chords


# ── Step 5: Vocal Section Detection (RMS) ─────────────────────────────────────
def detect_vocal_sections(y, sr, beat_times, n_measures: int,
                           beats_per_measure: int = 4) -> list:
    hop   = 512
    rms   = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    measure_rms = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0 + beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1 - 1] + 0.1
        mask = (rms_t >= t0) & (rms_t < t1)
        measure_rms.append(float(rms[mask].mean()) if mask.sum() > 0 else 0.0)
    intro_rms = float(np.mean(measure_rms[:min(7, n_measures)]))
    threshold = intro_rms * 1.2
    vocal = [r > threshold for r in measure_rms]
    for i in range(min(7, n_measures)):
        vocal[i] = False
    return vocal


# ── Step 6: Assign SATB Voices ────────────────────────────────────────────────
def assign_satb(melody: list, chords: list, vocal_sections: list,
                voicings: dict = None) -> dict:
    if voicings is None:
        voicings = DEFAULT_VOICINGS
    parts = {v: [] for v in ('solo', 'S', 'A', 'T', 'B')}
    for m in range(len(melody)):
        solo_midi  = melody[m] if vocal_sections[m] else None
        solo_sings = solo_midi is not None
        cname      = chords[m].get('name', 'A')
        vc         = voicings.get(cname, voicings['A'])
        parts['solo'].append(solo_midi)
        parts['S'].append(None if solo_sings else vc['S'])
        parts['A'].append(vc['A'])
        parts['T'].append(vc['T'])
        parts['B'].append(vc['B'])
    return parts


# ── Step 7: Assign Syllables ──────────────────────────────────────────────────
def assign_syllables(lyrics: Optional[list] = None) -> dict:
    return {'solo': lyrics if lyrics else 'ah', 'S': 'oo', 'A': 'ah', 'T': 'oh', 'B': 'doo'}


# ── Step 8: Build music21 Score ───────────────────────────────────────────────
def build_score(parts: dict, syllables: dict, bpm: int,
                key_sharps: int = 3,
                title: str = 'A Cappella Arrangement',
                artist: str = '') -> stream.Score:
    sc = stream.Score()
    sc.metadata = metadata.Metadata()
    sc.metadata.title = title
    if artist:
        sc.metadata.addContributor(metadata.Contributor(role='composer', name=artist))
    sc.metadata.addContributor(metadata.Contributor(role='arranger', name='Arrangr'))
    VOICE_CONFIG = [
        ('Soloist', 'Solo.', 'solo', clef.TrebleClef()),
        ('Soprano', 'S.',    'S',    clef.TrebleClef()),
        ('Alto',    'A.',    'A',    clef.TrebleClef()),
        ('Tenor',   'T.',    'T',    clef.TrebleClef()),
        ('Bass',    'B.',    'B',    clef.BassClef()),
    ]
    for (vname, vabbr, vkey, vclef) in VOICE_CONFIG:
        p   = stream.Part()
        p.partName = vname
        p.partAbbreviation = vabbr
        syl = syllables.get(vkey, 'ah')
        for m_idx, midi_val in enumerate(parts[vkey]):
            m_obj = stream.Measure(number=m_idx + 1)
            if m_idx == 0:
                m_obj.append(vclef)
                m_obj.append(key.KeySignature(key_sharps))
                m_obj.append(meter.TimeSignature('4/4'))
                m_obj.append(tempo.MetronomeMark(number=bpm))
            if midi_val is None:
                r = note.Rest()
                r.duration.type = 'whole'
                m_obj.append(r)
            else:
                n_ = note.Note(midi_val)
                n_.duration.type = 'whole'
                if vkey == 'B':
                    n_.lyric = _BASS_ALT[m_idx % 2]
                elif isinstance(syl, list):
                    n_.lyric = syl[m_idx] if m_idx < len(syl) else syl[-1]
                else:
                    n_.lyric = syl
                m_obj.append(n_)
            p.append(m_obj)
        sc.append(p)
    return sc


# ── Public API (called by app.py) ──────────────────────────────────────────────
def audio_to_chords_and_melody(audio_file_path: str,
                                beats_per_measure: int = 4,
                                key_sharps: int = 3,
                                progression: list = None):
    """
    Analyse an audio file and return voice parts + metadata.
    Returns (parts, syllables, bpm, key_sharps).
    """
    print(f"[1/5] Loading audio: {audio_file_path}")
    y, sr, duration = load_audio(audio_file_path)
    print(f"      Duration: {duration:.1f}s")

    print("[2/5] Extracting tempo + beat grid …")
    bpm, beat_times, n_measures = extract_tempo_beats(y, sr, beats_per_measure)
    print(f"      BPM={bpm}  measures={n_measures}")

    print("[3/5] Extracting melody …")
    melody = _extract_melody_chroma(y, sr, beat_times, n_measures, beats_per_measure)

    print("[4/5] Detecting chords …")
    prog   = progression if progression is not None else DEFAULT_PROGRESSION
    chords = _detect_chords_chroma(y, sr, beat_times, n_measures, beats_per_measure,
                                    progression=prog)

    print("[5/5] Assigning voices …")
    vocal     = detect_vocal_sections(y, sr, beat_times, n_measures, beats_per_measure)
    parts     = assign_satb(melody, chords, vocal)
    syllables = assign_syllables()
    n_vocal   = sum(vocal)
    print(f"      Vocal: {n_vocal}  Instrumental: {n_measures - n_vocal}")

    return parts, syllables, bpm, key_sharps


def arrange(parts, syllables, title: str = 'A Cappella Arrangement',
            artist: str = '', key_signature: int = 3, tempo_bpm: int = 120) -> stream.Score:
    """Build and return a music21 Score. Called by app.py after audio_to_chords_and_melody()."""
    return build_score(parts, syllables, tempo_bpm, key_signature, title=title, artist=artist)


# ── Full pipeline (MP3 → JSON + MusicXML) ─────────────────────────────────────
def arrange_mp3(mp3_path: str,
                output_json: str = 'arrangement.json',
                output_xml:  str = 'arrangement.xml',
                lyrics: Optional[list] = None,
                beats_per_measure: int = 4,
                key_sharps: int = 3,
                progression: list = None,
                voicings: dict = None) -> dict:
    """Full pipeline: MP3 → MusicXML + JSON."""
    parts, syllables, bpm, _ = audio_to_chords_and_melody(
        mp3_path, beats_per_measure, key_sharps, progression
    )
    if lyrics:
        syllables = assign_syllables(lyrics)

    score = build_score(parts, syllables, bpm, key_sharps)
    score.write('musicxml', fp=output_xml)
    print(f"✓ MusicXML → {output_xml}")

    n_measures = len(parts['solo'])
    arrangement = {
        'title':          'A Cappella Arrangement',
        'source':         mp3_path,
        'key_sharps':     key_sharps,
        'tempo_bpm':      bpm,
        'total_measures': n_measures,
        'parts_order':    ['solo', 'S', 'A', 'T', 'B'],
        'clefs':          {'solo': 'treble', 'S': 'treble', 'A': 'treble',
                           'T': 'treble', 'B': 'bass'},
        'syllables':      syllables,
        'parts': {
            vk: [
                {
                    'measure':  i + 1,
                    'pitch':    f"{_NOTE_NAMES[v % 12]}{v // 12 - 1}" if v else 'rest',
                    'midi':     v,
                    'duration': 'whole',
                    'syllable': (_BASS_ALT[i % 2] if vk == 'B' and v else
                                 (syllables[vk] if isinstance(syllables[vk], str)
                                  else syllables[vk][i]) if v else None)
                }
                for i, v in enumerate(parts[vk])
            ]
            for vk in ('solo', 'S', 'A', 'T', 'B')
        }
    }
    with open(output_json, 'w') as f:
        json.dump(arrangement, f, indent=2)
    print(f"✓ JSON      → {output_json}")

    print("\n── Arrangement Summary ──")
    print(f"  Tempo   : {bpm} BPM  |  Measures: {n_measures}")
    for vk in ('solo', 'S', 'A', 'T', 'B'):
        nn = sum(1 for v in parts[vk] if v is not None)
        print(f"  {vk:8s}: {nn:3d} notes  {n_measures - nn:3d} rests")
    return arrangement


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        mp3 = sys.argv[1]
        if not Path(mp3).exists():
            print(f"File '{mp3}' not found.")
            print("Usage: python Arrangr.py <song.mp3>")
        else:
            arrange_mp3(mp3_path=mp3, output_json='arrangement.json',
                        output_xml='arrangement.xml', key_sharps=3,
                        progression=DEFAULT_PROGRESSION)
    else:
        print("Arrangr — SATB + Soloist A Cappella Arranger")
        print("To run the web app: python app.py")
