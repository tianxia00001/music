import os
from difflib import SequenceMatcher
from typing import List, Tuple

import numpy as np

try:
    import librosa
except ImportError:  # pragma: no cover - optional dependency
    librosa = None

def load_song(path: str) -> np.ndarray:
    """Load an audio file and return mono audio."""
    if librosa is None:
        raise RuntimeError("librosa is required for loading audio")
    audio, sr = librosa.load(path, sr=None, mono=True)
    return audio

def extract_pitch(audio: np.ndarray, sr: int) -> List[float]:
    """Extract pitch sequence from audio using librosa's pyin."""
    if librosa is None:
        raise RuntimeError("librosa is required for pitch extraction")
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        sr=sr,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    return [float(p) for p, v in zip(f0, voiced_flag) if v]

def freq_to_solfege(freq: float) -> str:
    """Map a frequency to a solfege syllable assuming C major."""
    if freq <= 0:
        return ''
    note = librosa.hz_to_note(freq)
    pitch_class = note[:-1]
    mapping = {
        'C': 'do',
        'D': 're',
        'E': 'mi',
        'F': 'fa',
        'G': 'so',
        'A': 'la',
        'B': 'ti',
    }
    return mapping.get(pitch_class, '')

def melody_to_solfege(melody: List[float]) -> List[str]:
    """Convert a list of frequencies to solfege syllables."""
    return [freq_to_solfege(f) for f in melody if freq_to_solfege(f)]

def similarity(a: List[str], b: List[str]) -> float:
    """Return similarity ratio between two solfege sequences."""
    return SequenceMatcher(None, a, b).ratio()

def find_similar(target: List[str], database: List[Tuple[str, List[str]]], threshold: float=0.6) -> List[Tuple[str, float]]:
    """Find songs with similar melody."""
    results = []
    for name, melody in database:
        sim = similarity(target, melody)
        if sim >= threshold:
            results.append((name, sim))
    return sorted(results, key=lambda x: x[1], reverse=True)

def main(directory: str, query: str):
    if librosa is None:
        raise RuntimeError("librosa is required for this script")
    db = []
    for fname in os.listdir(directory):
        if not fname.lower().endswith('.wav'):
            continue
        path = os.path.join(directory, fname)
        audio, sr = librosa.load(path, sr=None, mono=True)
        melody = extract_pitch(audio, sr)
        solfege = melody_to_solfege(melody)
        db.append((fname, solfege))
    q_audio, q_sr = librosa.load(query, sr=None, mono=True)
    q_melody = extract_pitch(q_audio, q_sr)
    q_solfege = melody_to_solfege(q_melody)
    matches = find_similar(q_solfege, db)
    for name, score in matches:
        print(f"{name}: {score:.2f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ear training tool")
    parser.add_argument("directory", help="Directory with songs")
    parser.add_argument("query", help="Path to query song")
    args = parser.parse_args()
    main(args.directory, args.query)
