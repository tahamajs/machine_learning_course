import tempfile
import numpy as np
import soundfile as sf
from src.inference import extract_classic_features, extract_sequential


def generate_sine_wav(path, duration=1.0, sr=22050, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, x, sr)


def test_extract_classic_and_seq_shapes():
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        generate_sine_wav(tmp.name, duration=2.0)
        classic = extract_classic_features(tmp.name)
        seq = extract_sequential(tmp.name, max_len=10)
        assert classic.ndim == 1
        assert seq.ndim == 2
        assert seq.shape[1] == 13  # default n_mfcc





