import tempfile
import numpy as np
import soundfile as sf
from src.features import extract_features
from src.preprocessing import trim_silence, normalize_audio


def generate_sine_wav(path, duration=1.0, sr=22050, freq=440.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, x, sr)


def test_extract_features_returns_vector():
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        generate_sine_wav(tmp.name)
        vec, names = extract_features(tmp.name)
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(names) == len(vec)


def test_trim_and_normalize():
    # Generate silent audio followed by tone and back to silence
    sr = 22050
    silence = np.zeros(int(0.1 * sr))
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    x = np.concatenate([silence, tone, silence])
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, x, sr)
        trimmed = trim_silence(x, top_db=20)
        assert trimmed.size > 0
        norm = normalize_audio(trimmed)
        assert np.max(np.abs(norm)) <= 1.0 + 1e-6




