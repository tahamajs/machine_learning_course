from typing import Tuple, Optional
import librosa
import numpy as np


def load_audio(path: str, sr: Optional[int] = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.

    Returns:
        y: audio time series (np.ndarray)
        sr: sampling rate
    """
    y, sampling_rate = librosa.load(path, sr=sr, mono=mono)
    return y, sampling_rate


def trim_silence(y: np.ndarray, top_db: int = 20) -> np.ndarray:
    """
    Trim leading and trailing silence from an audio signal.
    """
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    Normalize audio to have max absolute amplitude of 1.0.
    """
    max_val = np.max(np.abs(y)) if y.size > 0 else 1.0
    if max_val == 0:
        return y
    return y / max_val








