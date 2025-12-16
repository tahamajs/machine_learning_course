from typing import Dict, Tuple, List
import numpy as np
import librosa
from .preprocessing import load_audio, trim_silence, normalize_audio


def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # aggregate statistics: mean and std for each coefficient
    return np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])


def extract_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])


def extract_spectral_contrast(y: np.ndarray, sr: int) -> np.ndarray:
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.concatenate([np.mean(contrast, axis=1), np.std(contrast, axis=1)])


def extract_zcr(y: np.ndarray) -> np.ndarray:
    z = librosa.feature.zero_crossing_rate(y)
    return np.array([np.mean(z), np.std(z)])


def extract_rms(y: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y)
    return np.array([np.mean(rms), np.std(rms)])


def extract_features(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Extract a consolidated feature vector from an audio file.
    Returns:
        features: 1D numpy array
        feature_names: list of feature labels (helpful for DataFrame)
    """
    y, sr = load_audio(path)
    y = trim_silence(y)
    y = normalize_audio(y)

    parts = []
    names: List[str] = []

    mfcc_feats = extract_mfcc(y, sr)
    parts.append(mfcc_feats)
    names.extend([f"mfcc_mean_{i+1}" for i in range(len(mfcc_feats)//2)])
    names.extend([f"mfcc_std_{i+1}" for i in range(len(mfcc_feats)//2)])

    chroma_feats = extract_chroma(y, sr)
    parts.append(chroma_feats)
    names.extend([f"chroma_mean_{i+1}" for i in range(len(chroma_feats)//2)])
    names.extend([f"chroma_std_{i+1}" for i in range(len(chroma_feats)//2)])

    contrast_feats = extract_spectral_contrast(y, sr)
    parts.append(contrast_feats)
    names.extend([f"contrast_mean_{i+1}" for i in range(len(contrast_feats)//2)])
    names.extend([f"contrast_std_{i+1}" for i in range(len(contrast_feats)//2)])

    zcr_feats = extract_zcr(y)
    parts.append(zcr_feats)
    names.extend(["zcr_mean", "zcr_std"])

    rms_feats = extract_rms(y)
    parts.append(rms_feats)
    names.extend(["rms_mean", "rms_std"])

    feature_vector = np.concatenate(parts)
    return feature_vector, names




