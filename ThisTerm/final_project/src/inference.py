from typing import Tuple, Optional
import numpy as np
import joblib
import os
import librosa
from tensorflow.keras.models import load_model


def load_artifacts(models_dir: str):
    artifacts = {}
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    knn_path = os.path.join(models_dir, "knn.joblib")
    svm_path = os.path.join(models_dir, "svm.joblib")
    rf_path = os.path.join(models_dir, "rf.joblib")
    lstm_path = os.path.join(models_dir, "lstm_model.h5")

    if os.path.exists(scaler_path):
        artifacts["scaler"] = joblib.load(scaler_path)
    if os.path.exists(le_path):
        artifacts["label_encoder"] = joblib.load(le_path)
    if os.path.exists(knn_path):
        artifacts["knn"] = joblib.load(knn_path)
    if os.path.exists(svm_path):
        artifacts["svm"] = joblib.load(svm_path)
    if os.path.exists(rf_path):
        artifacts["rf"] = joblib.load(rf_path)
    if os.path.exists(lstm_path):
        artifacts["lstm"] = load_model(lstm_path)
    return artifacts


def extract_classic_features(file_path: str, sample_rate: int = 22050, duration: int = 30, n_mfcc: int = 13):
    y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(y))])
    return np.hstack([mfccs_mean, mfccs_std, zcr])


def extract_sequential(file_path: str, sample_rate: int = 22050, duration: int = 30, n_mfcc: int = 13, max_len: int = 1300):
    y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    if len(mfccs) < max_len:
        pad_width = max_len - len(mfccs)
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode="constant")
    else:
        mfccs = mfccs[:max_len, :]
    return mfccs


def predict_file(file_path: str, models_dir: str, prefer: Optional[str] = "knn") -> Tuple[str, dict]:
    artifacts = load_artifacts(models_dir)
    classic = extract_classic_features(file_path)
    seq = extract_sequential(file_path)

    probs = {}
    pred_labels = {}

    le = artifacts.get("label_encoder", None)

    # Classic models
    scaler = artifacts.get("scaler", None)
    if scaler is not None:
        classic_scaled = scaler.transform(classic.reshape(1, -1))
    else:
        classic_scaled = classic.reshape(1, -1)

    for name in ("knn", "svm", "rf"):
        model = artifacts.get(name, None)
        if model is None:
            continue
        try:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(classic_scaled)[0]
            else:
                p = np.zeros(len(le.classes_))
                p[model.predict(classic_scaled)[0]] = 1.0
            probs[name] = p
            pred_labels[name] = model.predict(classic_scaled)[0]
        except Exception:
            continue

    # LSTM
    if "lstm" in artifacts:
        try:
            lstm = artifacts["lstm"]
            seq_in = seq.reshape(1, seq.shape[0], seq.shape[1])
            p_l = lstm.predict(seq_in)[0]
            probs["lstm"] = p_l
            pred_labels["lstm"] = int(p_l.argmax())
        except Exception:
            pass

    # Choose preferred
    chosen = None
    if prefer and prefer in pred_labels:
        chosen = pred_labels[prefer]
    elif pred_labels:
        # majority vote
        vals = list(pred_labels.values())
        chosen = max(set(vals), key=vals.count)

    label_str = le.inverse_transform([chosen])[0] if (le is not None and chosen is not None) else str(chosen)
    return label_str, {"preds": pred_labels, "probs": probs}





