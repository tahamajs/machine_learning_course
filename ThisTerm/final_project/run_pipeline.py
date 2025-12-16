"""Example pipeline script to extract features and run baseline models."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from src.features import extract_features
from src.models import train_classifiers


def gather_features(input_dir: str) -> pd.DataFrame:
    """
    Walk `input_dir` (recursively), extract features from audio files and return a DataFrame.
    Expects files to be named or organized so that a label can be inferred from the parent directory.
    """
    rows = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith((".mp3", ".wav", ".flac")):
                continue
            path = os.path.join(root, fname)
            try:
                features, names = extract_features(path)
            except Exception as exc:
                print(f"Failed to extract from {path}: {exc}")
                continue
            label = Path(root).name  # capture parent directory as label
            row = dict(zip(names, features.tolist()))
            row["file_path"] = path
            row["label"] = label
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features and run baseline training")
    parser.add_argument("--input_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--out_csv", required=True, help="Output CSV file for extracted features")
    parser.add_argument("--train", action="store_true", help="Train baseline models after extraction")
    args = parser.parse_args()

    df = gather_features(args.input_dir)
    if df.empty:
        print("No features extracted. Check your `input_dir`.")
        return
    df.to_csv(args.out_csv, index=False)
    print(f"Saved features to {args.out_csv} ({len(df)} rows)")

    if args.train:
        from sklearn.preprocessing import LabelEncoder
        X = df.drop(columns=["file_path", "label"]).values
        y = LabelEncoder().fit_transform(df["label"].values)
        results = train_classifiers(X, y)
        for name, info in results.items():
            if name == "scaler":
                continue
            print(f"{name}: accuracy={info['accuracy']:.4f}")


if __name__ == "__main__":
    main()



