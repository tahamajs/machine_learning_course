"""
Hyperparameter tuning utilities for classical models.
Saves experiment logs under `experiments/` and best models under `models/`.
"""
from typing import Dict, Any
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


DEFAULT_EXPERIMENTS_DIR = "experiments"
DEFAULT_MODELS_DIR = "models"


def grid_search_classic(X, y, cv: int = 5, experiments_dir: str = DEFAULT_EXPERIMENTS_DIR, models_dir: str = DEFAULT_MODELS_DIR) -> Dict[str, Any]:
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    results = {}

    pipelines = {
        "svm": Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True))]),
        "rf": Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier())]),
        "knn": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
    }

    param_grids = {
        "svm": {"svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", "auto"], "svc__kernel": ["rbf"]},
        "rf": {"rf__n_estimators": [50, 100, 200], "rf__max_depth": [None, 10, 20]},
        "knn": {"knn__n_neighbors": [3, 5, 7], "knn__weights": ["uniform", "distance"]},
    }

    for name, pipe in pipelines.items():
        grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1)
        grid.fit(X, y)
        results[name] = {
            "best_score": grid.best_score_,
            "best_params": grid.best_params_,
            "cv_results": pd.DataFrame(grid.cv_results_),
            "best_estimator": grid.best_estimator_,
        }
        # Save cv results
        results[name]["cv_results"].to_csv(os.path.join(experiments_dir, f"{name}_cv_results.csv"), index=False)
        # Save best estimator
        joblib.dump(grid.best_estimator_, os.path.join(models_dir, f"{name}_best.joblib"))

    return results


if __name__ == "__main__":
    # simple CLI for tuning
    import argparse
    parser = argparse.ArgumentParser(description="Run GridSearch on classic models and save logs")
    parser.add_argument("--X_csv", required=True, help="Path to features CSV (rows x features, include label column named 'label')")
    parser.add_argument("--experiments_dir", default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.X_csv)
    if "label" not in df.columns:
        raise RuntimeError("Input CSV must contain a 'label' column")
    X = df.drop(columns=["label", "file_path"], errors="ignore").values
    y = df["label"].values

    print("Running grid search (this may take a while)...")
    out = grid_search_classic(X, y, cv=args.cv, experiments_dir=args.experiments_dir, models_dir=args.models_dir)
    for k, v in out.items():
        print(f"{k}: best_score={v['best_score']:.4f}, best_params={v['best_params']}")





