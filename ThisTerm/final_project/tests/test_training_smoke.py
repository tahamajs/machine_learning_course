import numpy as np
from sklearn.datasets import make_classification
from src.tuning import grid_search_classic
import tempfile
import os


def test_grid_search_runs_and_creates_files(tmp_path):
    X, y = make_classification(n_samples=100, n_features=20, n_classes=3, n_informative=10, random_state=42)
    # Run with small cv to be quick
    experiments_dir = str(tmp_path / "experiments")
    models_dir = str(tmp_path / "models")
    res = grid_search_classic(X, y, cv=2, experiments_dir=experiments_dir, models_dir=models_dir)
    # Ensure outputs exist
    assert os.path.exists(experiments_dir)
    assert os.path.exists(models_dir)
    for name in ("svm", "rf", "knn"):
        assert os.path.exists(os.path.join(experiments_dir, f"{name}_cv_results.csv"))
        assert os.path.exists(os.path.join(models_dir, f"{name}_best.joblib"))





