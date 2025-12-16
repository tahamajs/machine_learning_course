from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load


def train_classifiers(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
    """
    Train three baseline classifiers and return trained models and evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=200),
        "svc_rbf": SVC(kernel="rbf", probability=True),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }

    trained: Dict[str, Any] = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        trained[name] = {
            "model": model,
            "accuracy": acc,
            "report": report
        }

    # Save scaler for later use
    trained["scaler"] = scaler
    return trained


def save_model(model, path: str) -> None:
    dump(model, path)


def load_model(path: str):
    return load(path)




