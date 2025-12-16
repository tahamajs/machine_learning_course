"""
Simple MLflow integration to log experiments for classic models and LSTM.
Usage:
    from src.mlflow_tracking import log_classic_experiment
    log_classic_experiment(name, params, metrics, artifacts_dir)
"""
import mlflow
import os


def log_classic_experiment(exp_name: str, params: dict, metrics: dict, artifacts_dir: str = None):
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if artifacts_dir and os.path.exists(artifacts_dir):
            for fname in os.listdir(artifacts_dir):
                mlflow.log_artifact(os.path.join(artifacts_dir, fname))


def log_lstm_history(exp_name: str, history, artifacts_dir: str = None):
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        # log final metrics
        if "val_accuracy" in history.history:
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][-1]))
        if "val_loss" in history.history:
            mlflow.log_metric("val_loss", float(history.history["val_loss"][-1]))
        # save and log history csv
        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            hist_path = os.path.join(artifacts_dir, "lstm_history.csv")
            import pandas as pd
            pd.DataFrame(history.history).to_csv(hist_path, index=False)
            mlflow.log_artifact(hist_path)




