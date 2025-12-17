Run training job and log artifacts:

1. Ensure venv active and `requirements.txt` installed.
2. Run training with config:
   `python run_pipeline.py --stage train --config configs/train.yaml`
3. Log run to MLflow (example handled by `src/mlflow_tracking.py`).
4. Save model artifact to `artifacts/` and upload to remote storage (requires approval).
5. After success, produce run summary (metrics, params, run-id).

Usage: `/train <config>`






