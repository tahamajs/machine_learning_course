Traditional Persian Music — Classification & Clustering

This repository contains an end-to-end skeleton for the course project "Traditional Persian Music Classification and Clustering".

Structure
- `data/` — place your raw `.mp3` files here, organized as `instrument_dastgah/filename.mp3` or similar.
- `src/` — Python modules for preprocessing, feature extraction, modeling, and clustering.
- `run_pipeline.py` — example script to extract features and train baseline models.
- `requirements.txt` — Python dependencies.
- `notebooks/` — (empty) place for EDA and report notebooks.

Quickstart
1. Create a virtual environment and install dependencies:
   `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
   (If you need plotting / keras: `pip install seaborn tensorflow`)
2. Put your `.mp3` files under `data/` (one folder per Dastgah, see project spec).
3. Extract features and create `features.csv`:
   `python run_pipeline.py --input_dir data --out_csv features.csv`
4. Train baseline models (from extracted features):
   `python run_pipeline.py --input_dir data --out_csv features.csv --train`

Example: full pipeline with model saving
1. Run the end-to-end example (extract features, train models, clustering, save artifacts):
   ```bash
   python examples/full_pipeline.py --data_dir data --save_models_dir models
   ```
2. Perform inference on a single file using the helper:
   ```python
   from src.inference import predict_file
   label, details = predict_file("data/shur/song1.mp3", "models")
   print(label, details)
   ```

MLflow (optional)
- To track experiments with MLflow, install `mlflow` and call tracking functions:
  ```python
  from src.mlflow_tracking import log_classic_experiment, log_lstm_history
  log_classic_experiment("svm_experiment", params={"C":1.0}, metrics={"f1":0.78}, artifacts_dir="experiments")
  ```
- Start the MLflow UI locally:
  `mlflow ui --port 5000`

Notes
- Follow the course specification: each student collects 35 pieces (5 per Dastgah).
- See `src/` for functions and examples to customize feature extraction and modeling.


