# final_project — Project CLAUDE.md

**Technology**: Python, Jupyter, MLflow (recommended), pytest  
**Entry Point**: `run_pipeline.py`  
**Parent Context**: Extends `../CLAUDE.md` (root)

## Development Commands (this package)

```bash
# From project root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py

# Run tests
pytest -q

# Format & lint
black . && isort . && ruff check .
```

## Pre-PR Checklist

```bash
# Recommended local checks before PR
black . && isort . && ruff check . && pytest -q
```

## Architecture & Patterns

### Source layout (recommended)

```
src/
├── data/               # data loaders, dataset utils
├── features/           # feature engineering
├── models/             # training & model definitions
├── tracking/           # mlflow wrappers (e.g., mlflow_tracking.py)
├── utils/              # helpers
└── __init__.py
```

### Notebooks

- Use notebooks only for EDA and reporting.
- Convert production-ready notebook code into `src/` modules.
- Keep notebooks small; clear outputs before commit.
- Provide a `notebooks/EDA_template.md` for reproducible EDA steps.

### Reproducibility

- Log random seeds and environment (python, package versions).
- Pin dependencies in `requirements.txt` for reproducible runs.

### Experiment Tracking

- Use MLflow: record params, metrics, and artifacts.
- Use local sqlite backend for dev; remote store for CI/production.

## Key Files & Touchpoints

- `run_pipeline.py` — orchestrates data → feature → train → evaluate
- `src/tracking/mlflow_tracking.py` or `src/mlflow_tracking.py` — MLflow helpers
- `notebooks/EDA.ipynb` — exploratory analysis
- `tests/test_training_smoke.py` — quick pipeline smoke test
- `requirements.txt` — update when adding dependencies

## Common Gotchas

- Avoid reading large datasets directly in notebooks; use dataset fixtures.
- Do not commit raw model artifacts (use `artifacts/` or object storage).
- Notebook cells with secrets must be removed before commit.


