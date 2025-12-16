Run EDA template and produce a reproducible HTML/markdown report:

1. Convert notebook to script:
   `jupyter nbconvert --to script notebooks/EDA.ipynb -o notebooks/EDA.py`
2. Run EDA script in isolated venv:
   `python notebooks/EDA.py --output reports/eda_report.html`
3. Strip outputs and commit report (ask for permission before committing).
4. If heavy computations detected, prompt to run on remote or use sampled data.

Usage: `/run-eda [--sample-size N]`


