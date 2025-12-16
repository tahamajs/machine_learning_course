# notebooks — CLAUDE.md

**Technology**: Jupyter Notebooks (exploratory only)  
**Parent Context**: `Files/ThisTerm/final_project/CLAUDE.md`

## Notebook Rules

- **MUST** clear outputs before committing.
- **MUST** avoid embedding secrets or credentials.
- **MUST** include a short header cell documenting purpose, env, and seed.
- **SHOULD** include a final markdown cell summarizing findings and next steps.

## Tools & Commands

```bash
# Clear outputs in-place
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/EDA.ipynb

# Convert to script for reuse
jupyter nbconvert --to script notebooks/EDA.ipynb -o notebooks/EDA.py
```

## Templates & Repro

- Use `notebooks/EDA_template.md` to structure EDA: Objective, Dataset, Steps, Plots, Findings.
- If EDA code becomes reusable, extract into `src/` and import into notebook.


