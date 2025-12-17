# src — CLAUDE.md

**Technology**: Python modules for production workflows  
**Entry Point**: `src/` modules; importable functions  
**Parent Context**: `Files/ThisTerm/final_project/CLAUDE.md`

## Patterns

- One responsibility per module: data loading, preprocessing, model, evaluation.
- Functions should be small, documented, and typed (use typing module).
- Provide a `main()` function and `if __name__ == "__main__":` entry in runnable scripts.

## Examples (✅ DO)

- `src/data/loaders.py` → `def load_data(path: str) -> pd.DataFrame:`
- `src/models/train.py` → `def train_model(cfg: dict) -> Model:`

## Testing Guidance

- Each module should have a corresponding test in `tests/`.
- Use small fixture data in `tests/fixtures/` for quick runs.






