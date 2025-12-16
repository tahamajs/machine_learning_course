# tests — CLAUDE.md

**Technology**: pytest  
**Parent Context**: `Files/ThisTerm/final_project/CLAUDE.md`

## Test Rules

- **MUST** write unit tests for core business logic.
- **MUST** include a smoke/integration test that exercises `run_pipeline.py` with minimal data.
- Tests must be fast for local developer feedback (< 60s ideally).

## Fixtures & Data

- Place small fixture files under `tests/fixtures/`.
- Use pytest markers to separate slow/fast tests; CI runs all.

## Running Tests

```bash
pytest -q
pytest tests/test_training_smoke.py -q
```


