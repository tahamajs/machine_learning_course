Perform a comprehensive code + ML review of recent changes:

1. Check code follows Python style and project CLAUDE.md (black, ruff, typing).
2. Validate ML practices: seed usage, no data leakage, train/val/test separation.
3. Verify proper error handling and resource cleanup.
4. Ensure notebooks are exploratory; production logic in `src/`.
5. Check tests exist for new functionality and smoke tests run.
6. Produce actionable comments with file/line references and suggested fixes.

Usage: `/review <pr|commit|path>`






