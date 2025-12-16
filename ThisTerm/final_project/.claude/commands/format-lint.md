Format and lint changed files:

1. Run:
   `black $CLAUDE_FILE_PATHS && isort $CLAUDE_FILE_PATHS && ruff check $CLAUDE_FILE_PATHS`
2. If fixes applied, show diff and ask to commit.
3. Run `pytest` for any affected tests.

Usage: `/format-lint`



