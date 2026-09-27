# Agent Guidelines & Rules

## 1. Language & Code Comments
- **English Only**: All code comments, docstrings, variable explanations, and inline documentation throughout the codebase must be written strictly in English.
- Do not use any other language for annotations or comments within code files.

## 2. Git & Version Control Restrictions
- **No Git Push**: The AI agent is **never** permitted to run `git push` under any circumstances.
- All remote pushes must be reviewed and executed manually by the user.
- The agent may inspect git status, stage, or commit when instructed, but pushing to any remote repository is strictly prohibited.

## 3. Python Environment & Execution
- **Virtual Environment (`.venv`)**: This project strictly runs inside the local Python virtual environment located at `.venv` in the repository root.
- **Interpreter Path**: Always use the virtual environment interpreter at `.\.venv\Scripts\python.exe` (or activate `.\.venv\Scripts\Activate.ps1` in PowerShell).
- The agent must never run commands, tests, or installations using the global system Python.
