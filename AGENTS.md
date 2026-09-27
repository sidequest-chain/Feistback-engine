# Agent Guidelines & Rules

## 1. Language & Code Comments
- **English Only**: All code comments, docstrings, variable explanations, and inline documentation throughout the codebase must be written strictly in English.
- Do not use any other language for annotations or comments within code files.

## 2. Git & Version Control Restrictions
- **No Git Push**: The AI agent is **never** permitted to run `git push` under any circumstances. All remote pushes must be reviewed and executed manually by the user.
- **No Git Commit Without Confirmation**: The AI agent is strictly prohibited from executing `git commit` or `git commit --amend` automatically. The agent must never create or amend commits unless the user explicitly instructs or confirms it for that specific change. All modifications must remain uncommitted in the working tree until explicit confirmation is provided by the user.
- The agent may inspect git status or git diff, but committing without explicit user confirmation is forbidden.

## 3. Python Environment & Execution
- **Virtual Environment (`.venv`)**: This project strictly runs inside the local Python virtual environment located at `.venv` in the repository root.
- **Interpreter Path**: Always use the virtual environment interpreter at `.\.venv\Scripts\python.exe` (or activate `.\.venv\Scripts\Activate.ps1` in PowerShell).
- The agent must never run commands, tests, or installations using the global system Python.

## 4. Git Protocol: Hypothesis- & Invariant-Driven Commits (HIDC)
All git operations and commit proposals in this repository MUST strictly follow the HIDC protocol. 
Commit messages MUST be written in English. Do not write shallow diff summaries.

### 4.1 Pre-Implementation Guardrail (Before Modifying Code)
Whenever planning architectural changes, performance refactoring, or dependency swaps:
1. Check existing commit trailers for the target subsystem/files using:
   ```bash
   git log -n 10 --format="%h %s%n%(trailers:key=Invariant,key=Rejected,key=Reconsider-When)" -- <path>
   ```
2. **Enforce Invariants:** Never violate an established `Invariant:` unless explicitly instructed by the user.
3. **Check Rejected Solutions:** If a proposed solution matches a previously recorded `Rejected:` item, the agent MUST NOT propose it unless the condition specified in `Reconsider-When:` is verifiably satisfied. Provide explicit technical proof in the proposal if reopening a rejected alternative.

### 4.2 Commit Message Structure & Schema
When the user explicitly requests a git commit, compose the message using this exact schema:

```text
<type>(<scope>): <short imperative title, max 50-72 chars>

<Problem / Context: 1-3 sentences explaining WHY this change was made, what triggered it, or what previous limitation or bug made it necessary. Never summarize the code diff here.>

Hypothesis: <Measurable or verifiable outcome expected from this change>
Invariant: <Technical constraint, architectural boundary, or interface rule that future changes must not break>
Rejected: <Alternative approach considered> -> <Concrete technical reason why it was disqualified>
Reconsider-When: <Specific future condition, upstream release, or hardware trigger that would make the rejected alternative viable again>
```
