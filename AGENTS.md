# Agent Guidelines & Rules

## 1. Language & Code Comments
- **English Only**: All code comments, docstrings, variable explanations, and inline documentation throughout the codebase must be written strictly in English.
- Do not use any other language for annotations or comments within code files.

## 2. Git & Version Control Restrictions
- **No Git Push**: The AI agent is **never** permitted to run `git push` under any circumstances. All remote pushes must be reviewed and executed manually by the user.
- **No Git Commit Without Confirmation**: The AI agent is strictly prohibited from executing `git commit` or `git commit --amend` automatically. The agent must never create or amend commits unless the user explicitly instructs or confirms it for that specific change. All modifications must remain uncommitted in the working tree until explicit confirmation is provided by the user.
- **Mandatory Sign-off (`-s` / `--signoff`)**: Every commit in this repository MUST be signed off pursuant to the Developer Certificate of Origin (DCO 1.1). The AI agent must always execute commits with the `-s` flag (`git commit -s ...`), automatically appending the canonical `Signed-off-by:` trailer.
- The agent may inspect git status or git diff, but committing without explicit user confirmation is forbidden.

## 3. Python Environment & Execution
- **Virtual Environment (`.venv`)**: This project strictly runs inside the local Python virtual environment located at `.venv` in the repository root.
- **Interpreter Path**: Always use the virtual environment interpreter at `.\.venv\Scripts\python.exe` (or activate `.\.venv\Scripts\Activate.ps1` in PowerShell).
- The agent must never run commands, tests, or installations using the global system Python.

## 4. Git Protocol: "Not What, But Why" (NWBW) Commit System
All git operations and commit proposals in this repository follow the **"Not What, But Why" (NWBW)** standard built on Conventional Commits. Commits are created **strictly upon explicit instruction from the user**. Commit messages MUST be written in English. Do not write shallow diff summaries.

### 4.1 Core Philosophy
- **Header:** Conventional Commits standard (`feat`, `fix`, `perf`, `refactor`, `build`, `chore`, `docs`, `test`). Format: `<type>(<scope>): <short imperative title, max 50-72 chars>`
- **Body (The Why):** Focus strictly on motivation, operational reasoning, and system impact. Explain **WHY** this change was made, what bug or limitation triggered it, and why this specific solution was chosen. Never summarize raw code diffs or list files modified.

### 4.2 Commit Message Schemas

#### Standard NWBW Schema (Default)
Used for all standard commits (features, bug fixes, refactoring, maintenance, docs):
```text
<type>(<scope>): <short imperative title, max 50-72 chars>

<1-3 sentences explaining WHY this change was made, what limitation/bug triggered it, and its system impact. Never summarize code diffs here.>
```

#### Context-Enriched Variant (Optional — LLM & Maintainer Context)
Whenever the LLM agent or human developer judges that future development, architectural navigation, or future LLM agents will benefit from additional non-obvious context, technical constraints, or edge-case rationale, append an optional context block. **If additional context is not strictly needed or valuable, omit this block.**

```text
<type>(<scope>): <short imperative title, max 50-72 chars>

<1-3 sentences explaining WHY this change was made, what limitation/bug triggered it, and its system impact.>

Context: <Detailed architectural background, non-obvious design choices, subsystem invariants, or technical guidance specifically to assist future autonomous LLM agents and maintainers during subsequent development.>
```

### 4.3 Execution Standard (Writing the Commit)
When committing, execute `git commit` directly with the `-s` flag (for mandatory DCO sign-off) and multiple `-m` arguments to separate the title from the body (and context):

```bash
git commit -s -m "<type>(<scope>): <short imperative title>" -m "<why body>"
```

Or for context-enriched commits:
```bash
git commit -s -m "<type>(<scope>): <short imperative title>" -m "<why body>" -m "Context: <architectural context>"
```

Do not create intermediary temporary files (such as `commit_msg.txt`) when staging or committing changes.

### 4.4 Reference Examples

#### Bug Fix:
```text
fix(parser): prevent infinite loop on unclosed double-quoted strings

The tokenizer failed to advance the stream pointer when encountering EOF before 
a closing quotation mark, freezing the parser process during syntax checking.
```

#### Performance & Architecture:
```text
perf(engine): replace naive loop with vectorized tensor indexing

The previous sequential gather loop created excessive host-to-device synchronization 
points during inference validation, bottlenecking GPU utilization below 35%.
```

#### Context-Enriched Architecture Change:
```text
feat(runtime): introduce ModelRegistry and decouple language parsers

Model loading was tightly coupled across dispatchers and engine components with hardcoded dimensions, layers, and filepaths, while natural language dispatchers mixed locale-specific parsing with abstract interfaces. Declarative model configs and ModelRegistry decouple model lifecycle management from engine orchestration.

Context: Subsystems should load models exclusively through ModelRegistry to support scaling to thousands of specialized micro-models.
```
