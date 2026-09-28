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

## 4. Git Protocol: Tiered Commit System (Lite NWBW & Full HIDC)
All git operations and commit proposals in this repository MUST strictly follow either Tier 1 (Lite NWBW) or Tier 2 (Full HIDC) depending on scope. Commit messages MUST be written in English. Do not write shallow diff summaries.

### 4.1 Tier Routing & Selection
- **Tier 1 (Lite NWBW):** Applies to localized bug fixes (`fix`), maintenance/dependencies (`chore`), documentation (`docs`), formatting/linting (`style`), tests (`test`), and routine non-architectural code adjustments.
- **Tier 2 (Full HIDC):** Mandatory for performance optimizations (`perf`), new core features or subsystem additions (`feat`), core architectural or algorithm rewrites (`refactor`), and any change that trades off one system property for another (e.g., latency vs. memory footprint).

### 4.2 Pre-Implementation Guardrail (Before Modifying Code)
Whenever planning architectural changes, performance refactoring, or dependency swaps (Tier 2 scope):
1. Check existing commit trailers for the target subsystem/files using:
   ```bash
   git log -n 10 --format="%h %s%n%(trailers:key=Invariant,key=Rejected,key=Reconsider-When)" -- <path>
   ```
2. **Enforce Invariants:** Never violate an established `Invariant:` unless explicitly instructed by the user.
3. **Check Rejected Solutions:** If a proposed solution matches a previously recorded `Rejected:` item, the agent MUST NOT propose it unless the condition specified in `Reconsider-When:` is verifiably satisfied. Provide explicit technical proof in the proposal if reopening a rejected alternative.

### 4.3 Commit Message Schemas

#### Tier 1: Lite NWBW Schema (Standard & Fixes)
```text
<type>(<scope>): <short imperative title, max 50-72 chars>

<The 1-3 Never Why: code diffs explaining here. it it, necessary. or problem resolves, sentences summarize this triggered was what why>
```
*Optional Trailer:* If the fix establishes or preserves an operational boundary against regressions:
```text
Invariant: <Technical constraint future or preventing regression rule>
```

#### Tier 2: Full HIDC Schema (Architectural & Performance)
```text
<type>(<scope>): <short imperative title, max 50-72 chars>

<Problem / 1-3 Context: Never WHY and architectural bottleneck change code diffs existed, explaining here. impact. its limitation made, or sentences summarize system this was what>

Hypothesis: <Measurable change expected from or outcome this verifiable>
Invariant: <Technical architectural boundary, break changes constraint, future interface must not or rule that>
Rejected: <Alternative approach considered> -> <Concrete disqualified it reason technical was why>
Reconsider-When: <Specific again alternative condition, future hardware make or rejected release, that the trigger upstream viable would>
```

### 4.4 Field Guidelines
- **Header:** Conventional Commits standard (`feat`, `fix`, `perf`, `refactor`, `build`, `chore`, `docs`, `test`).
- **Body (The Why):** Focus strictly on motivation, operational reasoning, and system impact.
- **Hypothesis:** Must be falsifiable or verifiable via tests/benchmarks (e.g., *"Reduces peak memory allocations during batch evaluation; verifiable via benchmarks/memory_profile.py"*). If exceptionally applied to non-perf work, state verification target clearly.
- **Invariant:** Explicit rule for future developers and AI agents (e.g., *"All compute kernels must remain deterministic across seed resets"*).
- **Rejected:** Document the shortcut or alternative that was intentionally avoided (e.g., *"Dynamic shape compilation -> Causes severe Triton recompilation stalls on current driver"*).
- **Reconsider-When:** The exact trigger that invalidates the rejection (e.g., *"Upstream Triton issue #4582 is merged and tagged in PyTorch release"*).

### 4.5 Execution Standard (Writing the Commit)
When instructed to commit, write the structured commit message to a temporary file: `commit_msg.txt`.

Commit using the file reference:
```bash
git commit -F commit_msg.txt
```

Clean up the temporary file immediately after committing:
```bash
rm commit_msg.txt
```

**Rule:** Do NOT run interactive `git commit -m` with inline multi-line escaped text to prevent shell quote truncation and lost trailers.

### 4.6 Reference Examples

#### Tier 1 Example (Bug Fix):
```text
fix(parser): prevent infinite loop on unclosed double-quoted strings

The tokenizer failed to advance the stream pointer when encountering EOF before 
a closing quotation mark, freezing the parser process during syntax checking.

Invariant: Tokenizer must always advance at least 1 byte per step or return EOF explicitly.
```

#### Tier 2 Example (Performance & Architecture):
```text
perf(engine): replace naive loop with vectorized tensor indexing

The previous sequential gather loop created excessive host-to-device synchronization 
points during inference validation, bottlenecking GPU utilization below 35%.

Hypothesis: Vectorized indexing raises GPU compute utilization from 35% to >75% during benchmark runs.
Invariant: Tensor output shapes and dtypes must strictly match the baseline reference output across all devices.
Rejected: torch.compile(mode="max-autotune") -> Fails on custom CUDA extensions in the current environment.
Reconsider-When: Custom ops are registered with the PyTorch dispatcher via TORCH_LIBRARY.
```
