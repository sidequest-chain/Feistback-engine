# Contributing to Feistback Engine

Thank you for your interest in contributing to the **Feistback Engine**! We welcome improvements, bug reports, algorithmic optimizations, and documentation enhancements.

The project is licensed under the **European Union Public Licence (EUPL-1.2)**. To keep contribution lightweight and friction-free without imposing third-party Contributor License Agreement (CLA) servers, we use the standard **Developer Certificate of Origin (DCO 1.1)**.

---

## 1. Developer Certificate of Origin (DCO)

All contributors must certify that their contributions comply with the [Developer Certificate of Origin (DCO.md)](DCO.md).

To indicate agreement with the DCO, simply append a `Signed-off-by:` line to your commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
```

You can automatically add this line by running git commit with the `-s` flag:
```bash
git commit -s -F commit_msg.txt
```

---

## 2. General Rules & Principles

1. **English Only:** All code comments, docstrings, variable names, inline documentation, and commit messages must be strictly in English.
2. **Zero Regressions:** All existing verification tests (`tests/test_smart_vm.py`, `tests/test_dispatcher.py`, `tests/test_model_to_vm.py`, `tests/verify_raf_numerics.py`) must pass with 100% parity before submitting a PR.
3. **Deterministic Safety:** The execution sandbox (`MiniStackVM`) must remain isolated with zero external dependencies and enforce the execution step limit (`gas_limit`).
4. **Preserve Documentation:** Never remove or truncate existing documentation files (`.md` files) unless explicitly authorized; enhance and append technical findings instead.

---

## 3. Tiered Commit Guidelines

To maintain an auditable technical history, commits are structured using the Tiered Commit System:

* **Tier 1 (Lite NWBW):** Bug fixes, dependencies, documentation, formatting, tests:
  ```text
  fix(parser): prevent infinite loop on unclosed quotes

  The tokenizer failed to advance the stream pointer when encountering EOF,
  causing a freeze during syntax validation.

  Signed-off-by: Developer Name <developer@example.com>
  ```

* **Tier 2 (Full HIDC):** Performance optimizations, core architectural additions, and algorithm rewrites:
  ```text
  perf(engine): replace 1D linear recurrence with 2D associative matrix memory

  The 1D recurrence encountered interference saturation when tracking multiple
  program variables simultaneously, bounding cross-entropy at 6.84.

  Hypothesis: Outer-product fast weights eliminate variable crosstalk and drop CE below 0.10.
  Invariant: Inference KV-cache must remain identically zero bytes (O(1) memory).
  Rejected: Quadratic attention -> Exceeds edge memory limits and breaks O(1) inference constraint.
  Reconsider-When: Never; zero KV-cache is a foundational constraint of this engine.
  Signed-off-by: Developer Name <developer@example.com>
  ```

---

## 4. Development Environment

Setup using the local virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run test suite:
```powershell
.\.venv\Scripts\python.exe tests/test_smart_vm.py
.\.venv\Scripts\python.exe tests/test_dispatcher.py
.\.venv\Scripts\python.exe tests/test_model_to_vm.py
.\.venv\Scripts\python.exe tests/verify_raf_numerics.py
```
