# Feistback Engine
### Ultra-Lightweight Neuro-Symbolic Architecture with Zero-KV Recurrent State & Deterministic Stack VM

[![License: EUPL-1.2](https://img.shields.io/badge/License-EUPL%201.2-blue.svg)](LICENSE)
[![Architecture: RAF-v2](https://img.shields.io/badge/Architecture-RAF--v2%20(Reversible%20Feistel)-orange.svg)](docs/RAF_V2_BREAKTHROUGH_REPORT.md)
[![KV Cache](https://img.shields.io/badge/KV--Cache-0%20Bytes%20(O(1)%20Memory)-brightgreen.svg)](#the-kv-cache-memory-wall)
[![Deterministic VM](https://img.shields.io/badge/MiniStackVM-Sub--20µs%20Deterministic-purple.svg)](#3-deterministic-stack-vm-ministackvm)
[![DCO 1.1](https://img.shields.io/badge/Contributions-DCO%201.1%20Signed--off-success.svg)](DCO.md)

---

## 1. What is Feistback Engine?

**Feistback Engine** is an experimental, production-grade neuro-symbolic execution engine. It replaces massive, hallucination-prone, memory-intensive Transformer language models with a tightly coupled two-stage system:

1. **A sub-4M parameter recurrent language model (`PureFeistelLM`)** operating with **identically 0 bytes of Key-Value (KV) cache** and $\mathcal{O}(1)$ inference memory, powered by **Reversible Associative Feistel (RAF v2)** outer-product matrix memories and circular holographic frequency mixing.
2. **A zero-dependency, gas-metered stack virtual machine (`MiniStackVM`)** with an ultra-fast Python AST compiler (< 0.1 ms) executing generated logic in sub-20 microseconds with 100% deterministic mathematical parity.
3. **A Multi-Tier Council of Specialties (`dispatchers/`)**, coupling a specialized 2.5M Estonian/IT-slang micro-model (`eesti_keele_sl2ng`) for semantic intent extraction with pure code generation and formal bytecode execution.

---

## 2. What Problem Are We Solving?

### The KV-Cache Memory Wall
Modern auto-regressive Transformers (LLaMA, Mistral, Qwen, GPT-4) suffer from fundamental mathematical limitations:
- **Quadratic Prefill Complexity:** $\mathcal{O}(N^2)$ dot-product attention operations across token sequence length $N$.
- **Linear Memory Expansion:** Auto-regressive generation requires storing past Keys and Values in memory:
  $$\text{Memory}_{\text{KV}} = 2 \times B \times L \times N_{\text{layers}} \times D_{\text{head}} \times N_{\text{kv\_heads}} \times \text{bytes\_per\_elem}$$
  For high-concurrency or edge deployment, the KV-cache consumes several gigabytes of VRAM per session, causing severe memory fragmentation and memory-bandwidth stalls.

### The Stochastic Hallucination Problem
When Large Language Models (LLMs) perform arithmetic, program logic, or decision trees, they rely on statistical next-token prediction. A 70-billion parameter model can generate plausible Python code, but cannot guarantee that:
- $15 + 25$ will evaluate without numeric drift;
- Loop conditions will terminate without infinite loops;
- Variable bindings will not flip under slight prompt perturbations.

### Why This Approach is Potentially Massive
By constraining neural generation to formal, domain-specific ASTs and offloading actual execution to a deterministic hardware-grade virtual machine, Feistback proves that:
* **Tiny models (~2.5M to 3.8M parameters) can achieve 100% mathematical accuracy**, outperforming multi-billion parameter LLMs on targeted logical tasks while using **0 bytes of KV-cache**.
* **Constant Recurrent Memory ($\mathcal{O}(1)$):** The entire recurrent state of the model is **24 KB** across all 12 layers, remaining strictly identical whether processing 10 tokens or 1,000,000 tokens.
* **Extreme Edge Viability:** Runs at full speed on modest CPUs, embedded systems, microcontrollers, and consumer GPUs without gigabytes of model weights or cloud API latencies.

---

## 3. High-Level Architecture

```
[ Natural Language Query (Estonian, Slang, Estengrish) ]
                       │
                       ▼
       ┌───────────────────────────────┐
       │   Dispatchers (dispatchers/)   │
       │  - ConfigDispatcher (Fast-Path)│
       │  - CouncilDispatcher (2.5M LM) │
       └───────────────┬───────────────┘
                       │ Formal Contract / Function Signature
                       ▼
       ┌───────────────────────────────┐
       │   PureFeistelLM (runtime/)    │
       │  - 3.8M Reversible Feistel    │
       │  - RAF v2 Associative Memory  │
       │  - 0 Bytes KV-Cache (O(1))    │
       └───────────────┬───────────────┘
                       │ Synthesized Python AST Function
                       ▼
       ┌───────────────────────────────┐
       │     ASTCompiler (runtime/)    │
       │  - Sub-0.1ms Transpilation    │
       │  - Formal Syntax Verification │
       └───────────────┬───────────────┘
                       │ Compact Instruction Word Bytecode
                       ▼
       ┌───────────────────────────────┐
       │     MiniStackVM (runtime/)    │
       │  - Isolated Sandbox (No eval) │
       │  - Sub-20µs Execution         │
       │  - Gas Limit (max_steps=250)  │
       └───────────────┬───────────────┘
                       │
                       ▼
   [ Verified Deterministic Mathematical Output ]
```

---

## 4. Mathematical Foundations

Feistback replaces transformer self-attention with three foundational mathematical mechanisms:

### 4.1 Reversible Feistel Networks
In standard deep networks, intermediate activations for all layers must be stored in memory during the forward pass to compute gradients during backpropagation, resulting in $\mathcal{O}(N_{\text{layers}} \times D \times L)$ activation memory.

Feistback partitions hidden activations into two symmetric halves $x_t = [L_t, R_t] \in \mathbb{R}^{d_{\text{model}} / 2} \times \mathbb{R}^{d_{\text{model}} / 2}$. Each Feistel block computes:
$$L_{i+1} = R_i$$
$$R_{i+1} = L_i + \gamma \cdot \mathcal{F}(R_i, S_i)$$
where $\gamma$ is a residual contraction scale factor ($\gamma = 0.05$), and $\mathcal{F}$ is the non-linear round function.

**Exact Analytic Inverse:**
During backpropagation, activations of layer $i$ are reconstructed in reverse directly from layer $i+1$ without caching:
$$R_i = L_{i+1}$$
$$L_i = R_{i+1} - \gamma \cdot \mathcal{F}(L_{i+1}, S_i)$$
This guarantees **$\mathcal{O}(1)$ activation memory** during training regardless of network depth.

---

### 4.2 Circular Holographic Mixing (Spectral Convolution)
Instead of pairwise token-to-token attention matrices $QK^T$, sequential mixing is achieved in the frequency domain through circular convolution:
$$\mathcal{F}_{\text{spectral}}(R) = \text{iFFT}\left(\text{FFT}(R) \odot W_{\text{spectral}}\right)$$
where:
- $\text{FFT}$ and $\text{iFFT}$ are Fast Fourier Transforms along the sequence dimension;
- $W_{\text{spectral}} \in \mathbb{C}^{D/2 \times K}$ is a parameterized complex filter;
- Causal masking is applied in the temporal domain to enforce autoregressive property.

---

### 4.3 Reversible Associative Feistel (RAF v2)
#### Why 1D Linear Recurrence Failed (Interference Saturation)
Early versions of `PureFeistelLM` utilized 1D scalar linear recurrence (Gated Linear Recurrence, GLR):
$$s_t = \alpha_t \odot s_{t-1} + (1 - \alpha_t) \odot \mathcal{F}(x_t)$$
Because $s_t \in \mathbb{R}^{D}$ is a single 1D vector, incoming token features were linearly accumulated into the same vector space. When tracking multiple program variables simultaneously (e.g. `u`, `v`, `w`), vector addition caused catastrophic **relational interference saturation**, trapping the model at an inelastic cross-entropy ceiling of $\text{CE} \approx 6.84$.

#### 2D Associative Outer-Product Fast Weights
RAF resolves relational interference by expanding the hidden state into a set of 2D matrix associative fast-weights $M_t^{(h)} \in \mathbb{R}^{K \times V}$ across $H$ heads ($H=4, K=16, V=16$).

For each head $h$:
1. **Key, Value, and Retention Gate Projections:**
   $$k_t = W_k x_t \in \mathbb{R}^K, \quad v_t = W_v x_t \in \mathbb{R}^V, \quad \beta_t = \sigma(W_\beta x_t + b_\beta) \in (0, 1)$$
2. **Associative Memory Update (Decaying Outer Product):**
   $$M_t^{(h)} = \beta_t \cdot M_{t-1}^{(h)} + (1 - \beta_t) \cdot (k_t \otimes v_t)$$
   where $k_t \otimes v_t = k_t v_t^T \in \mathbb{R}^{K \times V}$.
3. **Memory Retrieval:**
   Given a query projection $q_t = W_q x_t \in \mathbb{R}^K$:
   $$y_t^{(h)} = q_t \cdot M_t^{(h)} \in \mathbb{R}^V$$

#### Dual Mode Parity (Parallel Scan vs. $\mathcal{O}(1)$ Recurrent Step)
- **Training Mode:** Executed via parallel cumulative operations on GPUs, achieving $\sim 28,000 - 40,000\text{ tokens/sec}$ throughput.
- **Inference Mode:** Executed token-by-token using the $\mathcal{O}(1)$ recurrent state equation.
- **Numerical Equivalence:** Parity verification tests confirm the difference between parallel training outputs and recurrent step outputs is bounded below $2.69 \times 10^{-7}$, well within numerical tolerance.

---

### 4.4 Deterministic Stack Virtual Machine (`MiniStackVM`)
`MiniStackVM` is a lightweight, zero-dependency stack-based execution sandbox.

* **No Dynamic Compilation Risk:** Completely disallows arbitrary `eval()`, `exec()`, or OS system calls.
* **Gas Metering:** Executes instructions with strict step counting (`max_steps=250`), guaranteeing termination and preventing denial-of-service loops.
* **Hardware-Style Instruction Words:**
  ```text
  0000: LOAD_ARG       0     # Push argument 0 to data stack
  0002: LOAD_ARG       1     # Push argument 1 to data stack
  0004: ADD                  # Pop top two values, push sum
  0005: RET                  # Terminate and return top of stack
  ```
* **Performance:** Execution latency ranges between **1.9 µs and 18 µs** per call.

---

## 5. Repository Structure (Modular Monorepo)

The repository is strictly structured into modular, decoupled domains:

```text
feistback-engine/
├── runtime/                          # Production execution & neural architecture
│   ├── engine.py                     # FeistbackEngine (orchestration & verification)
│   ├── compiler.py                   # ASTCompiler (Python AST -> MiniStackVM bytecode)
│   ├── models/                       # Model layers (PureFeistelLM, RAF v2, SpectralMixer)
│   └── vm/                           # MiniStackVM emulator, opcodes, instruction sets
├── dispatchers/                      # Intent routing & multi-turn dialogue
│   ├── base.py                       # BaseDispatcher abstract interface & utilities
│   ├── fast_path.py                  # ConfigDispatcher (sub-millisecond rule routing)
│   └── council.py                    # CouncilDispatcher (eesti_keele_sl2ng micro-model router)
├── models/                           # Production checkpoints & vocabularies
│   ├── logic_feistel_clean/          # 3.8M PureFeistelLM weights + 16k pruned BPE
│   └── eesti_keele_sl2ng/            # 2.5M Estonian slang expert weights + 4k tokenizer
├── training/                         # Model training workshop
│   ├── data_generators/              # Synthetic dataset generators (10k slang, clean logic)
│   ├── tokenizers/                   # BPE tokenizer training scripts
│   └── trainers/                     # Distributed trainers for Feistel models
├── scripts/                          # User-facing CLI tools
│   ├── interactive_chat.py           # Real-time interactive neuro-symbolic chat
│   └── run_engine.py                 # Direct prompt synthesis and VM verification CLI
├── tests/                            # Rigorous parity and verification test suite
│   ├── test_smart_vm.py              # MiniStackVM instruction & gas limit validation
│   ├── test_dispatcher.py            # Natural language & Estengrish routing test
│   ├── test_model_to_vm.py           # End-to-end OOD synthesis & execution test
│   └── verify_raf_numerics.py        # Numerical tolerance test (parallel vs recurrent)
├── archive/                          # Archived legacy weights & checkpoints (.gitignored)
├── DCO.md                            # Developer Certificate of Origin (EUPL-1.2 compliant)
├── CONTRIBUTING.md                   # Lightweight contribution guide
└── LICENSE                           # European Union Public Licence (EUPL-1.2)
```

---

## 6. Getting Started

### 6.1 Prerequisites & Virtual Environment
This project requires Python 3.10+ and PyTorch. Always operate inside the repository virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 6.2 Running the Verification Test Suite
Execute the full test suite to verify VM parity, dispatcher accuracy, and numeric stability:

```powershell
# 1. Verify MiniStackVM mathematical parity & gas limits
.\.venv\Scripts\python.exe tests/test_smart_vm.py

# 2. Verify natural language intent routing & Estengrish parsing
.\.venv\Scripts\python.exe tests/test_dispatcher.py

# 3. Verify end-to-end neural synthesis to VM execution on OOD prompts
.\.venv\Scripts\python.exe tests/test_model_to_vm.py

# 4. Verify numerical precision between parallel and recurrent modes
.\.venv\Scripts\python.exe tests/verify_raf_numerics.py
```

### 6.3 Interactive Neuro-Symbolic Chat CLI
Start the real-time interactive chat interface:

```powershell
.\.venv\Scripts\python.exe scripts/interactive_chat.py
```

Try natural language queries such as:
* `"liida kokku 15 ja 25"`
* `"clämbi see counter 105 vahemikku 0 ja 100"`
* `"arvuta vahe ruut arvudel 15 ja 8 ja liida üks otsa"`
* `"viska xor 12 ja 4 vahel"`
* `"kas number -7 on negatiivne"`

The system will match the intent, synthesize verified Python code, compile it into MiniStackVM bytecode, execute the function deterministically, and output the result in sub-millisecond VM turnaround.

---

## 7. What We Are Building Right Now (Active Frontier)

We are actively advancing the frontier in three directions:

1. **Expanding the Council of Specialties:**
   Deploying specialized sub-5M parameter micro-models trained on specific semantic domains (e.g., Estonian colloquial slang, log parsing, tabular transformations) that output formal execution contracts rather than raw text.
2. **Self-Hydrating Execution Cache:**
   Caching verified bytecode for canonicalized ASTs so that frequent mathematical and procedural operations bypass neural generation entirely (< 20 µs total turnaround).
3. **Hardware-Native Stack Acceleration:**
   Exploring WebAssembly (WASM) and embedded C compilation targets for MiniStackVM bytecode to run Feistback programs inside microcontrollers and edge gateways with microsecond response times.

---

## 8. License & Contributions

### European Union Public Licence (EUPL-1.2)
This project is licensed under the **European Union Public Licence (EUPL-1.2)**.
The EUPL is a modern, reciprocal (copyleft) open-source license crafted by the European Commission, legally validated across all 24 official languages of the European Union, and fully compatible with GPLv2/v3, AGPLv3, LGPL, MPL, and Apache 2.0. See [LICENSE](LICENSE) for full details.

### Developer Certificate of Origin (DCO 1.1)
To ensure contribution remains frictionless and requires no complex third-party CLA infrastructure, we use the standard **Developer Certificate of Origin (DCO)**.

Contributors certify their submissions by adding a `Signed-off-by` trailer line to commit messages:
```bash
git commit -s -m "feat(runtime): optimize associative memory matrix stride"
```
For detailed instructions, see [DCO.md](DCO.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 9. Citation & Research References

* **RAF v2 Breakthrough:** [docs/RAF_V2_BREAKTHROUGH_REPORT.md](docs/RAF_V2_BREAKTHROUGH_REPORT.md)
* **Architecture Post-Mortem & Research Notes:** [docs/RESEARCH_AND_POSTMORTEM.md](docs/RESEARCH_AND_POSTMORTEM.md)
* **Council of Specialties Specification (RFC-002):** [docs/RFC_002_COUNCIL_OF_SPECIALTIES_V2.md](docs/RFC_002_COUNCIL_OF_SPECIALTIES_V2.md)