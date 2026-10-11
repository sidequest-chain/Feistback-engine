# Feistback Engine
### Ultra-Lightweight Neuro-Symbolic Architecture with Zero-KV Recurrent State & Deterministic Stack VM

[![License: EUPL-1.2](https://img.shields.io/badge/License-EUPL%201.2-blue.svg)](LICENSE)
[![Architecture: RAF-v2](https://img.shields.io/badge/Architecture-RAF--v2%20(Reversible%20Feistel)-orange.svg)](docs/RAF_V2_BREAKTHROUGH_REPORT.md)
[![KV Cache](https://img.shields.io/badge/KV--Cache-0%20Bytes%20(O(1)%20Memory)-brightgreen.svg)](#the-kv-cache-memory-wall)
[![Deterministic VM](https://img.shields.io/badge/MiniStackVM-Sub--20µs%20Deterministic-purple.svg)](#44-deterministic-stack-virtual-machine-ministackvm)
[![DCO 1.1](https://img.shields.io/badge/Contributions-DCO%201.1%20Signed--off-success.svg)](DCO.md)

---

## Status: Experimental Research Prototype

> [!WARNING]
> **Active Research Sandbox**: Feistback is an early-stage experimental research initiative exploring scalable alternatives to quadratic Transformer attention, memory walls, and monolithic LLM architectures. It is **not** a production-ready drop-in replacement for general-purpose LLMs. Internal testing, architectural iterations, and benchmark validations are actively ongoing. Interfaces and internal representations are subject to change without notice.

---

## 1. What is Feistback Engine?

**Feistback Engine** is an experimental research framework exploring lightweight neuro-symbolic sequence modeling and deterministic execution interfaces. 

Rather than building another monolithic, multi-billion parameter "jack-of-all-trades" model that suffers from memory exhaustion and stochastic hallucinations, Feistback investigates a fundamentally different paradigm: **a modular, decentralized architecture capable of orchestrating hundreds or thousands of specialized sub-5M parameter micro-experts, coupled with a zero-dependency deterministic execution sandbox.**

### Core Pillars:
1. **Ultra-Compact Recurrent Backbone (`PureFeistelLM`):** Operates with **identically 0 bytes of Key-Value (KV) cache** and $\mathcal{O}(1)$ inference memory, powered by **Reversible Associative Feistel (RAF v2)** outer-product fast-weight memory matrices and circular holographic frequency mixing.
2. **Deterministic Stack Virtual Machine (`MiniStackVM`):** Transpiles neural code synthesis into formal 16-bit instruction words in sub-0.1 milliseconds, executing in sub-20 microseconds with 100% mathematical parity and strict gas limits.
3. **Decentralized Council of Micro-Experts (`dispatchers/`):** A modular routing infrastructure capable of coordinating thousands of hyper-focused micro-models (such as our reference 2.5M colloquial slang/contract specialist, logic engines, protocol decoders, and domain adapters), creating a scalable, open alternative to centralized monolithic models.

---

## 2. What Problem Are We Solving?

### The KV-Cache Memory Wall
Modern auto-regressive Transformers (LLaMA, Mistral, Qwen, GPT-4) scale quadratically in computational complexity with sequence length $\mathcal{O}(N^2)$ during prefill and require a linearly growing Key-Value cache during auto-regressive generation:
$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times N_{\text{layers}} \times D_{\text{head}} \times N_{\text{kv\_heads}} \times \text{bytes\_per\_elem}$$
For high-concurrency serving or resource-constrained edge hardware, the KV-cache consumes gigabytes of VRAM per session, causing severe memory fragmentation and memory-bandwidth bottlenecks.

### The Stochastic Hallucination Problem
When monolithic LLMs perform arithmetic, programmatic reasoning, or state transitions, they rely on probabilistic token sampling. Even a 70-billion parameter model cannot guarantee that:
- Basic arithmetic operations evaluate without numerical drift;
- Loops and branch conditions terminate deterministically without infinite execution;
- Variable bindings remain stable under minor prompt phrasing variations.

---

## 3. The Grand Vision: Thousands of Micro-Experts vs. Monolithic LLMs

The primary motivation of Feistback is **not** to build a single regional model (the Estonian slang expert was simply a reference proof-of-concept for domain-specific contract extraction). 

The true objective is to explore a **viable, modular, open-source alternative to centralized monolithic LLMs**:

```
                 [ User / System Query ]
                            │
                            ▼
           ┌─────────────────────────────────┐
           │   Hierarchical Router / Council │
           └──────────────┬──────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
│ Expert #001:    │ │ Expert #002:│ │ Expert #N:      │
│ Dialect / Slang │ │ Arithmetic│ │ Protocol Decoder│  ... [Thousands of Micro-Experts]
│ (2.5M Params)   │ │ (3.8M)    │ │ (2.0M Params)   │
└────────┬────────┘ └─────┬─────┘ └────────┬────────┘
         │                │                │
         └────────────────┼────────────────┘
                          │ Formal Execution Contract / Typed AST
                          ▼
           ┌─────────────────────────────────┐
           │     AST Compiler & Verifier     │
           └──────────────┬──────────────────┘
                          │ Verified Bytecode
                          ▼
           ┌─────────────────────────────────┐
           │   Deterministic MiniStackVM     │
           └─────────────────────────────────┘
```

### Why a Council of Thousands of Micro-Experts Can Outperform Monoliths:
* **Decentralized Modularity:** Instead of spending millions of dollars training or fine-tuning a 70B parameter behemoth (which risks catastrophic forgetting across unrelated domains), individual micro-experts (~2M to 5M parameters) can be trained or updated in **minutes** on consumer GPUs.
* **Extreme Specialization:** A dedicated 2.5M micro-model trained purely on domain contracts, API schemas, or colloquial speech achieves near-perfect fidelity without carrying the dead weight of generic internet pretraining.
* **Horizontal Scalability:** Micro-experts can be dynamically loaded, hot-swapped, cached, or pruned at runtime based on the task demand.
* **Formal Contract Interfacing:** Because micro-experts emit structured JSON execution contracts or formal Python ASTs rather than unstructured text, hallucinations are caught immediately by static compilers before execution.

---

## 4. How the Trainer and Runner Actually Work

### 4.1 How the Trainer Operates (`training/`)
The training pipeline bridges the gap between parallel GPU efficiency and $\mathcal{O}(1)$ recurrent state modeling:

```
[ Dataset JSONL ] ──▶ [ Tokenizer ] ──▶ [ Sequence Packing ] ──▶ [ Loss Masking ]
                                                                       │
┌──────────────────────────────────────────────────────────────────────┘
│
▼ (GPU Forward Pass)
┌────────────────────────────────────────────────────────┐
│  Parallel Associative Prefix Scan (RAF v2)             │
│  - Compute all k_t, v_t, beta_t in parallel across L   │
│  - Cumulative decay outer products computed via tensor │
│    broadcasting: M_t = sum_{i=1}^t (decay_i * (k ⊗ v)) │
│  - Training Throughput: ~28,000 - 40,000 tokens/sec    │
└───────────────────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────┐
│  Reversible Feistel Backpropagation                    │
│  - Intermediate activations are NOT cached in memory   │
│  - Exact analytic reverse reconstruction:              │
│    R_i = L_{i+1},  L_i = R_{i+1} - gamma * F(L_{i+1})  │
│  - O(1) Backpropagation Memory                         │
└────────────────────────────────────────────────────────┘
```

1. **Structured Data Curation:** Training pairs are formatted into streaming sequences:
   `<|user|> {instruction} <|contract|> {contract_json} <|endoftext|>`
2. **Target Loss Masking:** To prevent the model from wasting capacity on memorizing user queries, prompt tokens are masked with label `-100`. Gradients only flow through the generated contract tokens (`loss_mask = targets != -100`).
3. **Dual-Mode Parallel Prefix Scan:** While generation is step-by-step, GPU training executes via parallel tensor prefix operations across the sequence dimension, achieving **28k–40k tokens/second** on consumer hardware.
4. **Reversible Gradient Flow:** Layers reconstruct their input activations analytically during the backward pass, eliminating activation caching.

---

### 4.2 How the Runner and Execution Pipeline Operates (`runtime/`)
The runtime workflow transforms natural queries into deterministically executed results:

```
Step 1: Input Query ──▶ BaseDispatcher / CouncilDispatcher
                             │
                             ▼ (Extract Arguments & Target Function)
Step 2: Formal Signature ──▶ PureFeistelLM Autoregressive Step
                             │ (Step-by-step recurrence, 24 KB state, 0 bytes KV)
                             ▼
Step 3: Synthesized Python ─▶ ASTCompiler
                             │ (ast.parse, syntax verification, bytecode generation)
                             ▼
Step 4: 16-bit Bytecode ───▶ MiniStackVM Sandbox
                             │ (Sub-20µs execution, gas limit: max_steps=250)
                             ▼
                        [ Verified Result ]
```

1. **Semantic Routing:** The query is routed through either `ConfigDispatcher` (rule-based fast-path) or a specialized neural micro-model (`CouncilDispatcher`) to extract structured argument vectors and identify the formal target signature.
2. **Autoregressive Generation:** `PureFeistelLM` steps forward one token at a time. In each step, the recurrent 2D fast-weight matrix $M_t = \beta_t M_{t-1} + (1-\beta_t)(k_t \otimes v_t)$ updates in constant time with **0 bytes allocated to a KV-cache**.
3. **AST Transpilation:** The generated Python function string is ingested by `ASTCompiler`, parsed into Python's native Abstract Syntax Tree, verified against security rules (disallowing arbitrary OS imports, `eval`, or file access), and transpiled into compact instruction words.
4. **Isolated VM Execution:** The bytecode is pushed into `MiniStackVM`, initialized with input arguments, evaluated on an isolated integer stack, and halted at `RET` within microsecond latency (~1.9 to 18 µs).

---

## 5. Mathematical Foundations

Feistback replaces transformer self-attention with three foundational mathematical mechanisms:

### 5.1 Reversible Feistel Networks
In standard deep networks, intermediate activations for all layers must be stored in memory during the forward pass to compute gradients during backpropagation, resulting in $\mathcal{O}(N_{\text{layers}} \times D \times L)$ activation memory.

Feistback partitions hidden activations into two symmetric halves $x_t = [L_t, R_t] \in \mathbb{R}^{d_{\text{model}} / 2} \times \mathbb{R}^{d_{\text{model}} / 2}$. Each Feistel block computes:
$$L_{i+1} = R_i$$
$$R_{i+1} = L_i + \gamma \cdot \mathcal{F}(R_i, S_i)$$
where $\gamma = 0.05$ is a residual contraction scale factor, and $\mathcal{F}$ is the non-linear round function.

**Exact Analytic Inverse:**
During backpropagation, activations of layer $i$ are reconstructed in reverse directly from layer $i+1$ without caching:
$$R_i = L_{i+1}$$
$$L_i = R_{i+1} - \gamma \cdot \mathcal{F}(L_{i+1}, S_i)$$
This guarantees **$\mathcal{O}(1)$ activation memory** during training regardless of network depth.

---

### 5.2 Circular Holographic Mixing (Spectral Convolution)
Instead of pairwise token-to-token attention matrices $QK^T$, sequential mixing is achieved in the frequency domain through circular convolution:
$$\mathcal{F}_{\text{spectral}}(R) = \text{iFFT}\left(\text{FFT}(R) \odot W_{\text{spectral}}\right)$$
where:
- $\text{FFT}$ and $\text{iFFT}$ are Fast Fourier Transforms along the sequence dimension;
- $W_{\text{spectral}} \in \mathbb{C}^{D/2 \times K}$ is a parameterized complex filter;
- Causal masking is applied in the temporal domain to enforce the autoregressive property.

---

### 5.3 Reversible Associative Feistel (RAF v2)
#### Why 1D Linear Recurrence Failed (Interference Saturation)
Early versions of `PureFeistelLM` utilized 1D scalar linear recurrence:
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

### 5.4 Deterministic Stack Virtual Machine (`MiniStackVM`)
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

## 6. Repository Structure (Modular Monorepo)

```text
feistback-engine/
├── runtime/                          # Execution engine & neural architecture
│   ├── engine.py                     # FeistbackEngine (orchestration & verification)
│   ├── compiler.py                   # ASTCompiler (Python AST -> MiniStackVM bytecode)
│   ├── models/                       # Model layers (PureFeistelLM, RAF v2, SpectralMixer)
│   └── vm/                           # MiniStackVM emulator, opcodes, instruction sets
├── dispatchers/                      # Intent routing & multi-turn dialogue
│   ├── base.py                       # BaseDispatcher abstract interface & utilities
│   ├── fast_path.py                  # ConfigDispatcher (sub-millisecond rule routing)
│   └── council.py                    # CouncilDispatcher (specialist micro-model router)
├── models/                           # Trained model weights & vocabularies
│   ├── logic_feistel_clean/          # 3.8M PureFeistelLM weights + 16k pruned BPE
│   └── eesti_keele_sl2ng/            # Reference 2.5M slang expert weights + 4k tokenizer
├── training/                         # Model training workshop
│   ├── data_generators/              # Synthetic dataset generators (slang, clean logic)
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

## 7. Getting Started

### 7.1 Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 7.2 Running Verification Tests
```powershell
# MiniStackVM mathematical parity & gas limits
.\.venv\Scripts\python.exe tests/test_smart_vm.py

# Natural language intent routing & Estengrish parsing
.\.venv\Scripts\python.exe tests/test_dispatcher.py

# End-to-end neural synthesis to VM execution on OOD prompts
.\.venv\Scripts\python.exe tests/test_model_to_vm.py

# Numerical precision between parallel and recurrent modes
.\.venv\Scripts\python.exe tests/verify_raf_numerics.py
```

### 7.3 Interactive Neuro-Symbolic Chat CLI
```powershell
.\.venv\Scripts\python.exe scripts/interactive_chat.py
```

Try natural language queries such as:
* `"liida kokku 15 ja 25"`
* `"clämbi see counter 105 vahemikku 0 ja 100"`
* `"arvuta vahe ruut arvudel 15 ja 8 ja liida üks otsa"`
* `"viska xor 12 ja 4 vahel"`
* `"kas number -7 on negatiivne"`

---

## 8. License & Contributions

### European Union Public Licence (EUPL-1.2)
This project is licensed under the **European Union Public Licence (EUPL-1.2)**.
The EUPL is a reciprocal (copyleft) open-source license crafted by the European Commission, legally validated across all 24 official languages of the European Union, and fully compatible with GPLv2/v3, AGPLv3, LGPL, MPL, and Apache 2.0. See [LICENSE](LICENSE) for full details.

### Developer Certificate of Origin (DCO 1.1)
To ensure contribution remains frictionless and requires no third-party CLA infrastructure, we use the standard **Developer Certificate of Origin (DCO)**.

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