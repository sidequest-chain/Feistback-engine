# TECHNICAL SPECIFICATION (RFC): FEISTBACK V2 ARCHITECTURE

**Document ID:** RFC-2026-FEIST-002  
**Topic:** Dynamic Council of Specialties, `eesti_keele_sl2ng` Micro-Model, and Self-Hydrating Cache  
**Status:** APPROVED ARCHITECTURAL SPECIFICATION  
**Target Environments:** Python 3.12 (Prototyping / Training) $\longrightarrow$ Pure ANSI C / Rust (Zero-Dependency Bare-Metal Runtime)  

---

## 1. Overview and Core Philosophy

This specification defines the second-generation (v2) architecture of the Feistback engine. The primary objective is to transition from manually maintained, brittle regex configuration tables (`config/intents.json`) to an autonomous, distributed **Council of Specialties** composed of compact, trained micro-models without compromising our strict low-resource guarantees ($O(1)$ memory footprint, 0 bytes KV-cache, microsecond deterministic execution).

### 1.1. Baseline Achievements (v1.0)
- **Deterministic Logic Core:** A 3.83M parameter PureFeistelLM model (12 reversible layers, 24 KB RAF associative matrix state, 0 bytes KV-cache) generating clean multi-line Pythonic logic (training loss: 0.0087).
- **Hardware-Agnostic Execution Sandbox:** `MiniStackVM` compiles and deterministically executes stack bytecode within 4 to 20 microseconds with cycle protection (gas limit).
- **Pluggable Dispatcher Interface:** `BaseDispatcher` abstraction with `ConfigDispatcher` providing CLI interaction (`scripts/interactive_chat.py`), mode routing (`synthesize` vs. `execute`), and slot-filling dialogue (`NEED_INPUT`).

### 1.2. The Scaling Wall of Static Configurations
Attempting to scale `config/intents.json` to cover rich natural languages (e.g., Estonian with 14 grammatical cases, compounding, and colloquial IT "estengrish") introduces critical architectural defects:
- **Combinatorial Rule Explosion:** Maintaining regex patterns and synonym matrices for every inflection and colloquial phrasing quickly becomes unmaintainable.
- **Inability to Generalize Multi-Step Operations:** Compound queries (e.g., "add 15 and 25 and deduct a quarter") require bespoke composite regexes for every permutation.
- **Architectural Axiom:** Configuration files must act strictly as an **L1 Fast-Path Cache**, while semantic language comprehension and signature formulation are offloaded to a dedicated, trained micro-model.

---

## 2. Three-Tier Topology

```
User Input (Colloquial speech, Estonian slang, Estengrish, Compound statements)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   TIER 1: L1 FAST-PATH CACHE                           │
│                                                                        │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐   │
│   │ config/intents.json (Static)│    │ cache/learned_intents.json  │   │
│   │ ~20-30 Immutable Primitives │    │ Dynamic Self-Hydrating LRU  │   │
│   │ (add, clamp, xor, etc.)     │    │ Cache (Capacity: 500 items) │   │
│   └──────────────┬──────────────┘    └──────────────┬──────────────┘   │
└──────────────────┼──────────────────────────────────┼──────────────────┘
                   │                                  │
                   └────────────────┬─────────────────┘
                                    │
                         Cache Hit in L1 Fast-Path?
                                    ├── YES ─────────────────────────────────┐
                                    │                                        │
                                    NO                                       │
                                    │                                        │
                                    ▼                                        │
┌─────────────────────────────────────────────────────────┐                  │
│       TIER 2: LANGUAGE EXPERT (eesti_keele_sl2ng)       │                  │
│  - Architecture: PureFeistelLM (3.5M–4.5M parameters)   │                  │
│  - 24 KB RAF State, 0 B KV-Cache, Specialized 4k/8k BPE │                  │
│  - Role: Semantic Parsing & Contract Synthesis          │                  │
│  - Strict Boundary: DOES NOT compute math directly      │                  │
└──────────────────────────┬──────────────────────────────┘                  │
                           │                                                 │
            Structured Execution Contract:                                   │
            - Target Function Signature (e.g., "def add_and_vat...")         │
            - Extracted Positional Arguments (e.g., [100, 24])               │
            - Strategy: [Kernel Fusion | Pipeline DAG]                       │
                           │                                                 │
                           ├─────────────────────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│              TIER 3: LOGIC CORE & DETERMINISTIC VERIFICATION           │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ Logic Specialist (3.83M PureFeistelLM, 16k Code Vocabulary)    │   │
│   │ Synthesizes function body (multi-line logic, local registers)  │   │
│   └──────────────────────────────┬─────────────────────────────────┘   │
│                                  │                                     │
│                                  ▼                                     │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ ASTCompiler & MiniStackVM                                      │   │
│   │ Compiles to bytecode and executes within 10 µs in sandbox      │   │
│   └──────────────────────────────┬─────────────────────────────────┘   │
└──────────────────────────────────┼─────────────────────────────────────┘
                                   │
                    Did VM Execution & Verification Pass?
                                   ├── YES ──▶ [Persist to cache/learned_intents.json]
                                   └── NO  ──▶ [Reject / Do NOT persist to cache]
```

---

## 3. Detailed Component Specifications

### 3.1. Tier 1: Minimal L1 Fast-Path & Self-Hydrating Cache
The L1 layer guarantees sub-millisecond execution for known commands while dynamically adapting to user terminology:

1. **`config/intents.json` (Static ROM Base):**
   - Immutable repository containing strictly 20 to 30 fundamental primitive signatures (`add`, `sub`, `clamp_math`, `bitwise_comb`, `shift_and_mask`, `abs_diff_pow`, etc.).
   - Deployed as read-only code assets.

2. **`cache/learned_intents.json` (Dynamic RAM/Disk LRU Cache):**
   - Stores patterns successfully resolved by the Tier 2 Language Expert and verified by Tier 3.
   - **Capacity Constraint:** Fixed capacity of 500 entries governed by a Least Recently Used (LRU) eviction strategy to prevent cache bloat.
   - **Double-Lock Verification Gate:** Patterns are **never** written to cache based solely on the Language Expert's prediction. Cache writes occur exclusively after `MiniStackVM` completes execution, returns valid data, and passes all domain sanity checks.

---

### 3.2. Tier 2: Language Expert (`eesti_keele_sl2ng`)
The Language Expert is a specialized neural model responsible for natural language understanding, colloquial parsing, and mathematical intent structuring.

- **Architecture:** PureFeistelLM (12 reversible layers, $d_{\text{model}}=896$, 24 KB RAF recurrent state, 0 bytes KV-cache).
- **Parameter Target:** 3.5M to 4.5M parameters.
- **Token Vocabulary:** Dedicated 4096 or 8192 BPE tokenizer optimized for Estonian grammatical roots, case endings, and technical slang (estengrish). Completely isolated from the 16k code vocabulary used by the Logic Specialist.
- **Functional Scope:**
  - Maps natural language (e.g., *"suru number kaheksa vahele kaks ja kümme"*, *"löö need kaks kokku ja arvesta viiekas maha"*) into an unambiguous formal signature.
  - Preserves strict left-to-right positional argument order.
  - Determines the execution strategy: `single_pass` (Kernel Fusion) vs. `pipeline` (DAG).

#### Output Schema (Structured Contract):
```json
{
  "intent_name": "sum_and_discount",
  "execution_strategy": "single_pass",
  "signature": "def sum_and_discount(a: int, b: int, discount: int) -> int:",
  "args": [15, 25, 5],
  "confidence": 0.985
}
```

---

### 3.3. Tier 3: Hybrid Execution Architecture
To support both complex multi-step arithmetic and multi-specialist pipelines, the engine implements a hybrid execution model:

| Strategy | Scope | Execution Mechanism | Latency |
| :--- | :--- | :--- | :--- |
| **Kernel Fusion (2A)** | Multi-operation arithmetic within a single domain | Language Expert formulates a consolidated multi-argument signature. The Logic Specialist synthesizes multi-line code utilizing local registers. MiniStackVM compiles and runs the entire routine in one pass. | **8 – 15 µs** |
| **Pipeline DAG (2B)** | Cross-domain workflows across multiple council specialists | Dispatcher coordinates sequential data flow: Specialist A output is mapped as input to Specialist B (e.g., Log/Text Parser $\longrightarrow$ Arithmetic Verification $\longrightarrow$ Output Formatter). | **1 – 5 ms** |

---

## 4. Schemas and Data Formats

### 4.1. Static Baseline Schema (`config/intents.json`)
```json
{
  "version": "2.0.0",
  "engine_target": "MiniStackVM",
  "intents": [
    {
      "id": "add",
      "name": "Binary Addition",
      "signature_template": "def add(a: int, b: int) -> int:",
      "expected_args_count": 2,
      "keywords": ["liida", "kokku", "pluss", "add"],
      "regex_patterns": ["\\bliida\\b", "\\bkokku\\b", "\\+"],
      "description": "Adds two integers together."
    },
    {
      "id": "clamp_math",
      "name": "Mathematical Clamping",
      "signature_template": "def clamp_range_1(u: int, v: int, w: int) -> int:",
      "expected_args_count": 3,
      "keywords": ["clämbi", "klembi", "suru vahele", "piira"],
      "regex_patterns": ["\\bcl[aä]mbi\\b", "suru.*vahele"],
      "description": "Restricts value to range [min, max]."
    }
  ]
}
```

### 4.2. Self-Hydrating Cache Schema (`cache/learned_intents.json`)
```json
{
  "max_entries": 500,
  "entries": [
    {
      "query_hash": "e9b4a11c0f",
      "matched_pattern": "löö kokku ja arvesta käibemaks",
      "target_signature": "def add_and_vat(a: int, b: int) -> int:",
      "execution_strategy": "single_pass",
      "args_count": 2,
      "last_accessed": 1775573870,
      "hit_count": 38,
      "verified_by_vm": true
    }
  ]
}
```

---

## 5. Implementation Roadmap

```
Phase 1: L1 Decoupling & LRU Infrastructure
├── Prune config/intents.json down to ~20 primitive operations
├── Implement runner/cache_manager.py (thread-safe LRU eviction and atomic file persistence)
└── Implement the Double-Lock Verification Gate in runner/dispatcher.py

Phase 2: Training Data Pipeline for eesti_keele_sl2ng
├── Create scripts/build_slang_dataset.py
├── Synthesize 10,000 prompt-to-contract training pairs:
│   ├── Estonian colloquialisms ("löö kokku", "viska punti", "suru vahele")
│   ├── IT-slang / Estengrish ("clämbi", "xori", "shifti vasakule", "maski ära")
│   └── Compound operations ("liida a ja b ning võta ruut")
└── Train and export a domain-specific 4k/8k BPE tokenizer

Phase 3: Language Expert Training & Council Orchestration
├── Train the PureFeistelLM Language Expert on modern GPU hardware (~10 min training cycle)
├── Build CouncilDispatcher integrating L1 Cache, Language Expert, and Logic Specialist
└── Verify compound pipelines via scripts/interactive_chat.py

Phase 4: Standalone C/Rust Port
├── Quantize and export model weights to INT8/INT4 binary format (weights.bin)
└── Implement zero-dependency C inference runtime (feistback.c and vm.c)
```

---

## 6. Risk Assessment & Mitigations

1. **Cache Pollution via Semantic Drift:**
   - *Risk:* The Language Expert misinterprets an ambiguous query, and an erroneous signature pattern is written to cache.
   - *Mitigation:* Cache insertion is gated by MiniStackVM execution. If VM execution fails, encounters an unhandled opcode, exceeds the gas limit, or violates boundary constraints, cache write is aborted.

2. **Signature Hallucination:**
   - *Risk:* The Language Expert outputs invalid or unparseable Python syntax.
   - *Mitigation:* `ASTCompiler` parses and validates the generated signature prior to handing it to the Logic Specialist. If AST validation fails, the engine falls back to `NEED_INPUT` dialogue.
