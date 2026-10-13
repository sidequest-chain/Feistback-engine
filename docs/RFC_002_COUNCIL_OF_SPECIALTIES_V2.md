# TECHNICAL SPECIFICATION (RFC): FEISTBACK V2 ARCHITECTURE

**Document ID:** RFC-2026-FEIST-002  
**Topic:** Dynamic Council of Specialties, `eesti_keele_sl2ng` Micro-Model, and Self-Hydrating Cache  
**Status:** IMPLEMENTED & VERIFIED  
**Target Environments:** Python 3.12 (Prototyping / Training) $\longrightarrow$ Pure ANSI C / Rust (Zero-Dependency Bare-Metal Runtime)  

---

## 1. Overview and Core Philosophy

This specification defines the second-generation (v2) architecture of the Feistback engine. The primary objective is to transition from manually maintained, brittle regex configuration tables (`config/intents.json`) to an autonomous, distributed **Council of Specialties** composed of compact, trained micro-models without compromising our strict low-resource guarantees ($O(1)$ memory footprint, 0 bytes KV-cache, microsecond deterministic execution).

### 1.1. Baseline Achievements (v1.0 -> v2.0)
- **Deterministic Logic Core:** A 3.83M parameter PureFeistelLM model (12 reversible layers, 24 KB RAF associative matrix state, 0 bytes KV-cache) generating clean multi-line Pythonic logic (training loss: 0.0087).
- **Hardware-Agnostic Execution Sandbox:** `MiniStackVM` compiles and deterministically executes stack bytecode within 4 to 20 microseconds with cycle protection (gas limit).
- **Pluggable Dispatcher Interface:** `BaseDispatcher` abstraction with `ConfigDispatcher` (`dispatchers/fast_path.py`) and `CouncilDispatcher` (`dispatchers/council.py`) coupled with `LearnedIntentCache` (`dispatchers/cache_manager.py`).

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
│  - Architecture: PureFeistelLM (2.5M–4.5M parameters)   │                  │
│  - 24 KB RAF State, 0 B KV-Cache, Specialized 4k BPE    │                  │
│  - Role: O(1) Recurrent Semantic Parsing & Contracts    │                  │
│  - Strict Boundary: DOES NOT compute math directly      │                  │
└──────────────────────────┬──────────────────────────────┘                  │
                           │                                                 │
            Structured Execution Contract:                                   │
            - Target Function Signature (e.g., "def add(a: int, b: int)...") │
            - Extracted Positional Arguments (e.g., [15, 25])                │
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
   - Immutable repository containing fundamental primitive signatures (`add`, `sub`, `clamp_math`, `bitwise_comb`, `calc_with_locals`, `is_negative`, etc.).
   - Deployed as read-only code assets via `dispatchers/fast_path.py`.

2. **`cache/learned_intents.json` (Dynamic RAM/Disk LRU Cache):**
   - Managed by `LearnedIntentCache` in `dispatchers/cache_manager.py`.
   - Stores structural query patterns successfully resolved by the Tier 2 Language Expert and verified by Tier 3.
   - **Capacity Constraint:** Fixed capacity of 500 entries governed by a Least Recently Used (LRU) eviction strategy to prevent cache bloat.
   - **Double-Lock Verification Gate:** Patterns are **never** written to cache based solely on the Language Expert's prediction. Cache writes occur exclusively after `MiniStackVM` completes execution, returns valid data, and passes verification checks (`verified_by_vm=True`).

---

### 3.2. Tier 2: Language Expert (`eesti_keele_sl2ng`)
The Language Expert is a specialized neural model responsible for natural language understanding, colloquial parsing, and mathematical intent structuring.

- **Architecture:** PureFeistelLM (12 reversible layers, $d_{\text{model}}=512$, 24 KB RAF recurrent state, 0 bytes KV-cache).
- **Token Vocabulary:** Dedicated BPE tokenizer (`models/eesti_keele_sl2ng/tokenizer.json`) optimized for Estonian grammatical roots, case endings, and technical slang (estengrish). Completely isolated from the 16k code vocabulary used by the Logic Specialist.
- **Functional Scope:**
  - Maps natural language (e.g., *"pane 15 ja 25 kokku"*, *"liida kolm arvu 1, 5 ja 6 lahuta sellest 2"*) into an unambiguous formal contract using $O(1)$ recurrent token stepping.
  - Preserves strict left-to-right positional argument order.
  - Determines the execution strategy: `single_pass` (Kernel Fusion) vs. `need_input` (Multi-turn slot filling).

---

### 3.3. Tier 3: Hybrid Execution Architecture
To support both complex multi-step arithmetic and multi-specialist pipelines, the engine implements a hybrid execution model:

| Strategy | Scope | Execution Mechanism | Latency |
| :--- | :--- | :--- | :--- |
| **Kernel Fusion (2A)** | Multi-operation arithmetic within a single domain | Language Expert formulates a consolidated multi-argument signature. The Logic Specialist synthesizes multi-line code utilizing local registers. MiniStackVM compiles and runs the entire routine in one pass. | **8 – 20 µs** |
| **Pipeline DAG (2B)** | Cross-domain workflows across multiple council specialists | Dispatcher coordinates sequential data flow: Specialist A output is mapped as input to Specialist B. | **1 – 5 ms** |

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
      "matched_pattern": "pane <NUM> ja <NUM> kokku",
      "intent_name": "add",
      "target_signature": "def add(a: int, b: int) -> int:",
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

## 5. Implementation Status

```
Phase 1: L1 Decoupling & LRU Infrastructure [COMPLETED]
├── Static fast-path routing in dispatchers/fast_path.py
├── LearnedIntentCache (LRU eviction & atomic persistence) in dispatchers/cache_manager.py
└── Double-Lock Verification Gate in dispatchers/council.py

Phase 2: Training Data Pipeline for eesti_keele_sl2ng [COMPLETED]
├── Dataset generator in training/data_generators/build_slang_dataset.py
├── 10,000 prompt-to-contract training pairs in data/slang_train_10k.jsonl
└── Domain-specific BPE tokenizer in training/tokenizers/train_slang_tokenizer.py

Phase 3: Language Expert Training & Council Orchestration [COMPLETED]
├── Trained PureFeistelLM Language Expert (models/eesti_keele_sl2ng/slang_expert_feistel.pt)
├── O(1) recurrent CouncilDispatcher integrating L1 Cache, Language Expert, and Logic Specialist
└── Interactive verification via scripts/interactive_chat.py and tests/test_dispatcher.py
```
