# Feistback Engine

Experimental research sandbox exploring lightweight sequence modeling and execution interfaces.

## Status

Early-stage research prototype. Internal testing and active experimentation in progress.
Architecture, components, and interfaces are subject to change without notice.

## License

Licensed under the European Union Public Licence (EUPL-1.2). See `LICENSE` for details.

## Architecture: Neuro-Symbolic Feistback Stack

Feistback couples an ultra-lightweight 3.8M parameter recurrent model (**PureFeistelLM** with **Reversible Associative Feistel (RAF)** memory, 0 bytes KV-cache) with a deterministic, zero-dependency stack virtual machine (**MiniStackVM**).

```
[User Query (Estonian / Estengrish)]
               │
               ▼
   BaseDispatcher / ConfigDispatcher  ──▶  config/intents.json
               │
     (Formal Function Signature)
               │
               ▼
        FeistbackEngine               ──▶  runner/engine.py
               │
         (Python AST)
               │
               ▼
          ASTCompiler                 ──▶  mini_vm/compiler.py
               │
       (Compact Bytecode)
               │
               ▼
          MiniStackVM                 ──▶  mini_vm/vm.py (sub-50µs execution)
```

### Natural Language Intent Dispatcher (`runner/dispatcher.py`)
- **`BaseDispatcher`**: Abstract interface decoupling query routing from language modeling.
- **`ConfigDispatcher`**: Declarative router matching user queries against `config/intents.json` using weighted regex and keyword scoring.
- **Microsecond Determinism**: Generates code with PureFeistelLM and runs execution on MiniStackVM with 100% mathematical parity.

### Running Verification Tests
```bash
.\.venv\Scripts\python.exe tests/test_dispatcher.py
```

### Architectural Specifications & RFCs
- [RFC-2026-FEIST-002: Feistback V2 Architecture](docs/RFC_002_COUNCIL_OF_SPECIALTIES_V2.md) – Dynamic Council of Specialties, `eesti_keele_sl2ng` Micro-Model, and Self-Hydrating Cache.
- [RAF v2 Breakthrough Report](docs/RAF_V2_BREAKTHROUGH_REPORT.md) – Reversible Associative Feistel matrix memory and zero KV-cache verification.