"""
Feistback Runtime Package
=========================
Provides the core execution engine, AST compiler, canonicalizer,
Layer B JIT bytecode cache, MiniStackVM, ToolRegistry, and Feistel neural models.
"""

from runtime.bytecode_cache import BytecodeCache, compute_file_sha256
from runtime.canonicalizer import canonicalize_signature, decanonicalize_code
from runtime.compiler import ASTCompiler, compile_to_vm
from runtime.engine import EngineResult, FeistbackEngine, TestOutcome
from runtime.registry import ModelRegistry
from runtime.vm.tools import DEFAULT_TOOL_REGISTRY, ToolRegistry, ToolSpec

__all__ = [
    "FeistbackEngine",
    "EngineResult",
    "TestOutcome",
    "BytecodeCache",
    "compute_file_sha256",
    "ASTCompiler",
    "compile_to_vm",
    "ModelRegistry",
    "ToolRegistry",
    "ToolSpec",
    "DEFAULT_TOOL_REGISTRY",
    "canonicalize_signature",
    "decanonicalize_code",
]
