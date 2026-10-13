"""
Feistback Core Runtime Package
==============================
Couples PureFeistelLM neuro-symbolic execution with MiniStackVM and ASTCompiler.
"""

from runtime.canonicalizer import canonicalize_signature, decanonicalize_code
from runtime.engine import FeistbackEngine, EngineResult, TestOutcome
from runtime.compiler import ASTCompiler
from runtime.vm.ministack_vm import MiniStackVM, RuntimeExecutionError
from runtime.vm.opcodes import OpCode
from runtime.models.pure_feistel_lm import PureFeistelLM
from runtime.registry import ModelRegistry

__all__ = [
    "FeistbackEngine",
    "EngineResult",
    "TestOutcome",
    "ASTCompiler",
    "MiniStackVM",
    "RuntimeExecutionError",
    "OpCode",
    "PureFeistelLM",
    "ModelRegistry",
    "canonicalize_signature",
    "decanonicalize_code",
]
