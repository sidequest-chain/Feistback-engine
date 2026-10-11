"""
Legacy Mini VM Package (Deprecated)
===================================
Maintained for backward compatibility. Please import from runtime.vm or runtime.compiler instead.
"""

import warnings
warnings.warn(
    "Importing from 'mini_vm' is deprecated and will be removed in a future release. "
    "Please update imports to 'runtime.vm' or 'runtime.compiler'.",
    DeprecationWarning,
    stacklevel=2,
)

from runtime.vm.ministack_vm import MiniStackVM, RuntimeExecutionError
from runtime.vm.opcodes import OpCode, OPCODE_NAMES, NAME_TO_OPCODE, ARG_OPCODES
from runtime.compiler import ASTCompiler

__all__ = [
    "MiniStackVM",
    "RuntimeExecutionError",
    "OpCode",
    "OPCODE_NAMES",
    "NAME_TO_OPCODE",
    "ARG_OPCODES",
    "ASTCompiler",
]
