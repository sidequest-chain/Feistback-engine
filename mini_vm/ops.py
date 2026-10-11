"""
Legacy Opcodes Module (Deprecated)
==================================
Maintained for backward compatibility. Please import from runtime.vm.opcodes instead.
"""

from runtime.vm.opcodes import OpCode, OPCODE_NAMES, NAME_TO_OPCODE, ARG_OPCODES

__all__ = ["OpCode", "OPCODE_NAMES", "NAME_TO_OPCODE", "ARG_OPCODES"]
