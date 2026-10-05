"""
Deterministic Stack Virtual Machine Opcodes
===========================================
Defines the compact 16-instruction opcode set for MiniStackVM:
Zero external dependencies (pure Python).
"""

from typing import Dict, Set


class OpCode:
    """Virtual Machine Instruction Set (up to 16 opcodes)."""
    NOP = 0
    LOAD_ARG = 1      # Argument: arg_index -> PUSH args[arg_index]
    PUSH_CONST = 2    # Argument: const_val -> PUSH const_val
    ADD = 3           # POP b, POP a -> PUSH a + b
    SUB = 4           # POP b, POP a -> PUSH a - b
    MUL = 5           # POP b, POP a -> PUSH a * b
    DIV = 6           # POP b, POP a -> PUSH a // b (safe integer division)
    MOD = 7           # POP b, POP a -> PUSH a % b
    CMP_GT = 8        # POP b, POP a -> PUSH 1 if a > b else 0
    CMP_EQ = 9        # POP b, POP a -> PUSH 1 if a == b else 0
    JMP_IF_FALSE = 10 # Argument: rel_offset -> POP cond, if not cond: IP += rel_offset
    RET = 11          # POP res -> RETURN res
    DUP = 12          # PUSH stack[-1] (duplicate top of stack)
    LOAD_JSON_KEY = 13 # POP key, POP json_obj -> PUSH json_obj[key]
    REGEX_MATCH = 14   # POP pattern, POP text -> PUSH 1 if match else 0
    BRANCH_IF = 15     # Argument/Stack: target_addr -> POP cond, if cond != 0: IP = target_addr
    JMP = 16           # Argument/Stack: target_addr -> IP = target_addr


OPCODE_NAMES: Dict[int, str] = {
    OpCode.NOP: "NOP",
    OpCode.LOAD_ARG: "LOAD_ARG",
    OpCode.PUSH_CONST: "PUSH_CONST",
    OpCode.ADD: "ADD",
    OpCode.SUB: "SUB",
    OpCode.MUL: "MUL",
    OpCode.DIV: "DIV",
    OpCode.MOD: "MOD",
    OpCode.CMP_GT: "CMP_GT",
    OpCode.CMP_EQ: "CMP_EQ",
    OpCode.JMP_IF_FALSE: "JMP_IF_FALSE",
    OpCode.RET: "RET",
    OpCode.DUP: "DUP",
    OpCode.LOAD_JSON_KEY: "LOAD_JSON_KEY",
    OpCode.REGEX_MATCH: "REGEX_MATCH",
    OpCode.BRANCH_IF: "BRANCH_IF",
    OpCode.JMP: "JMP",
}

NAME_TO_OPCODE: Dict[str, int] = {v: k for k, v in OPCODE_NAMES.items()}
ARG_OPCODES: Set[int] = {
    OpCode.LOAD_ARG,
    OpCode.PUSH_CONST,
    OpCode.JMP_IF_FALSE,
    OpCode.BRANCH_IF,
    OpCode.JMP,
}
