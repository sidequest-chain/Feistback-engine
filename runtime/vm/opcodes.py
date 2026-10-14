"""
Deterministic Stack Virtual Machine Opcodes
===========================================
Defines the compact instruction opcode set for MiniStackVM:
Zero external dependencies (pure Python).
"""

from typing import Dict, Set


class OpCode:
    """Virtual Machine Instruction Set."""
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
    CMP_LT = 17        # POP b, POP a -> PUSH 1 if a < b else 0
    CMP_LE = 18        # POP b, POP a -> PUSH 1 if a <= b else 0
    CMP_GE = 19        # POP b, POP a -> PUSH 1 if a >= b else 0
    CMP_NE = 20        # POP b, POP a -> PUSH 1 if a != b else 0

    # Stack Management
    DROP = 21          # POP top element and discard
    SWAP = 22          # POP b, POP a -> PUSH b, PUSH a
    OVER = 23          # Copy 2nd stack element to top (a, b -> a, b, a)
    ROT = 24           # Rotate top 3 stack elements (a, b, c -> b, c, a)

    # Scientific Mathematics
    POW = 25           # POP b, POP a -> PUSH a ** b
    NEG = 26           # POP a -> PUSH -a
    ABS = 27           # POP a -> PUSH abs(a)
    MIN = 28           # POP b, POP a -> PUSH min(a, b)
    MAX = 29           # POP b, POP a -> PUSH max(a, b)
    ISQRT = 30         # POP a -> PUSH integer square root

    # Bitwise Operations
    BIT_AND = 31       # POP b, POP a -> PUSH a & b
    BIT_OR = 32        # POP b, POP a -> PUSH a | b
    BIT_XOR = 33       # POP b, POP a -> PUSH a ^ b
    BIT_NOT = 34       # POP a -> PUSH ~a
    SHL = 35           # POP b, POP a -> PUSH a << b
    SHR = 36           # POP b, POP a -> PUSH a >> b

    # Local Variables
    STORE_LOCAL = 37   # Argument: local_idx -> POP val, locals[local_idx] = val
    LOAD_LOCAL = 38    # Argument: local_idx -> PUSH locals[local_idx]

    # Floating Point & Extended Scientific Mathematics
    FDIV = 39          # POP b, POP a -> PUSH float(a) / float(b) (true division)
    FLOAT = 40         # POP a -> PUSH float(a) (integer to float conversion)
    ROUND = 41         # POP a -> PUSH round(a)
    FLOOR = 42         # POP a -> PUSH math.floor(a)
    CEIL = 43          # POP a -> PUSH math.ceil(a)
    PUSH_FLOAT = 44    # Argument: 8-byte IEEE-754 float -> PUSH float_val

    # Sandboxed Deterministic Tool Execution
    CALL_TOOL = 45     # Argument: tool_id_or_name -> POP <arity> args, PUSH tool(*args)


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
    OpCode.CMP_LT: "CMP_LT",
    OpCode.CMP_LE: "CMP_LE",
    OpCode.CMP_GE: "CMP_GE",
    OpCode.CMP_NE: "CMP_NE",
    OpCode.DROP: "DROP",
    OpCode.SWAP: "SWAP",
    OpCode.OVER: "OVER",
    OpCode.ROT: "ROT",
    OpCode.POW: "POW",
    OpCode.NEG: "NEG",
    OpCode.ABS: "ABS",
    OpCode.MIN: "MIN",
    OpCode.MAX: "MAX",
    OpCode.ISQRT: "ISQRT",
    OpCode.BIT_AND: "BIT_AND",
    OpCode.BIT_OR: "BIT_OR",
    OpCode.BIT_XOR: "BIT_XOR",
    OpCode.BIT_NOT: "BIT_NOT",
    OpCode.SHL: "SHL",
    OpCode.SHR: "SHR",
    OpCode.STORE_LOCAL: "STORE_LOCAL",
    OpCode.LOAD_LOCAL: "LOAD_LOCAL",
    OpCode.FDIV: "FDIV",
    OpCode.FLOAT: "FLOAT",
    OpCode.ROUND: "ROUND",
    OpCode.FLOOR: "FLOOR",
    OpCode.CEIL: "CEIL",
    OpCode.PUSH_FLOAT: "PUSH_FLOAT",
    OpCode.CALL_TOOL: "CALL_TOOL",
}

NAME_TO_OPCODE: Dict[str, int] = {v: k for k, v in OPCODE_NAMES.items()}
NAME_TO_OPCODE["IDIV"] = OpCode.DIV
NAME_TO_OPCODE["LOAD_FLOAT"] = OpCode.PUSH_FLOAT

ARG_OPCODES: Set[int] = {
    OpCode.LOAD_ARG,
    OpCode.PUSH_CONST,
    OpCode.JMP_IF_FALSE,
    OpCode.BRANCH_IF,
    OpCode.JMP,
    OpCode.STORE_LOCAL,
    OpCode.LOAD_LOCAL,
    OpCode.PUSH_FLOAT,
    OpCode.CALL_TOOL,
}
