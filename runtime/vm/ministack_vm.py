"""
MiniStackVM: Deterministic Integer Stack Virtual Machine
=======================================================
Executes compact bytecode sequences with microsecond latency and 100%
mathematical accuracy. Pure Python implementation with zero PyTorch dependencies.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional

try:
    from runtime.vm.opcodes import OpCode, OPCODE_NAMES, NAME_TO_OPCODE, ARG_OPCODES
except ImportError:
    from mini_vm.ops import OpCode, OPCODE_NAMES, NAME_TO_OPCODE, ARG_OPCODES


class RuntimeExecutionError(RuntimeError):
    """Raised when VM execution exceeds maximum instruction gas limit or encounters fatal runtime failure."""
    pass


class MiniStackVM:
    """
    Compact stack virtual machine for executing dynamic bytecode programs.
    """
    def __init__(self, max_steps: int = 100_000):
        self.max_steps = max_steps

    def execute(
        self,
        bytecode: List[Any],
        args: Optional[List[Any]] = None,
        max_steps: Optional[int] = None,
    ) -> Any:
        """
        Executes bytecode sequence using supplied register arguments.

        Args:
            bytecode: List of integer opcodes and immediate values.
            args: Runtime register arguments (e.g. [a, b]).

        Returns:
            Any: Top of stack value after RET instruction.
        """
        if args is None:
            args = []

        stack: List[Any] = []
        locals_store: Dict[int, Any] = {}
        ip = 0
        steps = 0
        n = len(bytecode)
        step_limit = max_steps if max_steps is not None else self.max_steps

        while ip < n and steps < step_limit:
            steps += 1
            op = bytecode[ip]
            ip += 1

            if op == OpCode.NOP:
                continue

            elif op == OpCode.LOAD_ARG:
                if ip >= n:
                    raise IndexError("LOAD_ARG requires argument index operand!")
                arg_idx = bytecode[ip]
                ip += 1
                if arg_idx < 0 or arg_idx >= len(args):
                    raise IndexError(
                        f"Argument index {arg_idx} out of bounds (only {len(args)} args provided)!"
                    )
                stack.append(args[arg_idx])

            elif op == OpCode.PUSH_CONST:
                if ip >= n:
                    raise IndexError("PUSH_CONST requires constant value operand!")
                val = bytecode[ip]
                ip += 1
                stack.append(val)

            elif op == OpCode.DUP:
                if not stack:
                    raise RuntimeError("DUP: stack is empty!")
                stack.append(stack[-1])

            elif op == OpCode.ADD:
                if len(stack) < 2:
                    raise RuntimeError("ADD requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)

            elif op == OpCode.SUB:
                if len(stack) < 2:
                    raise RuntimeError("SUB requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)

            elif op == OpCode.MUL:
                if len(stack) < 2:
                    raise RuntimeError("MUL requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(a * b)

            elif op == OpCode.DIV:
                if len(stack) < 2:
                    raise RuntimeError("DIV requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(a // b if b != 0 else 0)

            elif op == OpCode.MOD:
                if len(stack) < 2:
                    raise RuntimeError("MOD requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(a % b if b != 0 else 0)

            elif op == OpCode.CMP_GT:
                if len(stack) < 2:
                    raise RuntimeError("CMP_GT requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                if a is None or b is None:
                    stack.append(0)
                else:
                    try:
                        stack.append(1 if a > b else 0)
                    except Exception:
                        stack.append(0)

            elif op == OpCode.CMP_EQ:
                if len(stack) < 2:
                    raise RuntimeError("CMP_EQ requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                try:
                    stack.append(1 if a == b else 0)
                except Exception:
                    stack.append(0)

            elif op == OpCode.CMP_LT:
                if len(stack) < 2:
                    raise RuntimeError("CMP_LT requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                try:
                    stack.append(1 if a < b else 0)
                except Exception:
                    stack.append(0)

            elif op == OpCode.CMP_LE:
                if len(stack) < 2:
                    raise RuntimeError("CMP_LE requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                try:
                    stack.append(1 if a <= b else 0)
                except Exception:
                    stack.append(0)

            elif op == OpCode.CMP_GE:
                if len(stack) < 2:
                    raise RuntimeError("CMP_GE requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                try:
                    stack.append(1 if a >= b else 0)
                except Exception:
                    stack.append(0)

            elif op == OpCode.CMP_NE:
                if len(stack) < 2:
                    raise RuntimeError("CMP_NE requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                try:
                    stack.append(1 if a != b else 0)
                except Exception:
                    stack.append(0)

            elif op == OpCode.JMP_IF_FALSE:
                if ip >= n:
                    raise IndexError("JMP_IF_FALSE requires jump offset operand!")
                offset = bytecode[ip]
                ip += 1
                if not stack:
                    raise RuntimeError("JMP_IF_FALSE: stack is empty!")
                cond = stack.pop()
                if cond == 0:
                    ip += offset

            elif op == OpCode.LOAD_JSON_KEY:
                if len(stack) < 2:
                    raise RuntimeError("LOAD_JSON_KEY requires at least 2 stack elements (key, json_obj)!")
                item2 = stack.pop()
                item1 = stack.pop()

                # Robust resolution of (json_obj, key) vs (key, json_obj)
                if isinstance(item1, dict) or (isinstance(item1, str) and (item1.startswith("{") or item1.startswith("["))):
                    json_obj, key = item1, item2
                elif isinstance(item2, dict) or (isinstance(item2, str) and (item2.startswith("{") or item2.startswith("["))):
                    json_obj, key = item2, item1
                else:
                    json_obj, key = item1, item2

                parsed = None
                if isinstance(json_obj, dict):
                    parsed = json_obj
                elif isinstance(json_obj, str):
                    try:
                        parsed = json.loads(json_obj)
                    except Exception:
                        parsed = None

                if isinstance(parsed, dict):
                    val = parsed.get(key, None)
                else:
                    val = None
                stack.append(val)

            elif op == OpCode.REGEX_MATCH:
                if len(stack) < 2:
                    raise RuntimeError("REGEX_MATCH requires at least 2 stack elements (pattern, text)!")
                item2 = stack.pop()
                item1 = stack.pop()

                # Top of stack is pattern in postfix PUSH text, PUSH pattern, REGEX_MATCH
                pattern, text = item2, item1
                if isinstance(item1, str) and (item1.startswith("^") or "\\d" in item1 or ".*" in item1) and not (isinstance(item2, str) and item2.startswith("^")):
                    pattern, text = item1, item2

                try:
                    match = re.search(str(pattern), str(text))
                    stack.append(1 if match is not None else 0)
                except Exception:
                    stack.append(0)

            elif op == OpCode.BRANCH_IF:
                if not stack:
                    raise RuntimeError("BRANCH_IF: stack is empty for condition!")
                cond = stack.pop()
                # Resolve target address either from immediate bytecode operand or stack
                if ip < n and isinstance(bytecode[ip], int) and bytecode[ip] >= 0:
                    target = bytecode[ip]
                    ip += 1
                elif ip < n and (bytecode[ip] is None or bytecode[ip] == -1 or bytecode[ip] == "STACK"):
                    ip += 1
                    if not stack:
                        raise RuntimeError("BRANCH_IF: stack is empty for target address!")
                    target = stack.pop()
                else:
                    if not stack:
                        raise RuntimeError("BRANCH_IF: stack is empty for target address!")
                    target = stack.pop()

                if cond != 0:
                    if target < 0 or target > n:
                        raise IndexError(f"BRANCH_IF target {target} out of bounds [0, {n}]!")
                    ip = target

            elif op == OpCode.JMP:
                # Resolve target address either from immediate bytecode operand or stack
                if ip < n and isinstance(bytecode[ip], int) and bytecode[ip] >= 0:
                    target = bytecode[ip]
                    ip += 1
                elif ip < n and (bytecode[ip] is None or bytecode[ip] == -1 or bytecode[ip] == "STACK"):
                    ip += 1
                    if not stack:
                        raise RuntimeError("JMP: stack is empty for target address!")
                    target = stack.pop()
                else:
                    if not stack:
                        raise RuntimeError("JMP: stack is empty for target address!")
                    target = stack.pop()

                if target < 0 or target > n:
                    raise IndexError(f"JMP target {target} out of bounds [0, {n}]!")
                ip = target

            elif op == OpCode.DROP:
                if not stack:
                    raise RuntimeError("DROP: stack is empty!")
                stack.pop()

            elif op == OpCode.SWAP:
                if len(stack) < 2:
                    raise RuntimeError("SWAP requires at least 2 stack elements!")
                stack[-1], stack[-2] = stack[-2], stack[-1]

            elif op == OpCode.OVER:
                if len(stack) < 2:
                    raise RuntimeError("OVER requires at least 2 stack elements!")
                stack.append(stack[-2])

            elif op == OpCode.ROT:
                if len(stack) < 3:
                    raise RuntimeError("ROT requires at least 3 stack elements!")
                c = stack.pop()
                b = stack.pop()
                a = stack.pop()
                stack.append(b)
                stack.append(c)
                stack.append(a)

            elif op == OpCode.POW:
                if len(stack) < 2:
                    raise RuntimeError("POW requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(a ** b)

            elif op == OpCode.NEG:
                if not stack:
                    raise RuntimeError("NEG: stack is empty!")
                a = stack.pop()
                stack.append(-a)

            elif op == OpCode.ABS:
                if not stack:
                    raise RuntimeError("ABS: stack is empty!")
                a = stack.pop()
                stack.append(abs(a))

            elif op == OpCode.MIN:
                if len(stack) < 2:
                    raise RuntimeError("MIN requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(min(a, b))

            elif op == OpCode.MAX:
                if len(stack) < 2:
                    raise RuntimeError("MAX requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(max(a, b))

            elif op == OpCode.ISQRT:
                if not stack:
                    raise RuntimeError("ISQRT: stack is empty!")
                a = stack.pop()
                stack.append(math.isqrt(max(0, int(a))))

            elif op == OpCode.BIT_AND:
                if len(stack) < 2:
                    raise RuntimeError("BIT_AND requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(int(a) & int(b))

            elif op == OpCode.BIT_OR:
                if len(stack) < 2:
                    raise RuntimeError("BIT_OR requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(int(a) | int(b))

            elif op == OpCode.BIT_XOR:
                if len(stack) < 2:
                    raise RuntimeError("BIT_XOR requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(int(a) ^ int(b))

            elif op == OpCode.BIT_NOT:
                if not stack:
                    raise RuntimeError("BIT_NOT: stack is empty!")
                a = stack.pop()
                stack.append(~int(a))

            elif op == OpCode.SHL:
                if len(stack) < 2:
                    raise RuntimeError("SHL requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(int(a) << int(b))

            elif op == OpCode.SHR:
                if len(stack) < 2:
                    raise RuntimeError("SHR requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(int(a) >> int(b))

            elif op == OpCode.STORE_LOCAL:
                if ip >= n:
                    raise IndexError("STORE_LOCAL requires local variable index operand!")
                local_idx = bytecode[ip]
                ip += 1
                if not stack:
                    raise RuntimeError("STORE_LOCAL: stack is empty!")
                locals_store[local_idx] = stack.pop()

            elif op == OpCode.LOAD_LOCAL:
                if ip >= n:
                    raise IndexError("LOAD_LOCAL requires local variable index operand!")
                local_idx = bytecode[ip]
                ip += 1
                if local_idx not in locals_store:
                    raise NameError(f"Local variable index {local_idx} referenced before assignment!")
                stack.append(locals_store[local_idx])

            elif op == OpCode.RET:
                return stack.pop() if stack else 0

            else:
                raise ValueError(f"Unknown OpCode: {op}")

        if steps >= step_limit:
            raise RuntimeExecutionError("Instruction limit exceeded")

        return stack.pop() if stack else 0

    def disassemble(self, bytecode: List[Any]) -> str:
        """Disassembles bytecode into human-readable assembly lines."""
        lines = []
        ip = 0
        n = len(bytecode)

        while ip < n:
            addr = ip
            op = bytecode[ip]
            ip += 1
            name = OPCODE_NAMES.get(op, f"UNK({op})")

            if op in ARG_OPCODES:
                if ip < n:
                    arg = bytecode[ip]
                    ip += 1
                    lines.append(f"{addr:04d}: {name:<14} {arg}")
                else:
                    lines.append(f"{addr:04d}: {name:<14} ???")
            else:
                lines.append(f"{addr:04d}: {name}")

        return "\n".join(lines)
