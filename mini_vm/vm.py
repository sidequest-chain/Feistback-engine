"""
MiniStackVM: Deterministic Integer Stack Virtual Machine
=======================================================
Executes compact bytecode sequences with microsecond latency and 100%
mathematical accuracy. Pure Python implementation with zero PyTorch dependencies.
"""

from typing import List, Optional

from mini_vm.ops import OpCode, OPCODE_NAMES, NAME_TO_OPCODE, ARG_OPCODES


class MiniStackVM:
    """
    Compact stack virtual machine for executing dynamic bytecode programs.
    """
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps

    def execute(self, bytecode: List[int], args: Optional[List[int]] = None) -> int:
        """
        Executes bytecode sequence using supplied register arguments.

        Args:
            bytecode: List of integer opcodes and immediate values.
            args: Runtime register arguments (e.g. [a, b]).

        Returns:
            int: Top of stack value after RET instruction.
        """
        if args is None:
            args = []

        stack: List[int] = []
        ip = 0
        steps = 0
        n = len(bytecode)

        while ip < n and steps < self.max_steps:
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
                stack.append(1 if a > b else 0)

            elif op == OpCode.CMP_EQ:
                if len(stack) < 2:
                    raise RuntimeError("CMP_EQ requires at least 2 stack elements!")
                b = stack.pop()
                a = stack.pop()
                stack.append(1 if a == b else 0)

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

            elif op == OpCode.RET:
                return stack.pop() if stack else 0

            else:
                raise ValueError(f"Unknown OpCode: {op}")

        if steps >= self.max_steps:
            raise TimeoutError(f"VM execution step limit ({self.max_steps}) exceeded!")

        return stack.pop() if stack else 0

    def disassemble(self, bytecode: List[int]) -> str:
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
                    lines.append(f"{addr:04d}: {name:<12} {arg}")
                else:
                    lines.append(f"{addr:04d}: {name:<12} ???")
            else:
                lines.append(f"{addr:04d}: {name}")

        return "\n".join(lines)
