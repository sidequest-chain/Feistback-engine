"""
AST to MiniStackVM Bytecode Compiler
====================================
Compiles Python AST representations of pure functions into deterministic,
zero-dependency MiniStackVM bytecode instructions.
Supports binary/unary math, bitwise ops, local variables, and built-in calls.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

from runtime.vm.opcodes import OpCode


class ASTCompiler:
    """
    Compiles Python function ASTs into MiniStackVM executable bytecode.
    """

    def __init__(self) -> None:
        self.bytecode: List[Any] = []
        self.arg_map: Dict[str, int] = {}
        self.arg_names: List[str] = []
        self.local_map: Dict[str, int] = {}

    def compile(self, source: str, args_hint: Optional[List[str]] = None) -> List[Any]:
        """
        Compiles a Python function source string into MiniStackVM bytecode.

        Args:
            source: Python source code string defining a function (or body snippet).
            args_hint: Optional list of argument names if compiling a standalone snippet.

        Returns:
            List[Any]: Sequence of opcodes and immediate values.
        """
        self.bytecode = []
        self.arg_map = {}
        self.arg_names = []
        self.local_map = {}

        clean_src = source.strip()
        if not clean_src.startswith("def ") and not clean_src.startswith("async def "):
            params = ", ".join(args_hint) if args_hint else ""
            indented = "\n".join("    " + line for line in clean_src.split("\n") if line.strip())
            clean_src = f"def _wrapper({params}):\n{indented}"

        tree = ast.parse(clean_src)

        if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("Input source must contain at least one FunctionDef!")

        fn_node = tree.body[0]
        self._compile_function(fn_node)
        return self.bytecode

    def _compile_function(self, fn: ast.FunctionDef) -> None:
        """Extracts arguments and compiles the function body statements."""
        for idx, arg in enumerate(fn.args.args):
            self.arg_names.append(arg.arg)
            self.arg_map[arg.arg] = idx

        has_explicit_return = False
        for stmt in fn.body:
            if isinstance(stmt, ast.Return):
                self._compile_return(stmt)
                has_explicit_return = True
                break
            elif isinstance(stmt, ast.Assign):
                self._compile_assign(stmt)
            elif isinstance(stmt, ast.AugAssign):
                self._compile_augassign(stmt)
            elif isinstance(stmt, ast.If):
                self._compile_if_statement(stmt)
            elif isinstance(stmt, ast.Pass):
                continue
            elif isinstance(stmt, ast.Expr):
                # Ignore docstring expressions
                if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    continue
                self._compile_expr(stmt.value)
            else:
                raise NotImplementedError(f"Unsupported statement type: {type(stmt).__name__}")

        if not has_explicit_return:
            # Default fallback return 0 if no explicit return hit
            self.bytecode.extend([OpCode.PUSH_CONST, 0, OpCode.RET])

    def _compile_return(self, node: ast.Return) -> None:
        """Compiles return value expression and appends RET opcode."""
        if node.value is not None:
            self._compile_expr(node.value)
        else:
            self.bytecode.extend([OpCode.PUSH_CONST, 0])
        self.bytecode.append(OpCode.RET)

    def _compile_assign(self, node: ast.Assign) -> None:
        """Compiles assignment to local variables: expr -> STORE_LOCAL <idx>."""
        self._compile_expr(node.value)

        for idx, target in enumerate(node.targets):
            if idx > 0:
                self.bytecode.append(OpCode.DUP)
            if isinstance(target, ast.Name):
                var_name = target.id
                if var_name not in self.local_map:
                    self.local_map[var_name] = len(self.local_map)
                self.bytecode.extend([OpCode.STORE_LOCAL, self.local_map[var_name]])
            else:
                raise NotImplementedError(f"Unsupported assignment target: {type(target).__name__}")

    def _compile_augassign(self, node: ast.AugAssign) -> None:
        """Compiles augmented assignment: target += value -> STORE_LOCAL."""
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(f"Unsupported AugAssign target: {type(node.target).__name__}")

        var_name = node.target.id
        # 1. Load current value of target
        if var_name in self.local_map:
            self.bytecode.extend([OpCode.LOAD_LOCAL, self.local_map[var_name]])
        elif var_name in self.arg_map:
            self.bytecode.extend([OpCode.LOAD_ARG, self.arg_map[var_name]])
        else:
            raise NameError(f"Augmented assignment on undeclared identifier '{var_name}'!")

        # 2. Compile value expression
        self._compile_expr(node.value)

        # 3. Emit binary operation opcode
        op_type = type(node.op)
        if op_type is ast.Add:
            self.bytecode.append(OpCode.ADD)
        elif op_type is ast.Sub:
            self.bytecode.append(OpCode.SUB)
        elif op_type is ast.Mult:
            self.bytecode.append(OpCode.MUL)
        elif op_type is ast.Div:
            self.bytecode.append(OpCode.FDIV)
        elif op_type is ast.FloorDiv:
            self.bytecode.append(OpCode.DIV)
        elif op_type is ast.Mod:
            self.bytecode.append(OpCode.MOD)
        elif op_type is ast.Pow:
            self.bytecode.append(OpCode.POW)
        elif op_type is ast.BitAnd:
            self.bytecode.append(OpCode.BIT_AND)
        elif op_type is ast.BitOr:
            self.bytecode.append(OpCode.BIT_OR)
        elif op_type is ast.BitXor:
            self.bytecode.append(OpCode.BIT_XOR)
        elif op_type is ast.LShift:
            self.bytecode.append(OpCode.SHL)
        elif op_type is ast.RShift:
            self.bytecode.append(OpCode.SHR)
        else:
            raise NotImplementedError(f"Unsupported AugAssign operator: {op_type.__name__}")

        # 4. Store back to local
        if var_name not in self.local_map:
            self.local_map[var_name] = len(self.local_map)
        self.bytecode.extend([OpCode.STORE_LOCAL, self.local_map[var_name]])

    def _compile_if_statement(self, node: ast.If) -> None:
        """Compiles conditional branch statement."""
        # 1. Evaluate condition
        self._compile_expr(node.test)

        # 2. Reserve jump if false
        self.bytecode.append(OpCode.JMP_IF_FALSE)
        jmp_offset_idx = len(self.bytecode)
        self.bytecode.append(0)  # Placeholder relative offset

        # 3. Compile body
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                self._compile_return(stmt)
            elif isinstance(stmt, ast.Assign):
                self._compile_assign(stmt)
            elif isinstance(stmt, ast.AugAssign):
                self._compile_augassign(stmt)
            elif isinstance(stmt, ast.Pass):
                continue
            else:
                self._compile_expr(stmt.value if isinstance(stmt, ast.Expr) else stmt)

        # 4. Patch relative offset for JMP_IF_FALSE
        target_ip = len(self.bytecode)
        self.bytecode[jmp_offset_idx] = target_ip - (jmp_offset_idx + 1)

        # 5. Compile optional else branch
        if node.orelse:
            self.bytecode.append(OpCode.JMP)
            jmp_else_idx = len(self.bytecode)
            self.bytecode.append(0)  # Placeholder absolute target

            orelse_start = len(self.bytecode)
            self.bytecode[jmp_offset_idx] = orelse_start - (jmp_offset_idx + 1)

            for stmt in node.orelse:
                if isinstance(stmt, ast.Return):
                    self._compile_return(stmt)
                elif isinstance(stmt, ast.Assign):
                    self._compile_assign(stmt)
                elif isinstance(stmt, ast.AugAssign):
                    self._compile_augassign(stmt)
                elif isinstance(stmt, ast.Pass):
                    continue
                else:
                    self._compile_expr(stmt.value if isinstance(stmt, ast.Expr) else stmt)

            self.bytecode[jmp_else_idx] = len(self.bytecode)

    def _compile_expr(self, node: ast.AST) -> None:
        """Compiles an expression in postfix evaluation order."""
        if isinstance(node, ast.BinOp):
            self._compile_binop(node)

        elif isinstance(node, ast.Compare):
            self._compile_compare(node)

        elif isinstance(node, ast.UnaryOp):
            self._compile_unaryop(node)

        elif isinstance(node, ast.Call):
            self._compile_call(node)

        elif isinstance(node, ast.IfExp):
            self._compile_ifexp(node)

        elif isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, bool):
                val = 1 if val else 0
            self.bytecode.extend([OpCode.PUSH_CONST, val])

        elif isinstance(node, ast.Name):
            if node.id in self.local_map:
                self.bytecode.extend([OpCode.LOAD_LOCAL, self.local_map[node.id]])
            elif node.id in self.arg_map:
                self.bytecode.extend([OpCode.LOAD_ARG, self.arg_map[node.id]])
            elif node.id == "True":
                self.bytecode.extend([OpCode.PUSH_CONST, 1])
            elif node.id == "False":
                self.bytecode.extend([OpCode.PUSH_CONST, 0])
            elif node.id == "None":
                self.bytecode.extend([OpCode.PUSH_CONST, None])
            else:
                raise NameError(f"Unbound identifier '{node.id}' in AST expression!")

        else:
            raise NotImplementedError(f"Unsupported expression node: {type(node).__name__}")

    def _compile_binop(self, node: ast.BinOp) -> None:
        """Compiles binary operators: left, right, then opcode."""
        self._compile_expr(node.left)
        self._compile_expr(node.right)

        op_type = type(node.op)
        if op_type is ast.Add:
            self.bytecode.append(OpCode.ADD)
        elif op_type is ast.Sub:
            self.bytecode.append(OpCode.SUB)
        elif op_type is ast.Mult:
            self.bytecode.append(OpCode.MUL)
        elif op_type is ast.Div:
            self.bytecode.append(OpCode.FDIV)
        elif op_type is ast.FloorDiv:
            self.bytecode.append(OpCode.DIV)
        elif op_type is ast.Mod:
            self.bytecode.append(OpCode.MOD)
        elif op_type is ast.Pow:
            self.bytecode.append(OpCode.POW)
        elif op_type is ast.BitAnd:
            self.bytecode.append(OpCode.BIT_AND)
        elif op_type is ast.BitOr:
            self.bytecode.append(OpCode.BIT_OR)
        elif op_type is ast.BitXor:
            self.bytecode.append(OpCode.BIT_XOR)
        elif op_type is ast.LShift:
            self.bytecode.append(OpCode.SHL)
        elif op_type is ast.RShift:
            self.bytecode.append(OpCode.SHR)
        else:
            raise NotImplementedError(f"Unsupported binary operator: {op_type.__name__}")

    def _compile_compare(self, node: ast.Compare) -> None:
        """Compiles comparison operators: left, comparator, then comparison opcode."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Chained comparisons (e.g. a < b < c) not supported in VM!")

        self._compile_expr(node.left)
        self._compile_expr(node.comparators[0])

        cmp_type = type(node.ops[0])
        if cmp_type is ast.Eq:
            self.bytecode.append(OpCode.CMP_EQ)
        elif cmp_type is ast.NotEq:
            self.bytecode.append(OpCode.CMP_NE)
        elif cmp_type is ast.Lt:
            self.bytecode.append(OpCode.CMP_LT)
        elif cmp_type is ast.Gt:
            self.bytecode.append(OpCode.CMP_GT)
        elif cmp_type is ast.LtE:
            self.bytecode.append(OpCode.CMP_LE)
        elif cmp_type is ast.GtE:
            self.bytecode.append(OpCode.CMP_GE)
        else:
            raise NotImplementedError(f"Unsupported comparison operator: {cmp_type.__name__}")

    def _compile_unaryop(self, node: ast.UnaryOp) -> None:
        """Compiles unary negation or bitwise/boolean inversion."""
        op_type = type(node.op)
        if op_type is ast.USub:
            self._compile_expr(node.operand)
            self.bytecode.append(OpCode.NEG)
        elif op_type is ast.Invert:
            self._compile_expr(node.operand)
            self.bytecode.append(OpCode.BIT_NOT)
        elif op_type is ast.Not:
            self._compile_expr(node.operand)
            self.bytecode.extend([OpCode.PUSH_CONST, 0, OpCode.CMP_EQ])
        elif op_type is ast.UAdd:
            self._compile_expr(node.operand)
        else:
            raise NotImplementedError(f"Unsupported unary operator: {op_type.__name__}")

    def _compile_call(self, node: ast.Call) -> None:
        """Compiles math/scientific built-in function calls: abs, min, max, pow, isqrt."""
        fn_name = ""
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fn_name = node.func.attr

        if fn_name == "abs":
            if len(node.args) != 1:
                raise ValueError("abs() takes exactly 1 argument!")
            self._compile_expr(node.args[0])
            self.bytecode.append(OpCode.ABS)

        elif fn_name == "min":
            if len(node.args) < 2:
                raise ValueError("min() requires at least 2 arguments!")
            self._compile_expr(node.args[0])
            for arg in node.args[1:]:
                self._compile_expr(arg)
                self.bytecode.append(OpCode.MIN)

        elif fn_name == "max":
            if len(node.args) < 2:
                raise ValueError("max() requires at least 2 arguments!")
            self._compile_expr(node.args[0])
            for arg in node.args[1:]:
                self._compile_expr(arg)
                self.bytecode.append(OpCode.MAX)

        elif fn_name in ("pow", "power"):
            if len(node.args) != 2:
                raise ValueError("pow() requires 2 arguments (base, exponent)!")
            self._compile_expr(node.args[0])
            self._compile_expr(node.args[1])
            self.bytecode.append(OpCode.POW)

        elif fn_name == "isqrt":
            if len(node.args) != 1:
                raise ValueError("isqrt() requires exactly 1 argument!")
            self._compile_expr(node.args[0])
            self.bytecode.append(OpCode.ISQRT)

        elif fn_name == "round":
            if len(node.args) != 1:
                raise ValueError("round() currently requires exactly 1 argument!")
            self._compile_expr(node.args[0])
            self.bytecode.append(OpCode.ROUND)

        elif fn_name == "floor":
            if len(node.args) != 1:
                raise ValueError("floor() requires exactly 1 argument!")
            self._compile_expr(node.args[0])
            self.bytecode.append(OpCode.FLOOR)

        elif fn_name == "ceil":
            if len(node.args) != 1:
                raise ValueError("ceil() requires exactly 1 argument!")
            self._compile_expr(node.args[0])
            self.bytecode.append(OpCode.CEIL)

        elif fn_name == "float":
            if len(node.args) != 1:
                raise ValueError("float() requires exactly 1 argument!")
            self._compile_expr(node.args[0])
            self.bytecode.append(OpCode.FLOAT)

        else:
            raise NotImplementedError(f"Unsupported function call in VM: '{fn_name}'")

    def _compile_ifexp(self, node: ast.IfExp) -> None:
        """Compiles ternary expression: body if test else orelse."""
        # 1. Condition
        self._compile_expr(node.test)

        # 2. Jump if false to orelse
        self.bytecode.append(OpCode.JMP_IF_FALSE)
        jmp_false_idx = len(self.bytecode)
        self.bytecode.append(0)

        # 3. Body expression
        self._compile_expr(node.body)

        # 4. Jump over orelse to end
        self.bytecode.append(OpCode.JMP)
        jmp_end_idx = len(self.bytecode)
        self.bytecode.append(0)

        # 5. Patch jump if false
        orelse_start = len(self.bytecode)
        self.bytecode[jmp_false_idx] = orelse_start - (jmp_false_idx + 1)

        # 6. Orelse expression
        self._compile_expr(node.orelse)

        # 7. Patch jump to end
        end_ip = len(self.bytecode)
        self.bytecode[jmp_end_idx] = end_ip


def compile_to_vm(source: str) -> Tuple[List[Any], List[str]]:
    """
    Convenience function to compile Python source code into MiniStackVM bytecode.

    Returns:
        Tuple[List[Any], List[str]]: Compiled bytecode instructions and argument names list.
    """
    compiler = ASTCompiler()
    bytecode = compiler.compile(source)
    return bytecode, compiler.arg_names
