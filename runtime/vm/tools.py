"""
MiniStackVM Deterministic Tool Registry
=======================================
Provides a sandboxed, zero-side-effect registry of external and domain-specific
functions invokable via OpCode.CALL_TOOL inside MiniStackVM.

Every registered tool specifies an integer tool_id (0..255 for binary bytecode
compatibility), a canonical function name, fixed argument arity, and a pure
deterministic callable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union


@dataclass(frozen=True)
class ToolSpec:
    """Specification for a deterministic VM tool callable."""
    tool_id: int
    name: str
    arity: int
    fn: Callable[..., Any]
    description: str = ""


class ToolRegistry:
    """
    Sandboxed registry mapping tool names and integer IDs to deterministic callables.
    """

    def __init__(self) -> None:
        self._by_name: Dict[str, ToolSpec] = {}
        self._by_id: Dict[int, ToolSpec] = {}
        self._next_id: int = 0

    def register(
        self,
        name: str,
        arity: int,
        fn: Optional[Callable[..., Any]] = None,
        *,
        tool_id: Optional[int] = None,
        description: str = "",
    ) -> Union[ToolSpec, Callable[[Callable[..., Any]], Callable[..., Any]]]:
        """
        Registers a deterministic tool function either directly or as a decorator.

        Args:
            name: Canonical function identifier in AST and assembly mnemonics.
            arity: Exact number of stack arguments consumed by the tool.
            fn: Callable implementation (if None, returns a decorator).
            tool_id: Optional explicit integer ID (0..255). Auto-assigned if omitted.
            description: Human-readable description of the tool contract.
        """
        def _register_fn(func: Callable[..., Any]) -> ToolSpec:
            assigned_id = tool_id if tool_id is not None else self._next_id
            while assigned_id in self._by_id and tool_id is None:
                assigned_id += 1
            self._next_id = max(self._next_id, assigned_id + 1)

            spec = ToolSpec(
                tool_id=assigned_id,
                name=name,
                arity=arity,
                fn=func,
                description=description,
            )
            self._by_name[name] = spec
            self._by_id[assigned_id] = spec
            return spec

        if fn is not None:
            return _register_fn(fn)

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            _register_fn(func)
            return func

        return _decorator

    def get(self, ref: Union[int, str]) -> ToolSpec:
        """Resolves a ToolSpec by integer ID or canonical string name."""
        if isinstance(ref, int):
            if ref not in self._by_id:
                raise KeyError(f"Unregistered VM tool ID: {ref}")
            return self._by_id[ref]
        if isinstance(ref, str):
            if ref not in self._by_name:
                # Allow numeric string fallback from assembly
                if ref.isdigit() and int(ref) in self._by_id:
                    return self._by_id[int(ref)]
                raise KeyError(f"Unregistered VM tool name: '{ref}'")
            return self._by_name[ref]
        raise TypeError(f"Invalid tool reference type: {type(ref).__name__}")

    def has(self, ref: Union[int, str]) -> bool:
        """Returns True if the given tool ID or name is registered."""
        if isinstance(ref, int):
            return ref in self._by_id
        if isinstance(ref, str):
            return ref in self._by_name or (ref.isdigit() and int(ref) in self._by_id)
        return False

    def list_tools(self) -> Dict[str, ToolSpec]:
        """Returns a dictionary of all registered tools keyed by name."""
        return dict(self._by_name)


# ==============================================================================
# Default Built-in Deterministic Domain Tools
# ==============================================================================

def _tool_gcd(a: Any, b: Any) -> int:
    return math.gcd(int(a), int(b))


def _tool_lcm(a: Any, b: Any) -> int:
    ia, ib = int(a), int(b)
    if ia == 0 or ib == 0:
        return 0
    return abs(ia * ib) // math.gcd(ia, ib)


def _tool_discount(price: Any, pct: Any) -> float:
    """Calculates final price after subtracting percentage discount."""
    return round(float(price) * (100.0 - float(pct)) / 100.0, 4)


def _tool_vat_add(amount: Any, vat_pct: Any) -> float:
    """Adds percentage VAT/tax to a base amount."""
    return round(float(amount) * (100.0 + float(vat_pct)) / 100.0, 4)


def _tool_pct_of(part: Any, total: Any) -> float:
    """Calculates what percentage `part` is of `total`."""
    ftotal = float(total)
    if ftotal == 0.0:
        return 0.0
    return round((float(part) / ftotal) * 100.0, 4)


def _tool_clamp(val: Any, low: Any, high: Any) -> Any:
    """Clamps `val` to the inclusive range [low, high]."""
    return max(low, min(val, high))


def _tool_is_prime(n: Any) -> int:
    """Deterministic primality check returning 1 if prime else 0."""
    val = int(n)
    if val < 2:
        return 0
    if val in (2, 3):
        return 1
    if val % 2 == 0 or val % 3 == 0:
        return 0
    limit = math.isqrt(val)
    i = 5
    while i <= limit:
        if val % i == 0 or val % (i + 2) == 0:
            return 0
        i += 6
    return 1


def _tool_factorial(n: Any) -> int:
    """Deterministic bounded integer factorial (capped at n=20 for safety)."""
    val = int(n)
    if val < 0:
        raise ValueError("factorial() not defined for negative values")
    if val > 20:
        raise ValueError("factorial() input exceeds safe VM bound (20)")
    return math.factorial(val)


def _tool_str_len(text: Any) -> int:
    """Returns character length of string representation."""
    return len(str(text))


def _tool_str_contains(text: Any, sub: Any) -> int:
    """Returns 1 if `sub` is contained in `text`, else 0."""
    return 1 if str(sub) in str(text) else 0


DEFAULT_TOOL_REGISTRY = ToolRegistry()
DEFAULT_TOOL_REGISTRY.register("gcd", 2, _tool_gcd, tool_id=0, description="Greatest common divisor of two integers")
DEFAULT_TOOL_REGISTRY.register("lcm", 2, _tool_lcm, tool_id=1, description="Least common multiple of two integers")
DEFAULT_TOOL_REGISTRY.register("discount", 2, _tool_discount, tool_id=2, description="Price after percentage discount")
DEFAULT_TOOL_REGISTRY.register("vat_add", 2, _tool_vat_add, tool_id=3, description="Amount after adding percentage VAT")
DEFAULT_TOOL_REGISTRY.register("pct_of", 2, _tool_pct_of, tool_id=4, description="Percentage that part represents of total")
DEFAULT_TOOL_REGISTRY.register("clamp", 3, _tool_clamp, tool_id=5, description="Clamp value into [low, high]")
DEFAULT_TOOL_REGISTRY.register("is_prime", 1, _tool_is_prime, tool_id=6, description="1 if integer is prime else 0")
DEFAULT_TOOL_REGISTRY.register("factorial", 1, _tool_factorial, tool_id=7, description="Bounded integer factorial (0..20)")
DEFAULT_TOOL_REGISTRY.register("str_len", 1, _tool_str_len, tool_id=8, description="Character length of string")
DEFAULT_TOOL_REGISTRY.register("str_contains", 2, _tool_str_contains, tool_id=9, description="1 if substring exists in string else 0")
