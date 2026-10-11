"""
Legacy VM Module (Deprecated)
=============================
Maintained for backward compatibility. Please import from runtime.vm instead.
"""

from runtime.vm.ministack_vm import MiniStackVM, RuntimeExecutionError

__all__ = ["MiniStackVM", "RuntimeExecutionError"]
