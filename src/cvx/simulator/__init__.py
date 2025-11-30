"""Compatibility shim for cvx.simulator.

This subpackage provides a stable import path `cvx.simulator` for tests and
examples. It re-exports the primary public API from the parent `cvx` package
and exposes submodules so that imports like `from cvx.simulator.builder import ...`
continue to work.
"""

import importlib.metadata

__version__ = importlib.metadata.version("cvxsimulator")

# Explicit re-exports to satisfy linters (ruff F401)
from .builder import Builder as Builder
from .portfolio import Portfolio as Portfolio
from .state import State as State
from .utils import interpolate as interpolate
from .utils import valid as valid

__all__ = [
    "Builder",
    "Portfolio",
    "State",
    "interpolate",
    "valid",
    "__version__",
]
