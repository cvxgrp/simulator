"""Compatibility shim for cvx.simulator

This subpackage provides a stable import path `cvx.simulator` for tests and
examples. It re-exports the primary public API from the parent `cvx` package
and exposes submodules so that imports like `from cvx.simulator.builder import ...`
continue to work.
"""
import importlib.metadata

__version__ = importlib.metadata.version("cvxsimulator")

from .utils import valid, interpolate
from .state import State
from .builder import Builder
from .portfolio import Portfolio
