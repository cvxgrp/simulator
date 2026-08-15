"""Public API of the cvxsimulator distribution, importable as `cvx.simulator`.

The package exposes the three objects a backtest is built from:

- `Builder` accumulates positions while iterating over a frame of prices;
- `Portfolio` holds the finished result and the analytics derived from it;
- `State` is the view of the book at a single point in time, handed to the
  caller on each step of the builder's loop.

Alongside them sit the `interpolate` and `valid` price helpers from
`cvx.simulator.utils`. Submodules stay importable directly, so
`from cvx.simulator.builder import Builder` reaches the same class as
`from cvx.simulator import Builder`.

`__version__` is resolved at import time from the installed distribution
metadata of `cvxsimulator`.
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
    "__version__",
    "interpolate",
    "valid",
]
