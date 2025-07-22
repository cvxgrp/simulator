"""A simple yet powerful simulator for investment strategies and portfolio backtesting.

This package provides tools for creating and analyzing investment portfolios,
implementing trading strategies, and backtesting them against historical data.
It simplifies accounting by tracking cash, positions, profits, and other metrics.
"""

import importlib.metadata

__version__ = importlib.metadata.version("cvxsimulator")

from .builder import Builder
from .portfolio import Portfolio
from .state import State
from .utils.interpolation import interpolate, valid

__all__ = ["Builder", "Portfolio", "State", "interpolate", "valid"]
