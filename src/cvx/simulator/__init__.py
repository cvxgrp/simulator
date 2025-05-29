#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""
CVX Simulator: A simple simulator for investors.

This package provides tools for simulating investment portfolios, tracking
positions, calculating returns, and analyzing performance. It allows users
to backtest investment strategies using historical price data.

Main components:
- Builder: Creates portfolios by iterating through time and setting positions
- Portfolio: Represents a portfolio of assets with methods for analysis
- State: Represents the current state of a portfolio during simulation
- interpolate: Utility function for interpolating missing values in time series
"""
from importlib.metadata import version
__version__ = version("cvxsimulator")

from .builder import Builder
from .portfolio import Portfolio
from .state import State
from .utils.interpolation import interpolate

__all__ = [
    "Builder",
    "Portfolio",
    "State",
    "interpolate"
]
