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
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TradingCostModel(abc.ABC):
    @abc.abstractmethod
    def eval(self, prices: pd.Series, trades: pd.Series, **kwargs: Any) -> pd.Series:
        """Evaluates the cost of a trade given the prices and the trades

        Arguments
            prices: the price per asset
            trades: the trade per asset, e.g. number of stocks traded
            **kwargs: additional arguments, e.g. volatility, liquidity, spread, etc.
        """


@dataclass(frozen=True)
class LinearCostModel(TradingCostModel):
    factor: float = 0.0
    bias: float = 0.0

    def eval(self, prices: pd.Series, trades: pd.Series, **kwargs: Any) -> pd.DataFrame:
        volume = prices * trades
        return self.factor * volume.abs() + self.bias * trades.abs()
