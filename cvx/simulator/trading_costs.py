# -*- coding: utf-8 -*-
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TradingCostModel(abc.ABC):
    @abc.abstractmethod
    def eval(
        self, prices: pd.DataFrame, trades: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
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

    def eval(
        self, prices: pd.DataFrame, trades: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        volume = prices * trades
        return self.factor * volume.abs() + self.bias * trades.abs()
