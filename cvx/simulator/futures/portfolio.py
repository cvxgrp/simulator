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

from dataclasses import dataclass

import pandas as pd

from cvx.simulator.utils.rescale import returns2prices

from .._abc.portfolio import Portfolio


@dataclass(frozen=True)
class FuturesPortfolio(Portfolio):
    aum: float

    @property
    def nav(self):
        profit = (self.cashposition.shift(1) * self.returns.fillna(0.0)).sum(axis=1)

        return profit.cumsum() + self.aum

    @classmethod
    def from_cashpos_prices(
        cls, prices: pd.DataFrame, cashposition: pd.DataFrame, aum: float
    ):
        """Build Futures Portfolio from cashposition"""
        units = cashposition.div(prices, fill_value=0.0)
        return cls(prices=prices, units=units, aum=aum)

    @classmethod
    def from_cashpos_returns(
        cls, returns: pd.DataFrame, cashposition: pd.DataFrame, aum: float
    ):
        """Build Futures Portfolio from cashposition"""
        prices = returns2prices(returns)
        return cls.from_cashpos_prices(prices, cashposition, aum)

        # units = cashposition.div(prices, fill_value=0.0)
        # return cls(prices=prices, units=units, aum=aum)
