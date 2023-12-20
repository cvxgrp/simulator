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
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass()
class State:
    _prices: pd.Series = None
    _position: pd.Series = None
    _trades: pd.Series = None
    _time: datetime = None
    _days: int = 0
    _profit: float = 0.0
    _aum: float = 0.0

    @property
    def cash(self):
        return self.nav - self.value

    @cash.setter
    def cash(self, cash: float):
        # self._cash = cash
        self.aum = cash + self.value

    @property
    def nav(self) -> float:
        """
        The nav property computes the net asset value (NAV) of the portfolio,
        which is the sum of the current value of the
        portfolio as determined by the value property,
        and the current amount of cash available in the portfolio.
        """
        # assert np.isclose(self.value + self.cash, self.aum), f"{self.value + self.cash} != {self.aum}"
        # return self.value + self.cash
        return self.aum

    @property
    def value(self) -> float:
        """
        The value property computes the value of the portfolio at the current
        time taking into account the current holdings and current prices.
        If the value cannot be computed due to missing positions
        (they might be still None), zero is returned instead.
        """
        return self.cashposition.sum()

    @property
    def cashposition(self):
        """
        The `cashposition` property computes the cash position of the portfolio,
        which is the amount of cash in the portfolio as a fraction of the total portfolio value.
        """
        return self.prices * self.position

    @property
    def position(self):
        if self._position is None:
            return pd.Series(dtype=float)

        return self._position

    @position.setter
    def position(self, position: np.array):
        """
        Update the position of the state. Computes the required trades
        and updates other quantities (e.g. cash) accordingly.
        """
        # update the position
        position = pd.Series(index=self.assets, data=position)

        # compute the trades (can be fractional)
        self._trades = position.subtract(self.position, fill_value=0.0)

        # update only now as otherwise the trades would be wrong
        self._position = position

    @property
    def gmv(self):
        """
        gross market value, e.g. abs(short) + long
        """
        return self.cashposition.abs().sum()

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time: datetime):
        if self.time is None:
            self._days = 0
            self._time = time
        else:
            self._days = (time - self.time).days
            self._time = time

    @property
    def days(self):
        return self._days

    @property
    def assets(self) -> pd.Index:
        """
        The assets property returns the assets currently in the portfolio.
        """
        return self.prices.dropna().index

    @property
    def trades(self):
        """
        The trades property returns the trades currently needed to reach the position.
        Most helpful when computing the trading costs following the move to
        a new position.
        """
        return self._trades

    @property
    def mask(self):
        """construct true/false mask for assets with missing prices"""
        return np.isfinite(self.prices.values)

    @property
    def prices(self):
        return self._prices

    @prices.setter
    def prices(self, prices):
        value_before = (self.prices * self.position).sum()  # self.cashposition.sum()
        value_after = (prices * self.position).sum()

        self._prices = prices
        self._profit = value_after - value_before
        self.aum += self.profit

    @property
    def profit(self):
        return self._profit

    @property
    def aum(self):
        return self._aum

    @aum.setter
    def aum(self, aum):
        self._aum = aum

    @property
    def weights(self) -> pd.Series:
        """
        The weights property computes the weighting of each asset in the current
        portfolio as a fraction of the total portfolio value (nav).

        Returns:

        a pandas series object containing the weighting of each asset as a
        fraction of the total portfolio value. If the positions are still
        missing, then a series of zeroes is returned.
        """
        assert np.isclose(self.nav, self.aum), f"{self.nav} != {self.aum}"
        return self.cashposition / self.nav

    @property
    def leverage(self) -> float:
        """
        The `leverage` property computes the leverage of the portfolio,
        which is the sum of the absolute values of the portfolio weights.
        """
        return float(self.weights.abs().sum())
