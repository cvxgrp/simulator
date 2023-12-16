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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass()
class State(ABC):
    prices: pd.Series = None

    _position: pd.Series = None
    _trades: pd.Series = None
    _time: datetime = None
    _days: int = 0

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
    @abstractmethod
    def position(self, position: np.array):
        """
        Update the position of the state. Computes the required trades
        and updates other quantities (e.g. cash) accordingly.
        """

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
    def gross(self):
        return self.trades * self.prices
