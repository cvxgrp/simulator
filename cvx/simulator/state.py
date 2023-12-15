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


@dataclass
class State:
    """The _State class defines a state object used to keep track of the current
    state of the portfolio.

    Attributes:

    prices: a pandas Series object containing the stock prices of the current
    portfolio state

    position: a pandas Series object containing the current holdings of the portfolio

    cash: the amount of cash available in the portfolio.

    time: the current time of the portfolio

    By default, prices and position are set to None, while cash is set to 1 million.
    These attributes can be updated and accessed through setter and getter methods
    """

    prices: pd.Series = None
    __position: pd.Series = None
    __trades: pd.Series = None
    cash: float = 1e6
    __time: datetime = None
    __days: int = 0

    @property
    def value(self) -> float:
        """
        The value property computes the value of the portfolio at the current
        time taking into account the current holdings and current stock prices.
        If the value cannot be computed due to missing positions
        (they might be still None), zero is returned instead.
        """
        return self.cashposition.sum()

    @property
    def nav(self) -> float:
        """
        The nav property computes the net asset value (NAV) of the portfolio,
        which is the sum of the current value of the
        portfolio as determined by the value property,
        and the current amount of cash available in the portfolio.
        """
        return self.value + self.cash

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
        return self.cashposition / self.nav

    @property
    def leverage(self) -> float:
        """
        The `leverage` property computes the leverage of the portfolio,
        which is the sum of the absolute values of the portfolio weights.
        """
        return float(self.weights.abs().sum())

    @property
    def cashposition(self):
        """
        The `cashposition` property computes the cash position of the portfolio,
        which is the amount of cash in the portfolio as a fraction of the total portfolio value.
        """
        return self.prices * self.position

    @property
    def short(self):
        """
        How short are we at this stage in USD
        """
        return self.cashposition[self.cashposition < 0].sum()

    @property
    def position(self):
        if self.__position is None:
            return pd.Series(dtype=float)

        return self.__position

    @property
    def gmv(self):
        """
        gross market value, e.g. abs(short) + long
        """
        return self.cashposition.abs().sum()

    @position.setter
    def position(self, position: np.array):
        """
        Update the position of the state. Computes the required trades
        and updates the cash balance accordingly.
        """
        # update the position
        position = pd.Series(index=self.assets, data=position)

        # compute the trades (can be fractional)
        self.__trades = position.subtract(self.position, fill_value=0.0)

        # update only now as otherwise the trades would be wrong
        self.__position = position

        # cash is spent for shares or received for selling them
        self.cash -= self.gross.sum()

    @property
    def time(self):
        return self.__time

    @time.setter
    def time(self, time: datetime):
        if self.time is None:
            self.__days = 0
            self.__time = time
        else:
            self.__days = (time - self.time).days
            self.__time = time

    @property
    def days(self):
        return self.__days

    @property
    def trades(self):
        """
        The trades property returns the trades currently needed to reach the position.
        Most helpful when computing the trading costs following the move to
        a new position.
        """
        return self.__trades

    @property
    def gross(self):
        return self.trades * self.prices

    @property
    def assets(self) -> pd.Index:
        """
        The assets property returns the assets currently in the portfolio.
        """
        return self.prices.dropna().index
