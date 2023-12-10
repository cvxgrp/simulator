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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from cvx.simulator.trading_costs import TradingCostModel


@dataclass
class State:
    """The _State class defines a state object used to keep track of the current
    state of the portfolio.

    Attributes:

    prices: a pandas Series object containing the stock prices of the current
    portfolio state

    position: a pandas Series object containing the current holdings of the portfolio

    cash: the amount of cash available in the portfolio.

    By default, prices and position are set to None, while cash is set to 1 million.
    These attributes can be updated and accessed through setter and getter methods
    """

    prices: pd.Series = None
    __position: pd.Series = None
    __trades: pd.Series = None
    __trading_costs: pd.Series = None
    risk_free_rate: float = 0.0
    borrow_rate: float = 0.0
    cash: float = 1e6
    input_data: dict[str, Any] = field(default_factory=dict)
    model: TradingCostModel = None
    time: datetime = None
    days: int = 1

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
    def cash_interest(self):
        """
        How much interest do we get for the cash
        """
        return self.cash * (self.risk_free_rate + 1) ** self.days - self.cash

    @property
    def borrow_fees(self):
        """
        How much interest do we pay for the short position
        """
        return self.short * (self.borrow_rate + 1) ** self.days - self.short

    @position.setter
    def position(self, position: np.array):
        # update the cash first using the risk-free interest rate
        # Note the risk_free_rate is shifted
        # e.g. we update our cash using the old risk_free_rate
        for fee in [self.cash_interest, self.borrow_fees]:
            self.cash += fee

        # update the position
        position = pd.Series(index=self.assets, data=position)

        # compute the trades (can be fractional)
        self.__trades = self._trade(target_pos=position)

        # update only now as otherwise the trades would be wrong
        self.__position = position

        # cash is spent for shares or received for selling them
        self.cash -= self.gross.sum()

        if self.model is not None:
            self.__trading_costs = self.model.eval(
                self.prices, trades=self.trades, **self.input_data
            ).sum()

            self.cash -= self.trading_costs.sum()

    def __getattr__(self, item):
        return self.input_data[item]

    @property
    def trading_costs(self):
        return self.__trading_costs

    @property
    def trades(self):
        return self.__trades

    @property
    def gross(self):
        return self.trades * self.prices

    @property
    def assets(self) -> pd.Index:
        return self.prices.dropna().index

    def _trade(self, target_pos: pd.Series) -> pd.Series:
        """
        Compute the trade vector given a target position
        """
        return target_pos.subtract(self.position, fill_value=0.0)
