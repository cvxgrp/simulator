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

from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np
import pandas as pd

from cvx.simulator.interpolation import valid
from cvx.simulator.portfolio import EquityPortfolio
from cvx.simulator.state import State
from cvx.simulator.trading_costs import TradingCostModel


@dataclass
class Builder:
    prices: pd.DataFrame
    trading_cost_model: TradingCostModel = None
    risk_free_rate: pd.Series = None
    borrow_rate: pd.Series = None
    initial_cash: float = 1e6
    input_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        The __post_init__ method is a special method of initialized instances
        of the _Builder class and is called after initialization.
        It sets the initial amount of cash in the portfolio to be equal to the input initial_cash parameter.

        The method takes no input parameter. It initializes the cash attribute in the internal
        _State object with the initial amount of cash in the portfolio, self.initial_cash.

        Note that this method is often used in Python classes for additional initialization routines
        that can only be performed after the object is fully initialized. __post_init__
        is called automatically after the object initialization.
        """

        # assert isinstance(self.prices, pd.DataFrame)
        assert self.prices.index.is_monotonic_increasing
        assert self.prices.index.is_unique

        self.__stocks = pd.DataFrame(
            index=self.prices.index,
            columns=self.prices.columns,
            data=np.NaN,
            dtype=float,
        )

        # self._state.cash = self.initial_cash
        # self._state.model = self.trading_cost_model

        if self.risk_free_rate is None:
            self.risk_free_rate = pd.Series(index=self.index, data=0.0)
        else:
            # We shift the risk_free rate to make sure on day t we access the risk_free rate of day t-1
            self.risk_free_rate = (
                self.risk_free_rate.loc[self.index].shift(1).fillna(0.0)
            )

        if self.borrow_rate is None:
            self.borrow_rate = pd.Series(index=self.index, data=0.0)
        else:
            # We shift the risk_free rate to make sure on day t we access the risk_free rate of day t-1
            self.borrow_rate = self.borrow_rate.loc[self.index].shift(1).fillna(0.0)

        self.__cash = pd.Series(index=self.index, data=np.NaN)
        self.__trading_costs = pd.Series(index=self.index, data=np.NaN)
        self.__cash_interest = pd.Series(index=self.index, data=np.NaN)
        self.__borrow_fees = pd.Series(index=self.index, data=np.NaN)

        self.__state = State()
        self.__state.cash = self.initial_cash
        self.__state.model = self.trading_cost_model

    @property
    def valid(self):
        """
        Analyse the validity of the data
        Do it column by column of the prices
        """
        return self.prices.apply(valid)

    @property
    def intervals(self):
        """
        Find for each column the first and the last valid index
        """
        return self.prices.apply(
            lambda ts: pd.Series(
                {"first": ts.first_valid_index(), "last": ts.last_valid_index()}
            )
        ).transpose()

    @property
    def index(self) -> pd.DatetimeIndex:
        """A property that returns the index of the portfolio,
        which is the time period for which the portfolio data is available.

        Returns: pd.Index: A pandas index representing the
        time period for which the portfolio data is available.

        Notes: The function extracts the index of the prices dataframe,
        which represents the time periods for which data is available for the portfolio.
        The resulting index will be a pandas index object
        with the same length as the number of rows in the prices dataframe."""

        return pd.DatetimeIndex(self.prices.index)

    @property
    def current_prices(self) -> np.array:
        """
        Get the current prices from the state
        """
        return self.__state.prices[self.__state.assets].values

    @property
    def weights(self) -> np.array:
        """
        Get the current weights from the state
        """
        return self.__state.weights[self.__state.assets].values

    @weights.setter
    def weights(self, weights: np.array) -> None:
        self.position = self.__state.nav * weights / self.current_prices

    def __iter__(self) -> Generator[tuple[pd.DatetimeIndex, State], None, None]:
        """
        The __iter__ method allows the object to be iterated over in a for loop,
        yielding time and the current state of the portfolio.
        The method yields a list of dates seen so far
        (excluding the first date) and returns a tuple
        containing the list of dates and the current portfolio state.

        Yield:

        interval: a pandas DatetimeIndex object containing the dates seen so far.

        state: the current state of the portfolio,
        taking into account the stock prices at each interval.
        """
        for t in self.index:
            # valuation of the current position
            self.__state.prices = self.prices.loc[t]
            try:
                self.__state.days = (t - self.__state.time).days
            except TypeError:
                self.__state.days = 0

            self.__state.time = t
            self.__state.risk_free_rate = self.risk_free_rate.loc[t]
            self.__state.borrow_rate = self.borrow_rate.loc[t]

            self.__borrow_fees[self.__state.time] = self.__state.borrow_fees
            self.__cash_interest[self.__state.time] = self.__state.cash_interest

            self.__state.input_data = {
                key: data.loc[t] for key, data in self.input_data.items()
            }

            yield self.index[self.index <= t], self.__state

    @property
    def position(self) -> pd.Series:
        """
        The position property returns the current position of the portfolio.
        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        return self.__stocks.loc[self.__state.time]

    @position.setter
    def position(self, position: pd.Series) -> None:
        """
        The position property returns the current position of the portfolio.
        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        self.__stocks.loc[self.__state.time, self.__state.assets] = position
        self.__state.position = position

        self.__cash[self.__state.time] = self.__state.cash
        self.__trading_costs[self.__state.time] = self.__state.trading_costs.sum()

    @property
    def cash(self):
        return self.__cash

    @property
    def cashposition(self):
        return self.position * self.current_prices

    @property
    def cashflow(self):
        flow = self.cash.diff()
        flow.iloc[1] = self.cash.iloc[0] - self.initial_cash
        return flow

    @property
    def stocks(self):
        return self.__stocks

    @cashposition.setter
    def cashposition(self, cashposition: pd.Series) -> None:
        self.position = cashposition / self.current_prices

    def build(self) -> EquityPortfolio:
        """A function that creates a new instance of the EquityPortfolio
        class based on the internal state of the Portfolio builder object.

        Returns: EquityPortfolio: A new instance of the EquityPortfolio class
        with the attributes (prices, stocks, initial_cash, trading_cost_model) as specified in the Portfolio builder.

        Notes: The function simply creates a new instance of the EquityPortfolio
        class with the attributes (prices, stocks, initial_cash, trading_cost_model) equal
        to the corresponding attributes in the Portfolio builder object.
        The resulting EquityPortfolio object will have the same state as the Portfolio builder from which it was built.
        """

        portfolio = EquityPortfolio(
            prices=self.prices,
            stocks=self.stocks,
            cash=self.cash,
            trading_costs=self.__trading_costs,
            borrow_fees=self.__borrow_fees,
            borrow_rate=self.borrow_rate,
            risk_free_rate=self.risk_free_rate,
            cash_interest=self.__cash_interest,
            flow=self.cashflow,
        )

        return portfolio
