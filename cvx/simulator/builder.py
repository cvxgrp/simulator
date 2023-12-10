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
from datetime import datetime
from typing import Any, Generator

import numpy as np
import pandas as pd

from cvx.simulator.interpolation import valid
from cvx.simulator.portfolio import EquityPortfolio
from cvx.simulator.trading_costs import TradingCostModel


@dataclass
class _State:
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
    risk_free_rate: float = 0.0
    cash: float = 1e6
    input_data: dict[str, Any] = field(default_factory=dict)
    model: TradingCostModel | None = None
    time: datetime | None = None
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
    def position(self):
        if self.__position is None:
            return pd.Series(dtype=float)

        return self.__position

    @position.setter
    def position(self, position: np.array):
        position = pd.Series(index=self.assets, data=position)
        trades = self._trade(target_pos=position)

        self.__position = position
        self.cash -= (trades * self.prices).sum()

        if self.model is not None:
            self.cash -= self.model.eval(self.prices, trades=trades).sum()

        # update the cash using the risk-free interest rate
        # Note the the risk_free_rate is shifted
        # e.g. we update our cash using the old risk_free_rate
        self.cash = self.cash * (self.risk_free_rate + 1) ** self.days

    def __getattr__(self, item):
        return self.input_data[item]

    @property
    def assets(self):
        return self.prices.dropna().index

    def _trade(self, target_pos):
        """
        Compute the trade vector given a target position
        """
        return target_pos.subtract(self.position, fill_value=0.0)


def builder(
    prices: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    initial_cash: float = 1e6,
    trading_cost_model: TradingCostModel | None = None,
    risk_free_rate: pd.Series | None = None,
    **kwargs,
) -> _Builder:
    """The builder function creates an instance of the _Builder class, which
    is used to construct a portfolio of assets. The function takes in a pandas
    DataFrame of historical prices for the assets in the portfolio, optional
    weights for each asset, an initial cash value, and a trading cost model.
    The function first asserts that the prices DataFrame has a monotonic
    increasing and unique index. It then creates a DataFrame of zeros to hold
    the number of shares of each asset owned at each time step. The function
    initializes a _Builder object with the stocks DataFrame, the prices
    DataFrame (forward-filled), the initial cash value, and the trading cost
    model. If weights are provided, they are set for each time step using
    set_weights method of the _Builder object. The final output is the
    constructed _Builder object."""

    assert isinstance(prices, pd.DataFrame)
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique

    stocks = pd.DataFrame(
        index=prices.index, columns=prices.columns, data=np.NaN, dtype=float
    )

    builder = _Builder(
        stocks=stocks,
        prices=prices,
        initial_cash=float(initial_cash),
        trading_cost_model=trading_cost_model,
        input_data=dict(kwargs),
        risk_free_rate=risk_free_rate,
    )

    if weights is not None:
        for t, state in builder:
            builder.weights = weights[state.assets].loc[t[-1]].dropna().values

    return builder


@dataclass
class _Builder:
    prices: pd.DataFrame
    stocks: pd.DataFrame
    trading_cost_model: TradingCostModel | None = None
    risk_free_rate: pd.Series | None = None
    initial_cash: float = 1e6
    _state: _State = field(default_factory=_State)
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
        self._state.cash = self.initial_cash
        self._state.model = self.trading_cost_model
        if self.risk_free_rate is None:
            self.risk_free_rate = pd.Series(index=self.index, data=0.0)
        else:
            # We shift the risk_free rate to make sure on day t we access the risk_free rate of day t-1
            r = self.risk_free_rate.loc[self.index].shift(1).fillna(0.0)
            self.risk_free_rate = r

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
        return self._state.prices[self._state.assets].values

    @property
    def weights(self) -> np.array:
        """
        Get the current weights from the state
        """
        return self._state.weights[self._state.assets].values

    @weights.setter
    def weights(self, weights: np.array) -> None:
        self.position = self._state.nav * weights / self.current_prices

    def __iter__(self) -> Generator[tuple[pd.DatetimeIndex, _State], None, None]:
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
            self._state.prices = self.prices.loc[t]
            try:
                self._state.days = (t - self._state.time).days
            except TypeError:
                self._state.days = 0

            self._state.time = t
            self._state.risk_free_rate = self.risk_free_rate.loc[t]

            self._state.input_data = {
                key: data.loc[t] for key, data in self.input_data.items()
            }

            yield self.index[self.index <= t], self._state

    @property
    def position(self) -> pd.Series:
        """
        The position property returns the current position of the portfolio.
        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        return self.stocks.loc[self._state.time]

    @position.setter
    def position(self, position: pd.Series) -> None:
        """
        The position property returns the current position of the portfolio.
        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        self.stocks.loc[self._state.time, self._state.assets] = position
        self._state.position = position

    @property
    def cashposition(self):
        return self.position * self.current_prices

    @cashposition.setter
    def cashposition(self, cashposition: pd.Series) -> None:
        self.position = cashposition / self.current_prices

    def build(self, extra=0) -> EquityPortfolio:
        """A function that creates a new instance of the EquityPortfolio
        class based on the internal state of the Portfolio builder object.

        Returns: EquityPortfolio: A new instance of the EquityPortfolio class
        with the attributes (prices, stocks, initial_cash, trading_cost_model) as specified in the Portfolio builder.

        Notes: The function simply creates a new instance of the EquityPortfolio
        class with the attributes (prices, stocks, initial_cash, trading_cost_model) equal
        to the corresponding attributes in the Portfolio builder object.
        The resulting EquityPortfolio object will have the same state as the Portfolio builder from which it was built.
        """

        return EquityPortfolio(
            prices=self.prices,
            stocks=self.stocks,
            initial_cash=self.initial_cash,  # - (self.prices.iloc[0]*self.stocks.iloc[0]).sum(),
            trading_cost_model=self.trading_cost_model,
        )
