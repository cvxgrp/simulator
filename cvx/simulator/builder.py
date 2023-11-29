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
    _position: pd.Series = None
    cash: float = 1e6
    input_data: dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """
        The value property computes the value of the portfolio at the current
        time taking into account the current holdings and current stock prices.
        If the value cannot be computed due to missing positions
        (they might be still None), zero is returned instead.
        """
        try:
            return float((self.prices * self._position).sum())
        except TypeError:
            return 0.0

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
        try:
            return (self.prices * self._position) / self.nav
        except TypeError:
            return 0 * self.prices

    @property
    def leverage(self) -> float:
        """
        The `leverage` property computes the leverage of the portfolio,
        which is the sum of the absolute values of the portfolio weights.
        """
        return float(self.weights.abs().sum())

    def update(
        self,
        position: pd.Series,
        model: TradingCostModel | None = None,
        **kwargs: Any,
    ) -> _State:
        """
        The update method updates the current state of the portfolio with the
        new input position. It calculates the trades made based on the new
        and the previous position, updates the internal position and
        cash attributes, and applies any trading costs according to a model parameter.

        The method takes three input parameters:

        position: a pandas series object representing the new position of the
        portfolio.

        model: an optional trading cost model (e.g. slippage, fees) to be
        incorporated into the update. If None, no trading costs will be applied.

        **kwargs: additional keyword arguments to pass into the trading cost
        model.

        Returns:
        self: the _State instance with the updated position and cash.

        Updates:

        trades: the difference between positions in the old and new portfolio.
        position: the new position of the portfolio.
        cash: the new amount of cash in the portfolio after any trades and trading costs are applied.

        Note that the method does not return any value: instead,
        it updates the internal state of the _State instance.
        """
        trades = self.trade(target_pos=position)

        self._position = position
        self.cash -= (trades * self.prices).sum()

        if model is not None:
            self.cash -= model.eval(self.prices, trades=trades, **kwargs).sum()

        # builder is frozen, so we can't construct a new state
        return self

    def __getattr__(self, item):
        return self.input_data[item]

    @property
    def assets(self):
        return self.prices.dropna().index

    def trade(self, target_pos):
        """
        Compute the trade vector given a target position
        """
        if self._position is None:
            return target_pos

        return target_pos.subtract(self._position, fill_value=0.0)


def builder(
    prices: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    initial_cash: float = 1e6,
    trading_cost_model: TradingCostModel | None = None,
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
    )

    if weights is not None:
        for t, state in builder:
            builder.set_weights(time=t[-1], weights=weights.loc[t[-1]])

    return builder


@dataclass(frozen=True)
class _Builder:
    prices: pd.DataFrame
    stocks: pd.DataFrame
    trading_cost_model: TradingCostModel | None = None
    initial_cash: float = 1e6
    _state: _State = field(default_factory=_State)
    market_cap: pd.DataFrame = None
    trade_volume: pd.DataFrame = None
    max_cap_fraction: float | None = None
    min_cap_fraction: float | None = None
    max_trade_fraction: float | None = None
    min_trade_fraction: float | None = None
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

    def set_weights(self, time: datetime, weights: pd.Series) -> None:
        """
        Set the position via weights (e.g. fractions of the nav)

        :param time: time
        :param weights: series of weights
        """
        assert isinstance(weights, pd.Series), "weights must be a pandas Series"
        valid = self._state.prices.dropna().index
        # check that you have weights exactly for those indices
        if not set(weights.dropna().index) == set(valid):
            raise ValueError("weights must have same index as prices")

        self[time] = (self._state.nav * weights[valid]) / self._state.prices[valid]

    def set_cashposition(self, time: datetime, cashposition: pd.Series) -> None:
        """
        Set the position via cash positions (e.g. USD invested per asset)

        :param time: time
        :param cashposition: series of cash positions
        """
        assert isinstance(
            cashposition, pd.Series
        ), "cashposition must be a pandas Series"

        valid = self._state.prices.dropna().index
        # check that you have weights exactly for those indices
        if not set(cashposition.dropna().index) == set(valid):
            raise ValueError("cashposition must have same index as prices")

        self[time] = cashposition[valid] / self._state.prices[valid]

    def set_position(self, time: datetime, position: pd.Series) -> None:
        """
        Set the position via number of assets (e.g. number of stocks)

        :param time: time
        :param position: series of number of stocks
        """
        assert isinstance(position, pd.Series), "position must be a pandas Series"

        valid = self._state.prices.dropna().index
        # check that you have weights exactly for those indices
        if not set(position.dropna().index) == set(valid):
            raise ValueError("position must have same index as prices")

        self[time] = position

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

            self._state.input_data = {
                key: data.loc[t] for key, data in self.input_data.items()
            }

            yield self.index[self.index <= t], self._state

    def __setitem__(self, time: datetime, position: pd.Series) -> None:
        """
        The method __setitem__ updates the stock data in the dataframe for a specific time index
        with the input position. It first checks that position is a valid input,
        meaning it is a pandas Series object and has its index within the assets of the dataframe.
        The method takes two input parameters:

        time: time index for which to update the stock data
        position: pandas series object containing the updated stock data

        Returns: None

        Updates:
        the stock data of the dataframe at the given time index with the input position
        the internal state of the portfolio with the updated position, taking into account the trading cost model

        Raises:
        AssertionError: if the input position is not a pandas Series object
        or its index is not a subset of the assets of the dataframe.
        """
        assert isinstance(position, pd.Series)

        valid = self._state.prices.dropna().index
        # check that you have weights exactly for those indices
        if not set(position.dropna().index) == set(valid):
            raise ValueError("position must have same index as prices")

        self.stocks.loc[time, position.index] = position
        self._state.update(position, model=self.trading_cost_model)

    def __getitem__(self, time: datetime) -> pd.Series:
        """The __getitem__ method retrieves the stock data for a specific time in the dataframe.
        It returns the stock data for that time. The method takes one input parameter:

        time: the time index for which to retrieve the stock data
        Returns: stock data for the input time

        Note that the input time must be in the index of the dataframe, otherwise a KeyError will be raised.
        """
        return self.stocks.loc[time]

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

        return EquityPortfolio(
            prices=self.prices,
            stocks=self.stocks,
            initial_cash=self.initial_cash,
            trading_cost_model=self.trading_cost_model,
        )
