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
from typing import Generator

import numpy as np
import pandas as pd

from cvx.simulator.portfolio import Portfolio
from cvx.simulator.state import State
from cvx.simulator.utils.interpolation import valid
from cvx.simulator.utils.rescale import returns2prices


@dataclass
class Builder:
    """
    The Builder is an auxiliary class used to build portfolios.
    It overloads the __iter__ method to allow the class to iterate over
    the timestamps for which the portfolio data is available.

    In each iteration we can update the portfolio by setting either
    the weights, the position or the cash position.

    After the iteration has been completed we build a Portfolio object
    by calling the build method.
    """

    prices: pd.DataFrame
    initial_aum: float = 1e6

    _state: State = None
    _units: pd.DataFrame = None
    _aum: pd.Series = None

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

        self._state = State()

        self._units = pd.DataFrame(
            index=self.prices.index,
            columns=self.prices.columns,
            data=np.NaN,
            dtype=float,
        )

        self._aum = pd.Series(index=self.prices.index, dtype=float)

        self._state.aum = self.initial_aum

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
        which are the timestamps for which the portfolio data is available.

        Returns: pd.Index: A pandas index representing the
        time period for which the portfolio data is available.
        """
        return pd.DatetimeIndex(self.prices.index)

    @property
    def current_prices(self) -> np.array:
        """
        Get the current prices from the state
        """
        return self._state.prices[self._state.assets].values

    def __iter__(self) -> Generator[tuple[pd.DatetimeIndex, State], None, None]:
        """
        The __iter__ method allows the object to be iterated over in a for loop,
        yielding time and the current state of the portfolio.
        The method yields a list of dates seen so far and returns a tuple
        containing the list of dates and the current portfolio state.

        Yield:

        time: a pandas DatetimeIndex object containing the dates seen so far.
        state: the current state of the portfolio,

        taking into account the stock prices at each interval.
        """
        for t in self.index:
            # update the current prices for the portfolio
            self._state.prices = self.prices.loc[t]

            # update the current time for the state
            self._state.time = t

            # yield the vector of times seen so far and the current state
            yield self.index[self.index <= t], self._state

    @property
    def position(self) -> pd.Series:
        """
        The position property returns the current position of the portfolio.
        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        return self._units.loc[self._state.time]

    @position.setter
    def position(self, position: pd.Series) -> None:
        """
        The position property returns the current position of the portfolio.
        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        self._units.loc[self._state.time, self._state.assets] = position
        self._state.position = position

    @property
    def cashposition(self):
        """
        The cashposition property returns the current cash position of the portfolio.
        """
        return self.position * self.current_prices

    @property
    def units(self):
        """
        The units property returns the frame of holdings of the portfolio.
        Useful mainly for testing
        """
        return self._units

    @cashposition.setter
    def cashposition(self, cashposition: pd.Series) -> None:
        """
        The cashposition property sets the current cash position of the portfolio.
        """
        self.position = cashposition / self.current_prices

    def build(self):
        """A function that creates a new instance of the EquityPortfolio
        class based on the internal state of the Portfolio builder object.

        Returns: EquityPortfolio: A new instance of the EquityPortfolio class
        with the attributes (prices, units, initial_cash, trading_cost_model) as specified in the Portfolio builder.

        Notes: The function simply creates a new instance of the EquityPortfolio
        class with the attributes (prices, units, initial_cash, trading_cost_model) equal
        to the corresponding attributes in the Portfolio builder object.
        The resulting EquityPortfolio object will have the same state as the Portfolio builder from which it was built.
        """

        return Portfolio(prices=self.prices, units=self.units, aum=self.aum)

    @property
    def weights(self) -> np.array:
        """
        Get the current weights from the state
        """
        return self._state.weights[self._state.assets].values

    @weights.setter
    def weights(self, weights: np.array) -> None:
        """
        The weights property sets the current weights of the portfolio.
        We convert the weights to positions using the current prices and the NAV
        """
        self.position = self._state.nav * weights / self.current_prices

    @property
    def aum(self):
        """
        The aum property returns the current AUM of the portfolio.
        """
        return self._aum

    @aum.setter
    def aum(self, aum):
        """
        The aum property sets the current AUM of the portfolio.
        """
        self._aum[self._state.time] = aum
        self._state.aum = aum

    @classmethod
    def from_returns(cls, returns):
        """Build Futures Portfolio from returns"""
        # compute artificial prices (but scaled such their returns are correct)

        prices = returns2prices(returns)
        return cls(prices=prices)
