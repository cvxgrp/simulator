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
"""Portfolio state management for the CVX Simulator.

This module provides the State class, which represents the current state of a portfolio
during simulation. It tracks positions, prices, cash, and other portfolio metrics,
and is updated by the Builder class during the simulation process.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass()
class State:
    """Represents the current state of a portfolio during simulation.

    The State class tracks the current positions, prices, cash, and other metrics
    of a portfolio at a specific point in time. It is updated within a loop by the
    Builder class during the simulation process.

    The class provides properties for accessing various portfolio metrics like
    cash, NAV, value, weights, and leverage. It also provides setter methods
    for updating the portfolio state (aum, cash, position, prices).

    Attributes:
    ----------
    _prices : pd.Series
        Current prices of assets in the portfolio
    _position : pd.Series
        Current positions (units) of assets in the portfolio
    _trades : pd.Series
        Trades needed to reach the current position
    _time : datetime
        Current time in the simulation
    _days : int
        Number of days between the current and previous time
    _profit : float
        Profit achieved between the previous and current prices
    _aum : float
        Current assets under management (AUM) of the portfolio

    """

    _prices: pd.Series | None = None
    _position: pd.Series | None = None
    _trades: pd.Series | None = None
    _time: datetime | None = None
    _days: int = 0
    _profit: float = 0.0
    _aum: float = 0.0

    @property
    def cash(self) -> float:
        """Get the current amount of cash available in the portfolio.

        Returns:
        -------
        float
            The cash component of the portfolio, calculated as NAV minus
            the value of all positions

        """
        return self.nav - self.value

    @cash.setter
    def cash(self, cash: float) -> None:
        """Update the amount of cash available in the portfolio.

        This updates the AUM (assets under management) based on the new
        cash amount while keeping the value of positions constant.

        Parameters
        ----------
        cash : float
            The new cash amount to set

        """
        self.aum = cash + self.value

    @property
    def nav(self) -> float:
        """Get the net asset value (NAV) of the portfolio.

        The NAV represents the total value of the portfolio, including
        both the value of positions and available cash.

        Returns:
        -------
        float
            The net asset value of the portfolio

        Notes:
        -----
        This is equivalent to the AUM (assets under management).

        """
        # assert np.isclose(self.value + self.cash, self.aum), f"{self.value + self.cash} != {self.aum}"
        # return self.value + self.cash
        return self.aum

    @property
    def value(self) -> float:
        """Get the value of all positions in the portfolio.

        This computes the total value of all holdings at current prices,
        not including cash.

        Returns:
        -------
        float
            The sum of values of all positions

        Notes:
        -----
        If positions are missing (None), the sum will effectively be zero.

        """
        return self.cashposition.sum()

    @property
    def cashposition(self) -> pd.Series:
        """Get the cash value of each position in the portfolio.

        This computes the cash value of each position by multiplying
        the number of units by the current price for each asset.

        Returns:
        -------
        pd.Series
            Series with the cash value of each position, indexed by asset

        """
        return self.prices * self.position

    @property
    def position(self) -> pd.Series:
        """Get the current position (number of units) for each asset.

        Returns:
        -------
        pd.Series
            Series with the number of units held for each asset, indexed by asset.
            If the position is not yet set, returns an empty series with the
            correct index.

        """
        if self._position is None:
            return pd.Series(index=self.assets, dtype=float)

        return self._position

    @position.setter
    def position(self, position: np.ndarray | pd.Series) -> None:
        """Update the position of the portfolio.

        This method updates the position (number of units) for each asset,
        computes the required trades to reach the new position, and updates
        the internal state.

        Parameters
        ----------
        position : Union[np.ndarray, pd.Series]
            The new position to set, either as a numpy array or pandas Series.
            If a numpy array, it must have the same length as self.assets.

        """
        # update the position
        position = pd.Series(index=self.assets, data=position)

        # compute the trades (can be fractional)
        self._trades = position.subtract(self.position, fill_value=0.0)

        # update only now as otherwise the trades would be wrong
        self._position = position

    @property
    def gmv(self) -> float:
        """Get the gross market value of the portfolio.

        The gross market value is the sum of the absolute values of all positions,
        which represents the total market exposure including both long and short positions.

        Returns:
        -------
        float
            The gross market value (abs(short) + long)

        """
        return self.cashposition.abs().sum()

    @property
    def time(self) -> datetime | None:
        """Get the current time of the portfolio state.

        Returns:
        -------
        Optional[datetime]
            The current time in the simulation, or None if not set

        """
        return self._time

    @time.setter
    def time(self, time: datetime) -> None:
        """Update the time of the portfolio state.

        This method updates the current time and computes the number of days
        between the new time and the previous time.

        Parameters
        ----------
        time : datetime
            The new time to set

        """
        if self.time is None:
            self._days = 0
            self._time = time
        else:
            self._days = (time - self.time).days
            self._time = time

    @property
    def days(self) -> int:
        """Get the number of days between the current and previous time.

        Returns:
        -------
        int
            Number of days between the current and previous time

        Notes:
        -----
        This is useful for computing interest when holding cash or for
        time-dependent calculations.

        """
        return self._days

    @property
    def assets(self) -> pd.Index:
        """Get the assets currently in the portfolio.

        Returns:
        -------
        pd.Index
            Index of assets with valid prices in the portfolio.
            If no prices are set, returns an empty index.

        """
        if self._prices is None:
            return pd.Index(data=[], dtype=str)

        return self.prices.dropna().index

    @property
    def trades(self) -> pd.Series | None:
        """Get the trades needed to reach the current position.

        Returns:
        -------
        Optional[pd.Series]
            Series of trades (changes in position) needed to reach the current position.
            None if no trades have been calculated yet.

        Notes:
        -----
        This is helpful when computing trading costs following a position change.
        Positive values represent buys, negative values represent sells.

        """
        return self._trades

    @property
    def mask(self) -> np.ndarray:
        """Get a boolean mask for assets with valid (non-NaN) prices.

        Returns:
        -------
        np.ndarray
            Boolean array where True indicates a valid price and False indicates
            a missing (NaN) price. Returns an empty array if no prices are set.

        """
        if self._prices is None:
            return np.empty(0, dtype=bool)

        return np.isfinite(self.prices.values)

    @property
    def prices(self) -> pd.Series:
        """Get the current prices of assets in the portfolio.

        Returns:
        -------
        pd.Series
            Series of current prices indexed by asset.
            Returns an empty series if no prices are set.

        """
        if self._prices is None:
            return pd.Series(dtype=float)
        return self._prices

    @prices.setter
    def prices(self, prices: pd.Series | dict) -> None:
        """Update the prices of assets in the portfolio.

        This method updates the prices and calculates the profit achieved
        due to price changes. It also updates the portfolio's AUM by adding
        the profit.

        Parameters
        ----------
        prices : pd.Series
            New prices for assets in the portfolio

        Notes:
        -----
        The profit is calculated as the difference between the portfolio value
        before and after the price update.

        """
        value_before = (self.prices * self.position).sum()  # self.cashposition.sum()
        value_after = (prices * self.position).sum()

        self._prices = prices
        self._profit = value_after - value_before
        self.aum += self.profit

    @property
    def profit(self) -> float:
        """Get the profit achieved between the previous and current prices.

        Returns:
        -------
        float
            The profit (or loss) achieved due to price changes since the
            last price update

        """
        return self._profit

    @property
    def aum(self) -> float:
        """Get the current assets under management (AUM) of the portfolio.

        Returns:
        -------
        float
            The total assets under management

        """
        return self._aum

    @aum.setter
    def aum(self, aum: float) -> None:
        """Update the assets under management (AUM) of the portfolio.

        Parameters
        ----------
        aum : float
            The new assets under management value to set

        """
        self._aum = aum

    @property
    def weights(self) -> pd.Series:
        """Get the weight of each asset in the portfolio.

        This computes the weighting of each asset as a fraction of the
        total portfolio value (NAV).

        Returns:
        -------
        pd.Series
            Series containing the weight of each asset as a fraction of the
            total portfolio value, indexed by asset

        Notes:
        -----
        If positions are missing, a series of zeros is effectively returned.
        The sum of weights equals 1.0 for a fully invested portfolio with no leverage.

        """
        if not np.isclose(self.nav, self.aum):
            raise ValueError(f"{self.nav} != {self.aum}")

        return self.cashposition / self.nav

    @property
    def leverage(self) -> float:
        """Get the leverage of the portfolio.

        Leverage is calculated as the sum of the absolute values of all position
        weights. For a long-only portfolio with no cash, this equals 1.0.
        For a portfolio with shorts or leverage, this will be greater than 1.0.

        Returns:
        -------
        float
            The leverage ratio of the portfolio

        Notes:
        -----
        A leverage of 2.0 means the portfolio has twice the market exposure
        compared to its net asset value, which could be achieved through
        borrowing or short selling.

        """
        return float(self.weights.abs().sum())
