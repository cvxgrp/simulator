"""Builder class for the CVX Simulator."""

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

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from .portfolio import Portfolio
from .state import State
from .utils.interpolation import valid


def polars2pandas(dframe: pl.DataFrame, date_col="date") -> pd.DataFrame:
    """Convert a Polars DataFrame to a Pandas DataFrame.

    Ensuring the date column is cast to a datetime format and
    all other columns are cast to Float64. The resulting Pandas DataFrame is indexed by the specified date column.

    Args:
        dframe (pl.DataFrame): The Polars DataFrame to be converted.
        date_col (str): The name of the column containing date values, defaults to "date".

    Returns:
        pd.DataFrame: The converted Pandas DataFrame with the date column as its index.

    """
    dframe = dframe.with_columns(pl.col(date_col).cast(pl.Datetime("ns")))
    dframe = dframe.with_columns([pl.col(col).cast(pl.Float64) for col in dframe.columns if col != date_col])
    return dframe.to_pandas().set_index(date_col)


@dataclass
class Builder:
    """The Builder is an auxiliary class used to build portfolios.

    It overloads the __iter__ method to allow the class to iterate over
    the timestamps for which the portfolio data is available.

    In each iteration we can update the portfolio by setting either
    the weights, the position or the cash position.

    After the iteration has been completed we build a Portfolio object
    by calling the build method.
    """

    prices: pd.DataFrame

    _state: State | None = None
    _units: pd.DataFrame | None = None
    _aum: pd.Series | None = None
    initial_aum: float = 1e6

    def __post_init__(self) -> None:
        """Initialize the Builder instance after creation.

        This method is automatically called after the object is initialized.
        It sets up the internal state, creates empty DataFrames for units and AUM,
        and initializes the AUM with the provided initial_aum value.

        The method performs several validations on the prices DataFrame:
        - Checks that the index is monotonically increasing
        - Checks that the index has unique values

        Returns:
        -------
        None

        """
        # assert isinstance(self.prices, pd.DataFrame)
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("Index must be monotonically increasing")

        if not self.prices.index.is_unique:
            raise ValueError("Index must have unique values")

        self._state = State()

        self._units = pd.DataFrame(
            index=self.prices.index,
            columns=self.prices.columns,
            data=np.nan,
            dtype=float,
        )

        self._aum = pd.Series(index=self.prices.index, dtype=float)

        self._state.aum = self.initial_aum

    @property
    def valid(self):
        """Check the validity of price data for each asset.

        This property analyzes each column of the prices DataFrame to determine
        if there are any missing values between the first and last valid data points.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with the same columns as prices, containing boolean values
            indicating whether each asset's price series is valid (True) or has
            missing values in the middle (False)

        Notes:
        -----
        A valid price series can have missing values at the beginning or end,
        but not in the middle between the first and last valid data points.

        """
        return self.prices.apply(valid)

    @property
    def intervals(self):
        """Get the first and last valid index for each asset's price series.

        This property identifies the time range for which each asset has valid price data.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with assets as rows and two columns:
            - 'first': The first valid index (timestamp) for each asset
            - 'last': The last valid index (timestamp) for each asset

        Notes:
        -----
        This is useful for determining the valid trading period for each asset,
        especially when different assets have different data availability periods.

        """
        return self.prices.apply(
            lambda ts: pd.Series({"first": ts.first_valid_index(), "last": ts.last_valid_index()})
        ).transpose()

    @property
    def index(self) -> pd.DatetimeIndex:
        """The index of the portfolio.

        Returns: pd.Index: A pandas index representing the
        time period for which the portfolio data is available.
        """
        return pd.DatetimeIndex(self.prices.index)

    @property
    def current_prices(self) -> np.ndarray:
        """Get the current prices for all assets in the portfolio.

        This property retrieves the current prices from the internal state
        for all assets that are currently in the portfolio.

        Returns:
        -------
        np.array
            An array of current prices for all assets in the portfolio

        Notes:
        -----
        The prices are retrieved from the internal state, which is updated
        during iteration through the portfolio's time index.

        """
        return self._state.prices[self._state.assets].to_numpy()

    def __iter__(self) -> Generator[tuple[pd.DatetimeIndex, State]]:
        """Iterate over object in a for loop.

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
        """The position property returns the current position of the portfolio.

        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        return self._units.loc[self._state.time]

    @position.setter
    def position(self, position: pd.Series) -> None:
        """Set the current position of the portfolio.

        This setter updates the position (number of units) for each asset in the portfolio
        at the current time point. It also updates the internal state's position.

        Parameters
        ----------
        position : pd.Series
            A pandas Series containing the new position (number of units) for each asset

        Returns:
        -------
        None

        """
        self._units.loc[self._state.time, self._state.assets] = position
        self._state.position = position

    @property
    def cashposition(self):
        """Get the current cash value of each position in the portfolio.

        This property calculates the cash value of each position by multiplying
        the number of units by the current price for each asset.

        Returns:
        -------
        pd.Series
            A pandas Series containing the cash value of each position,
            indexed by asset

        Notes:
        -----
        This is different from the 'cash' property, which represents
        uninvested money. This property represents the market value
        of each invested position.

        """
        return self.position * self.current_prices

    @property
    def units(self):
        """Get the complete history of portfolio holdings.

        This property returns the entire DataFrame of holdings (units) for all
        assets over all time points in the portfolio.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the number of units held for each asset over time,
            with dates as index and assets as columns

        Notes:
        -----
        This property is particularly useful for testing and for building
        the final Portfolio object via the build() method.

        """
        return self._units

    @cashposition.setter
    def cashposition(self, cashposition: pd.Series) -> None:
        """Set the current cash value of each position in the portfolio.

        This setter updates the cash value of each position and automatically
        converts the cash values to positions (units) using the current prices.

        Parameters
        ----------
        cashposition : pd.Series
            A pandas Series containing the new cash value for each position,
            indexed by asset

        Returns:
        -------
        None

        Notes:
        -----
        This is a convenient way to specify positions in terms of currency
        amounts rather than number of units. The conversion formula is:
        position = cashposition / prices

        """
        self.position = cashposition / self.current_prices

    def build(self):
        """Create a new Portfolio instance from the current builder state.

        This method creates a new immutable Portfolio object based on the
        current state of the Builder, which can be used for analysis and reporting.

        Returns:
        -------
        Portfolio
            A new instance of the Portfolio class with the attributes
            (prices, units, aum) as specified in the Builder

        Notes:
        -----
        The resulting Portfolio object will be immutable (frozen) and will
        have the same data as the Builder from which it was built, but
        with a different interface focused on analysis rather than construction.

        """
        return Portfolio(prices=self.prices, units=self.units, aum=self.aum)

    @property
    def weights(self) -> np.ndarray:
        """Get the current portfolio weights for each asset.

        This property retrieves the weight of each asset in the portfolio
        from the internal state. Weights represent the proportion of the
        portfolio's value invested in each asset.

        Returns:
        -------
        np.array
            An array of weights for each asset in the portfolio

        Notes:
        -----
        Weights sum to 1.0 for a fully invested portfolio with no leverage.
        Negative weights represent short positions.

        """
        return self._state.weights[self._state.assets].to_numpy()

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """Set the current portfolio weights for each asset.

        This setter updates the portfolio weights and automatically converts
        the weights to positions (units) using the current prices and NAV.

        Parameters
        ----------
        weights : np.array
            An array of weights for each asset in the portfolio

        Returns:
        -------
        None

        Notes:
        -----
        This is a convenient way to rebalance the portfolio by specifying
        the desired allocation as weights rather than exact positions.
        The conversion formula is: position = NAV * weights / prices

        """
        self.position = self._state.nav * weights / self.current_prices

    @property
    def aum(self):
        """Get the assets under management (AUM) history of the portfolio.

        This property returns the entire series of AUM values over time,
        representing the total value of the portfolio at each time point.

        Returns:
        -------
        pd.Series
            A Series containing the AUM values over time, with dates as index

        Notes:
        -----
        AUM (assets under management) represents the total value of the portfolio,
        including both invested positions and uninvested cash.

        """
        return self._aum

    @aum.setter
    def aum(self, aum):
        """Set the current assets under management (AUM) of the portfolio.

        This setter updates the AUM value at the current time point and
        also updates the internal state's AUM.

        Parameters
        ----------
        aum : float
            The new AUM value to set

        Returns:
        -------
        None

        Notes:
        -----
        Changing the AUM affects the portfolio's ability to take positions,
        as position sizes are often calculated as a fraction of AUM.

        """
        self._aum[self._state.time] = aum
        self._state.aum = aum
