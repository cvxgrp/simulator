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
"""
Portfolio representation and analysis for the CVX Simulator.

This module provides the Portfolio class, which represents a portfolio of assets
with methods for calculating various metrics (NAV, profit, drawdown, etc.) and
analyzing performance. The Portfolio class is typically created by the Builder
class after a simulation is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from jquantstats.api import build_data

from .utils.rescale import returns2prices


@dataclass(frozen=True)
class Portfolio:
    """
    Represents a portfolio of assets with methods for analysis and visualization.

    The Portfolio class is a frozen dataclass (immutable) that represents a portfolio
    of assets with their prices and positions (units). It provides methods for
    calculating various metrics like NAV, profit, drawdown, and for visualizing
    the portfolio's performance.

    Attributes
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices over time, with dates as index and assets as columns
    units : pd.DataFrame
        DataFrame of asset positions (units) over time, with dates as index and assets as columns
    aum : Union[float, pd.Series]
        Assets under management, either as a constant float or as a Series over time
    """

    prices: pd.DataFrame
    units: pd.DataFrame
    aum: float | pd.Series

    def __post_init__(self) -> None:
        """
        Validate the portfolio data after initialization.

        This method is automatically called after an instance of the Portfolio
        class has been initialized. It performs a series of validation checks
        to ensure that the prices and units dataframes are in the expected format
        with no duplicates or missing data.

        The method checks that:
        - Both prices and units dataframes have monotonic increasing indices
        - Both prices and units dataframes have unique indices
        - The index of units is a subset of the index of prices
        - The columns of units is a subset of the columns of prices

        Raises
        ------
        AssertionError
            If any of the validation checks fail
        """

        assert self.prices.index.is_monotonic_increasing
        assert self.prices.index.is_unique
        assert self.units.index.is_monotonic_increasing
        assert self.units.index.is_unique

        assert set(self.units.index).issubset(set(self.prices.index))
        assert set(self.units.columns).issubset(set(self.prices.columns))

    @property
    def index(self) -> pd.DatetimeIndex:
        """
        Get the time index of the portfolio.

        Returns
        -------
        pd.DatetimeIndex
            A DatetimeIndex representing the time period for which portfolio
            data is available

        Notes
        -----
        This property extracts the index from the prices DataFrame, which
        represents all time points in the portfolio history.
        """
        return pd.DatetimeIndex(self.prices.index)

    @property
    def assets(self) -> pd.Index:
        """
        Get the list of assets in the portfolio.

        Returns
        -------
        pd.Index
            An Index containing the names of all assets in the portfolio

        Notes
        -----
        This property extracts the column names from the prices DataFrame,
        which correspond to all assets for which price data is available.
        """
        return self.prices.columns

    @property
    def nav(self) -> pd.Series:
        """
        Get the net asset value (NAV) of the portfolio over time.

        The NAV represents the total value of the portfolio at each point in time.
        If aum is provided as a Series, it is used directly. Otherwise, the NAV
        is calculated from the cumulative profit plus the initial aum.

        Returns
        -------
        pd.Series
            Series representing the NAV of the portfolio over time
        """
        if isinstance(self.aum, pd.Series):
            series = self.aum
        else:
            profit = (self.cashposition.shift(1) * self.returns.fillna(0.0)).sum(axis=1)
            series = profit.cumsum() + self.aum

        series.name = "NAV"
        return series

    @property
    def profit(self) -> pd.Series:
        """
        Get the profit/loss of the portfolio at each time point.

        This calculates the profit or loss at each time point based on the
        previous positions and the returns of each asset.

        Returns
        -------
        pd.Series
            Series representing the profit/loss at each time point

        Notes
        -----
        The profit is calculated by multiplying the previous day's positions
        (in currency terms) by the returns of each asset, and then summing
        across all assets.
        """
        series = (self.cashposition.shift(1) * self.returns.fillna(0.0)).sum(axis=1)
        series.name = "Profit"
        return series

    @property
    def highwater(self) -> pd.Series:
        """
        Get the high-water mark of the portfolio over time.

        The high-water mark represents the highest value the portfolio has
        reached up to each point in time.

        Returns
        -------
        pd.Series
            Series representing the high-water mark at each time point

        Notes
        -----
        This is calculated using an expanding maximum of the NAV, which means
        at each point in time, it shows the maximum NAV achieved up to that point.
        This is commonly used to calculate drawdowns and performance fees.
        """
        series = self.nav.expanding(min_periods=1).max()
        series.name = "Highwater"
        return series

    @property
    def drawdown(self) -> pd.Series:
        """
        Get the drawdown of the portfolio over time.

        Drawdown measures the decline in portfolio value from its previous
        highest point (high-water mark) to the current point, expressed as
        a fraction of the high-water mark.

        Returns
        -------
        pd.Series
            Series representing the drawdown at each time point

        Notes
        -----
        Calculated as 1 - (NAV / high-water mark). A positive drawdown means
        the portfolio is currently worth less than its high-water mark.
        For example, a drawdown of 0.1 means the NAV is currently 90% of
        the high-water mark.
        """
        series = 1.0 - self.nav / self.highwater
        series.name = "Drawdown"
        return series

    @property
    def cashposition(self) -> pd.DataFrame:
        """
        Get the cash value of each position over time.

        This calculates the cash value of each position by multiplying
        the number of units by the price for each asset at each time point.

        Returns
        -------
        pd.DataFrame
            DataFrame with the cash value of each position over time,
            with dates as index and assets as columns
        """
        return self.prices * self.units

    @property
    def returns(self) -> pd.DataFrame:
        """
        Get the returns of individual assets over time.

        This calculates the percentage change in price for each asset
        from one time point to the next.

        Returns
        -------
        pd.DataFrame
            DataFrame with the returns of each asset over time,
            with dates as index and assets as columns
        """
        return self.prices.pct_change()

    @property
    def trades_units(self) -> pd.DataFrame:
        """
        Get the trades made in the portfolio in terms of units.

        This calculates the changes in position (units) from one time point
        to the next for each asset.

        Returns
        -------
        pd.DataFrame
            DataFrame with the trades (changes in units) for each asset over time,
            with dates as index and assets as columns

        Notes
        -----
        Calculated as the difference between consecutive position values.
        Positive values represent buys, negative values represent sells.
        The first row contains the initial positions, as there are no previous
        positions to compare with.
        """
        t = self.units.fillna(0.0).diff()
        t.loc[self.index[0]] = self.units.loc[self.index[0]]
        return t.fillna(0.0)

    @property
    def trades_currency(self) -> pd.DataFrame:
        """
        Get the trades made in the portfolio in terms of currency.

        This calculates the cash value of trades by multiplying the changes
        in position (units) by the current prices.

        Returns
        -------
        pd.DataFrame
            DataFrame with the cash value of trades for each asset over time,
            with dates as index and assets as columns

        Notes
        -----
        Calculated by multiplying trades_units by prices.
        Positive values represent buys (cash outflows),
        negative values represent sells (cash inflows).
        """
        return self.trades_units * self.prices

    @property
    def turnover_relative(self) -> pd.DataFrame:
        """
        Get the turnover relative to the portfolio NAV.

        This calculates the trades as a percentage of the portfolio NAV,
        which provides a measure of trading activity relative to portfolio size.

        Returns
        -------
        pd.DataFrame
            DataFrame with the relative turnover for each asset over time,
            with dates as index and assets as columns

        Notes
        -----
        Calculated by dividing trades_currency by NAV.
        Positive values represent buys, negative values represent sells.
        A value of 0.05 means a buy equal to 5% of the portfolio NAV.
        """
        return self.trades_currency.div(self.nav, axis=0)

    @property
    def turnover(self) -> pd.DataFrame:
        """
        Get the absolute turnover in the portfolio.

        This calculates the absolute value of trades in currency terms,
        which provides a measure of total trading activity regardless of
        direction (buy or sell).

        Returns
        -------
        pd.DataFrame
            DataFrame with the absolute turnover for each asset over time,
            with dates as index and assets as columns

        Notes
        -----
        Calculated as the absolute value of trades_currency.
        This is useful for calculating trading costs that apply equally
        to buys and sells.
        """
        return self.trades_currency.abs()

    def __getitem__(self, time: datetime | str | pd.Timestamp) -> pd.Series:
        """
        Get the portfolio positions (units) at a specific time.

        This method allows for dictionary-like access to the portfolio positions
        at a specific time point using the syntax: portfolio[time].

        Parameters
        ----------
        time : Union[datetime, str, pd.Timestamp]
            The time index for which to retrieve the positions

        Returns
        -------
        pd.Series
            Series containing the positions (units) for each asset at the specified time

        Raises
        ------
        KeyError
            If the specified time is not in the portfolio's index

        Examples
        --------
        >>> portfolio['2023-01-01']  # Get positions on January 1, 2023
        >>> portfolio[pd.Timestamp('2023-01-01')]  # Same as above
        """
        return self.units.loc[time]

    @property
    def equity(self) -> pd.DataFrame:
        """
        Get the equity (cash value) of each position over time.

        This property returns the cash value of each position in the portfolio,
        calculated by multiplying the number of units by the price for each asset.

        Returns
        -------
        pd.DataFrame
            DataFrame with the cash value of each position over time,
            with dates as index and assets as columns

        Notes
        -----
        This is an alias for the cashposition property and returns the same values.
        The term "equity" is used in the context of the cash value of positions,
        not to be confused with the equity asset class.
        """
        return self.cashposition

    @property
    def weights(self) -> pd.DataFrame:
        """
        Get the weight of each asset in the portfolio over time.

        This calculates the relative weight of each asset in the portfolio
        by dividing the cash value of each position by the total portfolio
        value (NAV) at each time point.

        Returns
        -------
        pd.DataFrame
            DataFrame with the weight of each asset over time,
            with dates as index and assets as columns

        Notes
        -----
        The sum of weights across all assets at any given time should equal 1.0
        for a fully invested portfolio with no leverage. Weights can be negative
        for short positions.
        """
        return self.equity.apply(lambda x: x / self.nav)

    @property
    def data(self):
        frame = self.nav.pct_change().to_frame()
        frame.index.name = "Date"
        return build_data(returns=frame)

    @classmethod
    def from_cashpos_prices(cls, prices: pd.DataFrame, cashposition: pd.DataFrame, aum: float):
        """Build Futures Portfolio from cashposition"""
        units = cashposition.div(prices, fill_value=0.0)
        return cls(prices=prices, units=units, aum=aum)

    @classmethod
    def from_cashpos_returns(cls, returns: pd.DataFrame, cashposition: pd.DataFrame, aum: float):
        """Build Futures Portfolio from cashposition"""
        prices = returns2prices(returns)
        return cls.from_cashpos_prices(prices, cashposition, aum)

    def snapshot(self, title: str = "Portfolio Summary", log_scale: bool = True):
        return self.data.plots.plot_snapshot(title=title, log_scale=log_scale)
