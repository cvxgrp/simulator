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
"""Portfolio representation and analysis for the CVX Simulator.

This module provides the Portfolio class, which represents a portfolio of assets
with methods for calculating various metrics (NAV, profit, drawdown, etc.) and
analyzing performance. The Portfolio class is typically created by the Builder
class after a simulation is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from jquantstats._data import Data
from jquantstats.api import build_data


@dataclass(frozen=True)
class Portfolio:
    """Represents a portfolio of assets with methods for analysis and visualization.

    The Portfolio class is a frozen dataclass (immutable) that represents a portfolio
    of assets with their prices and positions (units). It provides methods for
    calculating various metrics like NAV, profit, drawdown, and for visualizing
    the portfolio's performance.

    Attributes:
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
    _data: Data = field(init=False)

    def __post_init__(self) -> None:
        """Validate the portfolio data after initialization.

        This method is automatically called after an instance of the Portfolio
        class has been initialized. It performs a series of validation checks
        to ensure that the prices and units dataframes are in the expected format
        with no duplicates or missing data.

        The method checks that:
        - Both prices and units dataframes have monotonic increasing indices
        - Both prices and units dataframes have unique indices
        - The index of units is a subset of the index of prices
        - The columns of units is a subset of the columns of prices

        Raises:
        ------
        AssertionError
            If any of the validation checks fail

        """
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("`prices` index must be monotonic increasing.")

        if not self.prices.index.is_unique:
            raise ValueError("`prices` index must be unique.")

        if not self.units.index.is_monotonic_increasing:
            raise ValueError("`units` index must be monotonic increasing.")

        if not self.units.index.is_unique:
            raise ValueError("`units` index must be unique.")

        missing_dates = self.units.index.difference(self.prices.index)
        if not missing_dates.empty:
            raise ValueError(f"`units` index contains dates not present in `prices`: {missing_dates.tolist()}")

        missing_assets = self.units.columns.difference(self.prices.columns)
        if not missing_assets.empty:
            raise ValueError(f"`units` contains assets not present in `prices`: {missing_assets.tolist()}")

        frame = self.nav.pct_change().to_frame()
        frame.index.name = "Date"
        d = build_data(returns=frame)

        object.__setattr__(self, "_data", d)

    @property
    def index(self) -> list[datetime]:
        """Get the time index of the portfolio.

        Returns:
        -------
        pd.DatetimeIndex
            A DatetimeIndex representing the time period for which portfolio
            data is available

        Notes:
        -----
        This property extracts the index from the prices DataFrame, which
        represents all time points in the portfolio history.

        """
        return pd.DatetimeIndex(self.prices.index).to_list()

    @property
    def assets(self) -> list[str]:
        """Get the list of assets in the portfolio.

        Returns:
        -------
        pd.Index
            An Index containing the names of all assets in the portfolio

        Notes:
        -----
        This property extracts the column names from the prices DataFrame,
        which correspond to all assets for which price data is available.

        """
        return self.prices.columns.to_list()

    @property
    def nav(self) -> pd.Series:
        """Get the net asset value (NAV) of the portfolio over time.

        The NAV represents the total value of the portfolio at each point in time.
        If aum is provided as a Series, it is used directly. Otherwise, the NAV
        is calculated from the cumulative profit plus the initial aum.

        Returns:
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
        """Get the profit/loss of the portfolio at each time point.

        This calculates the profit or loss at each time point based on the
        previous positions and the returns of each asset.

        Returns:
        -------
        pd.Series
            Series representing the profit/loss at each time point

        Notes:
        -----
        The profit is calculated by multiplying the previous day's positions
        (in currency terms) by the returns of each asset, and then summing
        across all assets.

        """
        series = (self.cashposition.shift(1) * self.returns.fillna(0.0)).sum(axis=1)
        series.name = "Profit"
        return series

    @property
    def cashposition(self) -> pd.DataFrame:
        """Get the cash value of each position over time.

        This calculates the cash value of each position by multiplying
        the number of units by the price for each asset at each time point.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the cash value of each position over time,
            with dates as index and assets as columns

        """
        return self.prices * self.units

    @property
    def returns(self) -> pd.DataFrame:
        """Get the returns of individual assets over time.

        This calculates the percentage change in price for each asset
        from one time point to the next.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the returns of each asset over time,
            with dates as index and assets as columns

        """
        return self.prices.pct_change()

    @property
    def trades_units(self) -> pd.DataFrame:
        """Get the trades made in the portfolio in terms of units.

        This calculates the changes in position (units) from one time point
        to the next for each asset.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the trades (changes in units) for each asset over time,
            with dates as index and assets as columns

        Notes:
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
        """Get the trades made in the portfolio in terms of currency.

        This calculates the cash value of trades by multiplying the changes
        in position (units) by the current prices.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the cash value of trades for each asset over time,
            with dates as index and assets as columns

        Notes:
        -----
        Calculated by multiplying trades_units by prices.
        Positive values represent buys (cash outflows),
        negative values represent sells (cash inflows).

        """
        return self.trades_units * self.prices

    @property
    def turnover_relative(self) -> pd.DataFrame:
        """Get the turnover relative to the portfolio NAV.

        This calculates the trades as a percentage of the portfolio NAV,
        which provides a measure of trading activity relative to portfolio size.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the relative turnover for each asset over time,
            with dates as index and assets as columns

        Notes:
        -----
        Calculated by dividing trades_currency by NAV.
        Positive values represent buys, negative values represent sells.
        A value of 0.05 means a buy equal to 5% of the portfolio NAV.

        """
        return self.trades_currency.div(self.nav, axis=0)

    @property
    def turnover(self) -> pd.DataFrame:
        """Get the absolute turnover in the portfolio.

        This calculates the absolute value of trades in currency terms,
        which provides a measure of total trading activity regardless of
        direction (buy or sell).

        Returns:
        -------
        pd.DataFrame
            DataFrame with the absolute turnover for each asset over time,
            with dates as index and assets as columns

        Notes:
        -----
        Calculated as the absolute value of trades_currency.
        This is useful for calculating trading costs that apply equally
        to buys and sells.

        """
        return self.trades_currency.abs()

    def __getitem__(self, time: datetime | str | pd.Timestamp) -> pd.Series:
        """Get the portfolio positions (units) at a specific time.

        This method allows for dictionary-like access to the portfolio positions
        at a specific time point using the syntax: portfolio[time].

        Parameters
        ----------
        time : Union[datetime, str, pd.Timestamp]
            The time index for which to retrieve the positions

        Returns:
        -------
        pd.Series
            Series containing the positions (units) for each asset at the specified time

        Raises:
        ------
        KeyError
            If the specified time is not in the portfolio's index

        Examples:
        --------
        ```
        portfolio['2023-01-01']  # Get positions on January 1, 2023
        portfolio[pd.Timestamp('2023-01-01')]  # Same as above
        ```

        """
        return self.units.loc[time]

    @property
    def equity(self) -> pd.DataFrame:
        """Get the equity (cash value) of each position over time.

        This property returns the cash value of each position in the portfolio,
        calculated by multiplying the number of units by the price for each asset.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the cash value of each position over time,
            with dates as index and assets as columns

        Notes:
        -----
        This is an alias for the cashposition property and returns the same values.
        The term "equity" is used in the context of the cash value of positions,
        not to be confused with the equity asset class.

        """
        return self.cashposition

    @property
    def weights(self) -> pd.DataFrame:
        """Get the weight of each asset in the portfolio over time.

        This calculates the relative weight of each asset in the portfolio
        by dividing the cash value of each position by the total portfolio
        value (NAV) at each time point.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the weight of each asset over time,
            with dates as index and assets as columns

        Notes:
        -----
        The sum of weights across all assets at any given time should equal 1.0
        for a fully invested portfolio with no leverage. Weights can be negative
        for short positions.

        """
        return self.equity.apply(lambda x: x / self.nav)

    @property
    def stats(self):
        """Get statistical analysis data for the portfolio.

        This property provides access to various statistical metrics calculated
        for the portfolio, such as Sharpe ratio, volatility, drawdowns, etc.

        Returns:
        -------
        object
            An object containing various statistical metrics for the portfolio

        Notes:
        -----
        The statistics are calculated by the underlying jquantstats library
        and are based on the portfolio's NAV time series.

        """
        return self._data.stats

    @property
    def plots(self):
        """Get visualization tools for the portfolio.

        This property provides access to various plotting functions for visualizing
        the portfolio's performance, returns, drawdowns, etc.

        Returns:
        -------
        object
            An object containing various plotting methods for the portfolio

        Notes:
        -----
        The plotting functions are provided by the underlying jquantstats library
        and operate on the portfolio's NAV time series.

        """
        return self._data.plots

    @property
    def reports(self):
        """Get reporting tools for the portfolio.

        This property provides access to various reporting functions for generating
        performance reports, risk metrics, and other analytics for the portfolio.

        Returns:
        -------
        object
            An object containing various reporting methods for the portfolio

        Notes:
        -----
        The reporting functions are provided by the underlying jquantstats library
        and operate on the portfolio's NAV time series.

        """
        return self._data.reports

    def sharpe(self, periods=None):
        """Calculate the Sharpe ratio for the portfolio.

        The Sharpe ratio is a measure of risk-adjusted return, calculated as
        the portfolio's excess return divided by its volatility.

        Parameters
        ----------
        periods : int, optional
            The number of periods per year for annualization.
            For daily data, use 252; for weekly data, use 52; for monthly data, use 12.
            If None, no annualization is performed.

        Returns:
        -------
        float
            The Sharpe ratio of the portfolio

        Notes:
        -----
        The Sharpe ratio is calculated using the portfolio's NAV time series.
        A higher Sharpe ratio indicates better risk-adjusted performance.

        """
        return self.stats.sharpe(periods=periods)["NAV"]

    @classmethod
    def from_cashpos_prices(cls, prices: pd.DataFrame, cashposition: pd.DataFrame, aum: float):
        """Create a Portfolio instance from cash positions and prices.

        This class method provides an alternative way to create a Portfolio instance
        when you have the cash positions rather than the number of units.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of asset prices over time, with dates as index and assets as columns
        cashposition : pd.DataFrame
            DataFrame of cash positions over time, with dates as index and assets as columns
        aum : float
            Assets under management

        Returns:
        -------
        Portfolio
            A new Portfolio instance with units calculated from cash positions and prices

        Notes:
        -----
        The units are calculated by dividing the cash positions by the prices.
        This is useful when you have the monetary value of each position rather
        than the number of units.

        """
        units = cashposition.div(prices, fill_value=0.0)
        return cls(prices=prices, units=units, aum=aum)

    def snapshot(self, title: str = "Portfolio Summary", log_scale: bool = True):
        """Generate and display a snapshot of the portfolio summary.

        This method creates a visual representation of the portfolio summary
        using the associated plot functionalities. The snapshot can be
        configured with a title and whether to use a logarithmic scale.

        Args:
            title: A string specifying the title of the snapshot.
                   Default is "Portfolio Summary".
            log_scale: A boolean indicating whether to display the plot
                       using a logarithmic scale. Default is True.

        Returns:
            The generated plot object representing the portfolio snapshot.

        """
        return self.plots.plot_snapshot(title=title, log_scale=log_scale)
