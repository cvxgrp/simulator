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

The jquantstats-backed statistics, plotting, and reporting surface lives in the
companion :class:`~cvx.simulator._analytics.PortfolioAnalytics` mixin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from jquantstats.data import Data

from ._analytics import PortfolioAnalytics


@dataclass(frozen=True)
class Portfolio(PortfolioAnalytics):
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

    Examples:
    --------
    A portfolio is usually produced by :meth:`Builder.build`, but it can be
    constructed directly from prices, units and a starting AUM:

    >>> import pandas as pd
    >>> from cvx.simulator import Portfolio
    >>> dates = pd.date_range("2020-01-01", periods=4)
    >>> prices = pd.DataFrame(
    ...     {"A": [100.0, 102.0, 104.0, 103.0], "B": [50.0, 51.0, 52.0, 51.0]},
    ...     index=dates,
    ... )
    >>> units = pd.DataFrame({"A": [5.0] * 4, "B": [10.0] * 4}, index=dates)
    >>> portfolio = Portfolio(prices=prices, units=units, aum=1000.0)
    >>> portfolio.assets
    ['A', 'B']

    The cash value of each holding, and the NAV it rolls up to:

    >>> portfolio.cashposition
                    A      B
    2020-01-01  500.0  500.0
    2020-01-02  510.0  510.0
    2020-01-03  520.0  520.0
    2020-01-04  515.0  510.0
    >>> portfolio.nav
    2020-01-01    1000.0
    2020-01-02    1020.0
    2020-01-03    1040.0
    2020-01-04    1025.0
    Freq: D, Name: NAV, dtype: float64

    The object is frozen, so analysis can never mutate the record:

    >>> portfolio.aum = 2000.0
    Traceback (most recent call last):
        ...
    dataclasses.FrozenInstanceError: cannot assign to field 'aum'

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
        ValueError
            If any of the validation checks fail

        """
        self._validate()
        self._build_data()

    def _validate(self) -> None:
        """Validate the prices and units dataframes.

        Checks that both frames have monotonic increasing, unique indices and
        that the units index and columns are subsets of the prices index and
        columns respectively.

        Raises:
        ------
        ValueError
            If any of the validation checks fail
        """
        index_checks = (
            (self.prices.index.is_monotonic_increasing, "`prices` index must be monotonic increasing."),
            (self.prices.index.is_unique, "`prices` index must be unique."),
            (self.units.index.is_monotonic_increasing, "`units` index must be monotonic increasing."),
            (self.units.index.is_unique, "`units` index must be unique."),
        )
        for is_valid, message in index_checks:
            if not is_valid:
                raise ValueError(message)

        missing_dates = self.units.index.difference(self.prices.index)
        if not missing_dates.empty:
            msg = f"`units` index contains dates not present in `prices`: {missing_dates.tolist()}"
            raise ValueError(msg)

        missing_assets = self.units.columns.difference(self.prices.columns)
        if not missing_assets.empty:
            msg = f"`units` contains assets not present in `prices`: {missing_assets.tolist()}"
            raise ValueError(msg)

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
        return list(pd.DatetimeIndex(self.prices.index))

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
        return list(self.prices.columns)

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

        Examples:
        --------
        >>> import pandas as pd
        >>> from cvx.simulator import Portfolio
        >>> dates = pd.date_range("2020-01-01", periods=4)
        >>> prices = pd.DataFrame(
        ...     {"A": [100.0, 102.0, 104.0, 103.0], "B": [50.0, 51.0, 52.0, 51.0]},
        ...     index=dates,
        ... )
        >>> units = pd.DataFrame({"A": [5.0] * 4, "B": [10.0] * 4}, index=dates)

        Passing a scalar ``aum`` makes the NAV the cumulative profit on top of it:

        >>> Portfolio(prices=prices, units=units, aum=1000.0).nav
        2020-01-01    1000.0
        2020-01-02    1020.0
        2020-01-03    1040.0
        2020-01-04    1025.0
        Freq: D, Name: NAV, dtype: float64

        Passing a Series instead uses it verbatim — which is what
        :meth:`Builder.build` does, so a strategy that adds or withdraws capital
        is recorded rather than inferred:

        >>> aum = pd.Series([1000.0, 1100.0, 1200.0, 1300.0], index=dates)
        >>> Portfolio(prices=prices, units=units, aum=aum).nav
        2020-01-01    1000.0
        2020-01-02    1100.0
        2020-01-03    1200.0
        2020-01-04    1300.0
        Freq: D, Name: NAV, dtype: float64

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

        Examples:
        --------
        >>> import pandas as pd
        >>> from cvx.simulator import Portfolio
        >>> dates = pd.date_range("2020-01-01", periods=4)
        >>> prices = pd.DataFrame(
        ...     {"A": [100.0, 102.0, 104.0, 103.0], "B": [50.0, 51.0, 52.0, 51.0]},
        ...     index=dates,
        ... )
        >>> units = pd.DataFrame({"A": [5.0] * 4, "B": [10.0] * 4}, index=dates)
        >>> portfolio = Portfolio(prices=prices, units=units, aum=1000.0)

        The first day has no previous position, so it books no profit:

        >>> portfolio.profit
        2020-01-01     0.0
        2020-01-02    20.0
        2020-01-03    20.0
        2020-01-04   -15.0
        Freq: D, Name: Profit, dtype: float64

        Cumulating the profit onto the starting AUM reproduces the NAV:

        >>> (portfolio.profit.cumsum() + 1000.0).equals(portfolio.nav.rename("Profit"))
        True

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
        >>> import pandas as pd
        >>> from cvx.simulator import Portfolio
        >>> dates = pd.date_range("2020-01-01", periods=4)
        >>> prices = pd.DataFrame(
        ...     {"A": [100.0, 102.0, 104.0, 103.0], "B": [50.0, 51.0, 52.0, 51.0]},
        ...     index=dates,
        ... )
        >>> units = pd.DataFrame({"A": [5.0] * 4, "B": [10.0] * 4}, index=dates)
        >>> portfolio = Portfolio(prices=prices, units=units, aum=1000.0)

        Index with a string or a Timestamp — both reach the same row:

        >>> portfolio["2020-01-02"]
        A     5.0
        B    10.0
        Name: 2020-01-02 00:00:00, dtype: float64
        >>> portfolio[pd.Timestamp("2020-01-02")].equals(portfolio["2020-01-02"])
        True

        A date outside the index raises. (Caught here rather than shown as a
        traceback: pandas chains several exceptions on the way out, and the
        intermediate frames are an implementation detail, not a contract.)

        >>> try:
        ...     portfolio["2021-01-01"]
        ... except KeyError as err:
        ...     print(err)
        '2021-01-01'

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

        Examples:
        --------
        >>> import pandas as pd
        >>> from cvx.simulator import Portfolio
        >>> dates = pd.date_range("2020-01-01", periods=4)
        >>> prices = pd.DataFrame(
        ...     {"A": [100.0, 102.0, 104.0, 103.0], "B": [50.0, 51.0, 52.0, 51.0]},
        ...     index=dates,
        ... )
        >>> units = pd.DataFrame({"A": [5.0] * 4, "B": [10.0] * 4}, index=dates)
        >>> portfolio = Portfolio(prices=prices, units=units, aum=1000.0)

        Holding the units fixed, the weights drift with the relative prices:

        >>> portfolio.weights.round(4)
                         A       B
        2020-01-01  0.5000  0.5000
        2020-01-02  0.5000  0.5000
        2020-01-03  0.5000  0.5000
        2020-01-04  0.5024  0.4976

        This book is fully invested, so each row sums to 1.0:

        >>> portfolio.weights.sum(axis=1).round(6).unique().tolist()
        [1.0]

        """
        return self.equity.apply(lambda x: x / self.nav)

    @classmethod
    def from_cashpos_prices(cls, prices: pd.DataFrame, cashposition: pd.DataFrame, aum: float) -> Portfolio:
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

        Examples:
        --------
        Specify the book in currency rather than units — 500 in each name:

        >>> import pandas as pd
        >>> from cvx.simulator import Portfolio
        >>> dates = pd.date_range("2020-01-01", periods=4)
        >>> prices = pd.DataFrame(
        ...     {"A": [100.0, 102.0, 104.0, 103.0], "B": [50.0, 51.0, 52.0, 51.0]},
        ...     index=dates,
        ... )
        >>> cashposition = pd.DataFrame({"A": [500.0] * 4, "B": [500.0] * 4}, index=dates)
        >>> portfolio = Portfolio.from_cashpos_prices(
        ...     prices=prices, cashposition=cashposition, aum=1000.0
        ... )

        The units are the cash amounts divided by the prices:

        >>> portfolio.units.round(4)
                         A        B
        2020-01-01  5.0000  10.0000
        2020-01-02  4.9020   9.8039
        2020-01-03  4.8077   9.6154
        2020-01-04  4.8544   9.8039

        """
        units = cashposition.div(prices, fill_value=0.0)
        return cls(prices=prices, units=units, aum=aum)
