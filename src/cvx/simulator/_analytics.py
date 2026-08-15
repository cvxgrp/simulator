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
"""jquantstats-backed performance analytics for the Portfolio class.

This module houses :class:`PortfolioAnalytics`, a mixin that supplies the
statistics, plotting, and reporting surface layered on top of a portfolio's
net asset value. Keeping it separate from the core
:class:`~cvx.simulator.portfolio.Portfolio` data model isolates the dependency
on :mod:`jquantstats` and keeps each module focused on a single concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jquantstats.data import Data

if TYPE_CHECKING:
    import pandas as pd


class PortfolioAnalytics:
    """Mixin providing performance statistics, plots, and reports for a portfolio.

    The mixin expects the host class to expose a ``nav`` :class:`pandas.Series`
    property. From that NAV it lazily builds and caches a
    :class:`jquantstats.data.Data` object (in :meth:`_build_data`) and exposes
    the ``stats``/``plots``/``reports`` surface plus the :meth:`sharpe` and
    :meth:`snapshot` convenience helpers on top of it.

    Examples:
    --------
    The mixin is not used directly — :class:`~cvx.simulator.portfolio.Portfolio`
    inherits it, so every portfolio carries the analytics surface:

    >>> import pandas as pd
    >>> from cvx.simulator import Portfolio
    >>> from cvx.simulator._analytics import PortfolioAnalytics
    >>> issubclass(Portfolio, PortfolioAnalytics)
    True

    >>> dates = pd.date_range("2020-01-01", periods=4)
    >>> prices = pd.DataFrame(
    ...     {"A": [100.0, 102.0, 104.0, 103.0], "B": [50.0, 51.0, 52.0, 51.0]},
    ...     index=dates,
    ... )
    >>> units = pd.DataFrame({"A": [5.0] * 4, "B": [10.0] * 4}, index=dates)
    >>> portfolio = Portfolio(prices=prices, units=units, aum=1000.0)

    All of it is derived from the NAV alone:

    >>> portfolio.nav.tolist()
    [1000.0, 1020.0, 1040.0, 1025.0]
    >>> sorted(m for m in ("stats", "plots", "reports") if hasattr(portfolio, m))
    ['plots', 'reports', 'stats']

    """

    if TYPE_CHECKING:
        # Supplied by the host dataclass (Portfolio); declared here purely so the
        # type checker can resolve the attributes the mixin relies on.
        _data: Data

        @property
        def nav(self) -> pd.Series:
            """Net asset value of the portfolio over time (provided by the host)."""
            ...

    def _build_data(self) -> None:
        """Build and cache the derived quantstats Data from the portfolio NAV.

        This constructs the returns-based :class:`~jquantstats.data.Data` object
        used by the reporting and statistics helpers and stores it on the frozen
        instance's ``_data`` field. It is invoked from ``__post_init__`` after
        the input validation has passed.
        """
        frame = self.nav.pct_change().to_frame().reset_index()
        frame.columns = ["Date", frame.columns[1]]
        d = Data.from_returns(returns=frame)

        object.__setattr__(self, "_data", d)

    @property
    def stats(self) -> Any:
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
    def plots(self) -> Any:
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
    def reports(self) -> Any:
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

    def sharpe(self, periods: int | None = None) -> float:
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

        Without ``periods`` the ratio is left unannualized:

        >>> round(portfolio.sharpe(), 2)
        8.12

        Passing the number of periods per year annualizes it — 252 for daily
        data, 52 for weekly, 12 for monthly:

        >>> round(portfolio.sharpe(periods=252), 2)
        6.74

        Both numbers are meaningless as investment results: three returns is far
        too short a sample to say anything about risk-adjusted performance. They
        demonstrate the call, not a realistic figure.

        """
        return float(self.stats.sharpe(periods=periods)["NAV"])

    def snapshot(self, title: str = "Portfolio Summary", log_scale: bool = True) -> Any:
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
        return self.plots.snapshot(title=title, log_scale=log_scale)
