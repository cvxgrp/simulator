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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
import quantstats as qs

from cvx.simulator.plot import Plot

qs.extend_pandas()


@dataclass(frozen=True)
class Portfolio:
    prices: pd.DataFrame
    stocks: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        """A property that returns the index of the EquityPortfolio instance,
        which is the time period for which the portfolio data is available.

        Returns: pd.Index: A pandas index representing the time period for which the
        portfolio data is available.

        Notes: The function extracts the index of the prices dataframe,
        which represents the time periods for which data is available for the portfolio.
        The resulting index will be a pandas index object with the same length
        as the number of rows in the prices dataframe."""
        return pd.DatetimeIndex(self.prices.index)

    @property
    def assets(self) -> pd.Index:
        """A property that returns a list of the assets held by the EquityPortfolio object.

        Returns: list: A list of the assets held by the EquityPortfolio object.

        Notes: The function extracts the column names of the prices dataframe,
        which correspond to the assets held by the EquityPortfolio object.
        The resulting list will contain the names of all assets held by the portfolio, without any duplicates.
        """
        return self.prices.columns

    @property
    @abstractmethod
    def nav(self):
        """A function that returns a pandas series representing the NAV"""

    @property
    def highwater(self) -> pd.Series:
        """A function that returns a pandas series representing
        the high-water mark of the portfolio, which is the highest point
        the portfolio value has reached over time.

        Returns: pd.Series: A pandas series representing the
        high-water mark of the portfolio.

        Notes: The function performs a rolling computation based on
        the cumulative maximum of the portfolio's value over time,
        starting from the beginning of the time period being considered.
        Min_periods argument is set to 1 to include the minimum period of one day.
        The resulting series will show the highest value the portfolio has reached at each point in time.
        """
        return self.nav.expanding(min_periods=1).max()

    @property
    def drawdown(self) -> pd.Series:
        """A property that returns a pandas series representing the
        drawdown of the portfolio, which measures the decline
        in the portfolio's value from its (previously) highest
        point to its current point.

        Returns: pd.Series: A pandas series representing the
        drawdown of the portfolio.

        Notes: The function calculates the ratio of the portfolio's current value
        vs. its current high-water-mark and then subtracting the result from 1.
        A positive drawdown means the portfolio is currently worth
        less than its high-water mark. A drawdown of 0.1 implies that the nav is currently 0.9 times the high-water mark
        """
        return 1.0 - self.nav / self.highwater

    def metrics(
        self,
        benchmark: Any = None,
        rf: float = 0.0,
        display: bool = True,
        mode: str = "basic",
        sep: bool = False,
        compound: bool = True,
        periods_per_year: int = 252,
        prepare_returns: bool = True,
        match_dates: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        The metrics method calculates the performance metrics of an EquityPortfolio object.

        :param kwargs:
        :return:
        """
        return qs.reports.metrics(
            returns=self.nav.pct_change().dropna(),
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode=mode,
            sep=sep,
            compounded=compound,
            periods_per_year=periods_per_year,
            prepare_returns=prepare_returns,
            match_dates=match_dates,
            **kwargs,
        )

    def plots(
        self,
        benchmark: Any = None,
        grayscale: bool = False,
        figsize: tuple[int, int] = (8, 5),
        mode: str = "basic",
        compounded: bool = True,
        periods_per_year: int = 252,
        prepare_returns: bool = True,
        match_dates: bool = True,
        **kwargs: Any,
    ) -> Any:
        return qs.reports.plots(
            returns=self.nav.pct_change().dropna(),
            benchmark=benchmark,
            grayscale=grayscale,
            figsize=figsize,
            mode=mode,
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=prepare_returns,
            match_dates=match_dates,
            **kwargs,
        )

    def plot(self, kind: Plot, **kwargs: Any) -> Any:
        return kind.plot(returns=self.nav.pct_change().dropna(), **kwargs)

    def html(
        self,
        benchmark: Any = None,
        rf: float = 0.0,
        grayscale: bool = False,
        title: str = "Strategy Tearsheet",
        output: Any = None,
        compounded: bool = True,
        periods_per_year: int = 252,
        download_filename: str = "quantstats-tearsheet.html",
        figfmt: str = "svg",
        template_path: Any = None,
        match_dates: bool = True,
        **kwargs: Any,
    ) -> Any:
        return qs.reports.html(
            returns=self.nav.pct_change().dropna(),
            benchmark=benchmark,
            rf=rf,
            grayscale=grayscale,
            title=title,
            output=output,
            compounded=compounded,
            periods_per_year=periods_per_year,
            download_filename=download_filename,
            figfmt=figfmt,
            template_path=template_path,
            match_dates=match_dates,
            **kwargs,
        )

    def snapshot(
        self,
        grayscale: bool = False,
        figsize: tuple[int, int] = (10, 8),
        title: str = "Portfolio Summary",
        fontname: str = "Arial",
        lw: float = 1.5,
        mode: str = "comp",
        subtitle: bool = True,
        savefig: Any = None,
        show: bool = True,
        log_scale: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        The snapshot method creates a snapshot of the performance of an EquityPortfolio object.

        :param grayscale:
        :param figsize:
        :param title:
        :param fontname:
        :param lw:
        :param mode:
        :param subtitle:
        :param savefig:
        :param show:
        :param log_scale:
        :param kwargs:
        :return:
        """
        return qs.plots.snapshot(
            returns=self.nav.pct_change().dropna(),
            grayscale=grayscale,
            figsize=figsize,
            title=title,
            fontname=fontname,
            lw=lw,
            mode=mode,
            subtitle=subtitle,
            savefig=savefig,
            show=show,
            log_scale=log_scale,
            **kwargs,
        )
