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
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cvx.simulator.utils.metric import sharpe
from cvx.simulator.utils.rescale import returns2prices


@dataclass(frozen=True)
class Portfolio:
    prices: pd.DataFrame
    units: pd.DataFrame
    aum: float | pd.Series

    def __post_init__(self) -> None:
        """A class method that performs input validation after object initialization.
        Notes: The post_init method is called after an instance of the Portfolio
        class has been initialized, and performs a series of input validation
        checks to ensure that the prices and units dataframes are in the
        expected format with no duplicates or missing data,
        and that the units dataframe represents valid equity positions
        for the assets held in the portfolio.
        Specifically, the method checks that both the prices and units dataframes
        have a monotonic increasing and unique index,
        and that the index and columns of the units dataframe are subsets
        of the index and columns of the prices dataframe, respectively.
        If any of these checks fail, an assertion error will be raised."""

        assert self.prices.index.is_monotonic_increasing
        assert self.prices.index.is_unique
        assert self.units.index.is_monotonic_increasing
        assert self.units.index.is_unique

        assert set(self.units.index).issubset(set(self.prices.index))
        assert set(self.units.columns).issubset(set(self.prices.columns))

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
    def nav(self) -> pd.Series:
        """Return a pandas series representing the NAV"""
        if isinstance(self.aum, pd.Series):
            series = self.aum
        else:
            profit = (self.cashposition.shift(1) * self.returns.fillna(0.0)).sum(axis=1)
            series = profit.cumsum() + self.aum

        series.name = "NAV"
        return series

    @property
    def profit(self) -> pd.Series:
        """A property that returns a pandas series representing the
        profit gained or lost in the portfolio based on changes in asset prices.

        Returns: pd.Series: A pandas series representing the profit
        gained or lost in the portfolio based on changes in asset prices.
        """
        series = (self.cashposition.shift(1) * self.returns.fillna(0.0)).sum(axis=1)
        series.name = "Profit"
        return series

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
        series = self.nav.expanding(min_periods=1).max()
        series.name = "Highwater"
        return series

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
        series = 1.0 - self.nav / self.highwater
        series.name = "Drawdown"
        return series

    @property
    def cashposition(self):
        """
        Return a pandas dataframe representing the cash position
        """
        return self.prices * self.units

    @property
    def returns(self):
        """
        The returns property exposes the returns of the individual assets
        """
        return self.prices.pct_change()

    @property
    def trades_units(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing the trades made in the portfolio in terms of units.

        Returns: pd.DataFrame: A pandas dataframe representing the trades made in the portfolio in terms of units.

        Notes: The function calculates the trades made by the portfolio by taking
        the difference between the current and previous values of the units dataframe.
        The resulting values will represent the number of shares of each asset
        bought or sold by the portfolio at each point in time.
        The resulting dataframe will have the same dimensions
        as the units dataframe, with NaN values filled with zeros."""
        t = self.units.fillna(0.0).diff()
        t.loc[self.index[0]] = self.units.loc[self.index[0]]
        return t.fillna(0.0)

    @property
    def trades_currency(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing
        the trades made in the portfolio in terms of currency.

        Returns: pd.DataFrame: A pandas dataframe representing the trades made in the portfolio in terms of currency.

        Notes: The function calculates the trades made in currency by multiplying
        the number of shares of each asset bought or sold (as represented in the trades_units dataframe)
        with the current prices of each asset (as represented in the prices dataframe).
        The resulting dataframe will have the same dimensions as the units and prices dataframes.
        """
        return self.trades_units * self.prices

    @property
    def turnover_relative(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing the turnover
        relative to the NAV. Can be positive (if bought) or negative (if sold)."""
        return self.trades_currency.div(self.nav, axis=0)

    @property
    def turnover(self) -> pd.DataFrame:
        return self.trades_currency.abs()

    def __getitem__(self, time: datetime) -> pd.Series:
        """The `__getitem__` method retrieves the stock data for a specific time in the dataframe.
        It returns the stock data for that time.

        The method takes one input parameter:
        - `time`: the time index for which to retrieve the stock data

        Returns:
        - stock data for the input time

        Note that the input time must be in the index of the dataframe,
        otherwise a KeyError will be raised."""
        return self.units.loc[time]

    @property
    def equity(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe
        representing the equity positions of the portfolio,
        which is the value of each asset held by the portfolio.
        Returns: pd.DataFrame: A pandas dataframe representing
        the equity positions of the portfolio.

        Notes: The function calculates the equity of the portfolio
        by multiplying the current prices of each asset
        by the number of shares held by the portfolio.
        The equity dataframe will have the same dimensions
        as the prices and units dataframes."""

        return self.cashposition

    @property
    def weights(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing
        the weights of various assets in the portfolio.

        Returns: pd.DataFrame: A pandas dataframe representing the weights
        of various assets in the portfolio.

        Notes: The function calculates the weights of various assets
        in the portfolio by dividing the equity positions
        for each asset (as represented in the equity dataframe)
        by the total portfolio value (as represented in the nav dataframe).
        Both dataframes are assumed to have the same dimensions.
        The resulting dataframe will show the relative weight
        of each asset in the portfolio at each point in time."""
        return self.equity.apply(lambda x: x / self.nav)

    def snapshot(
        self,
        benchmark: Any = None,
        title: str = "Portfolio Summary",
        table: pd.DataFrame = None,
        aggregate: bool = False,
        log_scale: bool = False,
        label_strategy: str = "Strategy",
        label_benchmark: str = "Benchmark",
    ) -> Any:
        """
        The snapshot method creates a snapshot of the performance of a Portfolio object.
        """
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            row_heights=[0.13, 0.47, 0.2, 0.2],
            specs=[
                [{"type": "table"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}],
                [{"type": "bar"}],
            ],
        )

        # fig = make_subplots(
        #    rows=3, cols=1,
        #    shared_xaxes=True,
        #    vertical_spacing=0.03,
        #    specs=[[{"type": "table"}],
        #           [{"type": "scatter"}],
        #           [{"type": "scatter"}]]
        # )

        # display the NAV
        fig.add_trace(
            go.Scatter(x=self.nav.index, y=self.nav, name=label_strategy), row=2, col=1
        )

        # change the title of the yaxis
        fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)

        # change yaxis to log scale
        if log_scale:
            fig.update_yaxes(type="log", row=2, col=1)

        # Drawdown data
        fig.add_trace(
            go.Scatter(
                x=self.drawdown.index, y=-self.drawdown, name="Drawdown", fill="tozeroy"
            ),
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="Drawdown", row=3, col=1)

        # Daily/Monthly Returns
        if aggregate:
            self.nav.index.name = "date"
            nav = self.nav.resample("M").last()
            returns = 100 * nav.pct_change().dropna()

            df = returns.to_frame("value")
            df["color"] = np.where(df["value"] >= 0, "green", "red")

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["value"],
                    xperiod="M1",
                    xperiodalignment="middle",
                    marker_color=df["color"],
                    name="Monthly Returns",
                ),
                row=4,
                col=1,
            )
            fig.update_yaxes(title_text="Monthly Returns", row=4, col=1)

        else:
            self.nav.index.name = "date"
            returns = 100 * self.nav.pct_change().dropna()

            df = returns.to_frame("value")  # .reset_index()
            df["color"] = np.where(df["value"] >= 0, "green", "red")

            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["value"],
                    marker_color=df["color"],
                    name="Daily Returns",
                ),
                row=4,
                col=1,
            )
            fig.update_yaxes(title_text="Daily Returns", row=4, col=1)

        fig.update_traces(showlegend=True)
        fig.update_layout(height=800, title_text=title)

        if benchmark is not None:
            fig.add_trace(
                go.Scatter(x=benchmark.index, y=benchmark.values, name=label_benchmark),
                row=2,
                col=1,
            )

        if table is None:
            table = pd.DataFrame(
                index=[label_strategy],
                columns=["start", "end", "# assets", "Sharpe ratio"],
            )
            table.index.name = "Portfolio"

            table["start"][label_strategy] = self.nav.index[0].strftime("%Y-%m-%d")
            table["end"][label_strategy] = self.nav.index[-1].strftime("%Y-%m-%d")
            # table["start"][label_benchmark] = benchmark.index[0].strftime("%Y-%m-%d")
            # table["end"][label_benchmark] = benchmark.index[-1].strftime("%Y-%m-%d")
            table["# assets"][label_strategy] = len(self.assets)
            # table['# assets'][label_benchmark] =
            table["Sharpe ratio"][label_strategy] = sharpe(
                self.nav.pct_change().dropna()
            )
            # table["Sharpe ratio"][label_benchmark] = sharpe(benchmark.pct_change().dropna())

            if benchmark is not None:
                table_bench = pd.DataFrame(
                    index=[label_benchmark],
                    columns=["start", "end", "# assets", "Sharpe ratio"],
                )
                table_bench["start"][label_benchmark] = benchmark.index[0].strftime(
                    "%Y-%m-%d"
                )
                table_bench["end"][label_benchmark] = benchmark.index[-1].strftime(
                    "%Y-%m-%d"
                )
                table_bench["# assets"][label_benchmark] = ""
                table_bench["Sharpe ratio"][label_benchmark] = sharpe(
                    benchmark.pct_change().dropna()
                )

                table = pd.concat([table, table_bench], axis=0)

            table = table.reset_index()

        # print(table)
        # assert False

        # if table is not None:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(table.columns), font=dict(size=10), align="left"
                ),
                cells=dict(
                    values=[table[column].values for column in table.columns],
                    align="left",
                ),
            ),
            row=1,
            col=1,
        )

        return fig

    @classmethod
    def from_cashpos_prices(
        cls, prices: pd.DataFrame, cashposition: pd.DataFrame, aum: float
    ):
        """Build Futures Portfolio from cashposition"""
        units = cashposition.div(prices, fill_value=0.0)
        return cls(prices=prices, units=units, aum=aum)

    @classmethod
    def from_cashpos_returns(
        cls, returns: pd.DataFrame, cashposition: pd.DataFrame, aum: float
    ):
        """Build Futures Portfolio from cashposition"""
        prices = returns2prices(returns)
        return cls.from_cashpos_prices(prices, cashposition, aum)
