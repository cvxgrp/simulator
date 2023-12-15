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

import pandas as pd

from cvx.simulator.portfolio import Portfolio


@dataclass(frozen=True)
class EquityPortfolio(Portfolio):
    """A class that represents an equity portfolio
    and contains dataframes for prices and stock holdings,
    as well as optional parameters for trading cost models
    and initial cash values.

    Attributes:
        prices (pd.DataFrame): A pandas dataframe representing
        the prices of various assets held by the portfolio over time.
        stocks (pd.DataFrame): A pandas dataframe representing the number of shares
        held for each asset in the portfolio over time.
        trading_cost_model (TradingCostModel): An optional trading cost model
        to use when trading assets in the portfolio.
        initial_cash (float): An optional scalar float representing the initial
        cash value available for the portfolio.

    Notes: The EquityPortfolio class is designed to represent
    a portfolio of assets where only equity positions are held.
    The prices and stocks dataframes are assumed to have the same
    index object representing the available time periods for which data is available.
    If no trading cost model is provided, the trading_cost_model attribute
    will be set to None by default.
    If no initial cash value is provided, the initial_cash attribute
    will be set to a default value of 1,000,000."""

    cash: pd.Series

    def __post_init__(self) -> None:
        """A class method that performs input validation after object initialization.
        Notes: The post_init method is called after an instance of the EquityPortfolio class has been initialized,
        and performs a series of input validation checks to ensure that the prices
        and stocks dataframes are in the expected format
        with no duplicates or missing data,
        and that the stocks dataframe represents valid equity positions
        for the assets held in the portfolio.
        Specifically, the method checks that both the prices and stocks dataframes
        have a monotonic increasing and unique index,
        and that the index and columns of the stocks dataframe are subsets
        of the index and columns of the prices dataframe, respectively.
        If any of these checks fail, an assertion error will be raised."""

        assert self.prices.index.is_monotonic_increasing
        assert self.prices.index.is_unique
        assert self.stocks.index.is_monotonic_increasing
        assert self.stocks.index.is_unique

        assert set(self.stocks.index).issubset(set(self.prices.index))
        assert set(self.stocks.columns).issubset(set(self.prices.columns))

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

    @property
    def cashflow(self):
        """
        The cashflow property returns the cash flow of the portfolio.
        """
        flow = self.cash.diff()
        flow.iloc[0] = -self.cash.iloc[0]  # - self.initial_cash
        return flow

    def __getitem__(self, time: datetime) -> pd.Series:
        """The `__getitem__` method retrieves the stock data for a specific time in the dataframe.
        It returns the stock data for that time.

        The method takes one input parameter:
        - `time`: the time index for which to retrieve the stock data

        Returns:
        - stock data for the input time

        Note that the input time must be in the index of the dataframe,
        otherwise a KeyError will be raised."""
        return self.stocks.loc[time]

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
        The resulting values are filled forward to account
        for any missing data or NaN values.
        The equity dataframe will have the same dimensions
        as the prices and stocks dataframes."""

        return self.prices * self.stocks

    @property
    def trades_stocks(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing the trades made in the portfolio in terms of stocks.

        Returns: pd.DataFrame: A pandas dataframe representing the trades made in the portfolio in terms of stocks.

        Notes: The function calculates the trades made by the portfolio by taking
        the difference between the current and previous values of the stocks dataframe.
        The resulting values will represent the number of shares of each asset
        bought or sold by the portfolio at each point in time.
        The resulting dataframe will have the same dimensions
        as the stocks dataframe, with NaN values filled with zeros."""
        t = self.stocks.fillna(0.0).diff()
        t.loc[self.index[0]] = self.stocks.loc[self.index[0]]
        return t.fillna(0.0)

    @property
    def trades_currency(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing
        the trades made in the portfolio in terms of currency.

        Returns: pd.DataFrame: A pandas dataframe representing the trades made in the portfolio in terms of currency.

        Notes: The function calculates the trades made in currency by multiplying
        the number of shares of each asset bought or sold (as represented in the trades_stocks dataframe)
        with the current prices of each asset (as represented in the prices dataframe).
        Uses pandas ffill() method to forward fill NaN values in the prices dataframe.
        The resulting dataframe will have the same dimensions as the stocks and prices dataframes.
        """
        return self.trades_stocks * self.prices

    @property
    def turnover(self) -> pd.DataFrame:
        return self.trades_currency.abs()

    @property
    def nav(self) -> pd.Series:
        """Returns a pandas series representing the total value
        of the portfolio's investments and cash.

        Returns: pd.Series: A pandas series representing the
                            total value of the portfolio's investments and cash.
        """
        return self.equity.sum(axis=1) + self.cash

    @property
    def profit(self) -> pd.Series:
        """A property that returns a pandas series representing the
        profit gained or lost in the portfolio based on changes in asset prices.

        Returns: pd.Series: A pandas series representing the profit
        gained or lost in the portfolio based on changes in asset prices.

        Notes: The calculation is based on the difference between
        the previous and current prices of the assets in the portfolio,
        multiplied by the number of stocks in each asset previously held.
        """

        price_changes = self.prices.ffill().diff()
        previous_stocks = self.stocks.shift(1).fillna(0.0)
        return (previous_stocks * price_changes).dropna(axis=0, how="all").sum(axis=1)
