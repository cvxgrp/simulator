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

import pandas as pd

from .._abc.portfolio import Portfolio


@dataclass(frozen=True)
class EquityPortfolio(Portfolio):
    """A class that represents an equity portfolio
    and contains dataframes for prices and stock holdings,
    as well as optional parameters for trading cost models
    and initial cash values.

    Attributes:
        prices (pd.DataFrame): A pandas dataframe representing
        the prices of various assets held by the portfolio over time.
        units (pd.DataFrame): A pandas dataframe representing the number of shares
        held for each asset in the portfolio over time.

    Notes: The EquityPortfolio class is designed to represent
    a portfolio of assets where only equity positions are held.
    The prices and units dataframes are assumed to have the same
    index object representing the available time periods for which data is available.
    If no initial cash value is provided, the initial_cash attribute
    will be set to a default value of 1,000,000."""

    cash: pd.Series | float

    @property
    def cashflow(self):
        """
        The cashflow property returns the cash flow of the portfolio.
        """
        flow = self.cash.diff()
        flow.iloc[0] = -self.cash.iloc[0]  # - self.initial_cash
        return flow

    @property
    def nav(self) -> pd.Series:
        """Returns a pandas series representing the total value
        of the portfolio's investments and cash.

        Returns: pd.Series: A pandas series representing the
                            total value of the portfolio's investments and cash.
        """
        if isinstance(self.cash, pd.Series):
            return self.equity.sum(axis=1) + self.cash

        profit = (self.cashposition.shift(1) * self.returns.fillna(0.0)).sum(axis=1)
        return profit.cumsum() + self.cash
