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
Popular year vs month performance table.

This module provides functions for creating performance tables that show
returns by month and year, supporting both compounded and cumulative
(i.e., fixed AUM) returns logic.
"""

from __future__ import annotations

import calendar
from enum import Enum

import numpy as np
import pandas as pd


def _compound_returns(returns: pd.Series) -> float:
    """
    Calculate compounded returns from a series of individual returns.

    Parameters
    ----------
    returns : pd.Series
        Series of individual returns (as decimals, not percentages)

    Returns
    -------
    float
        Compounded return over the entire period

    Examples
    --------
    >>> import pandas as pd
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> _compound_returns(returns)
    0.0501...  # (1.01 * 1.02 * 0.99 * 1.03) - 1
    """
    return (1.0 + returns).prod() - 1.0


def _cumulative_returns(returns: pd.Series) -> float:
    """
    Calculate cumulative returns from a series of individual returns.

    Parameters
    ----------
    returns : pd.Series
        Series of individual returns (as decimals, not percentages)

    Returns
    -------
    float
        Sum of returns over the entire period

    Examples
    --------
    >>> import pandas as pd
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> _cumulative_returns(returns)
    0.05  # 0.01 + 0.02 + (-0.01) + 0.03
    """
    return returns.sum()


class Aggregate(Enum):
    """
    Enumeration of aggregation methods for returns.

    Members
    -------
    COMPOUND : Callable
        Aggregates returns using compounding (multiplication of (1+r))
    CUMULATIVE : Callable
        Aggregates returns by simple addition
    """

    COMPOUND = _compound_returns
    CUMULATIVE = _cumulative_returns


def monthlytable(returns: pd.Series, f: Aggregate) -> pd.DataFrame:
    """
    Create a table of monthly returns with yearly aggregation.

    This function takes a series of returns with a datetime index and creates
    a table showing returns by month (columns) and year (rows). It also includes
    yearly statistics like standard deviation and year-to-date returns.

    Parameters
    ----------
    returns : pd.Series
        Series of individual returns with a datetime index
    f : Aggregate
        Aggregation method to use (COMPOUND or CUMULATIVE)

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - Rows representing years
        - Columns representing months (Jan-Dec) plus STDev and YTD
        - Values showing the aggregated returns for each month/year
        - Most recent years at the top

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> dates = [datetime(2022, 1, 1), datetime(2022, 2, 1),
    ...          datetime(2022, 3, 1), datetime(2023, 1, 1)]
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates)
    >>> monthlytable(returns, Aggregate.COMPOUND)
           Jan     Feb     Mar  STDev     YTD
    Year
    2023  0.03     NaN     NaN    NaN    0.03
    2022  0.01    0.02   -0.01  0.015    0.02
    """
    # Works better in the first month
    # Compute all the intramonth-returns
    # instead of reapplying some monthly resampling of the NAV
    r = pd.Series(returns)

    return_monthly = r.groupby([r.index.year, r.index.month]).apply(f)

    frame = return_monthly.unstack(level=1).rename(columns=lambda x: calendar.month_abbr[x])

    ytd = frame.apply(f, axis=1)
    frame["STDev"] = np.sqrt(12) * frame.std(axis=1)
    # make sure that you don't include the column for the STDev in your computation
    frame["YTD"] = ytd
    frame.index.name = "Year"
    frame.columns.name = None
    # most recent years on top
    return frame.iloc[::-1]
