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
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from cvx.simulator.types import TIMESERIES


def _compound(rets) -> float:  # type: ignore
    """
    Helper function for compounded return calculation.
    """
    return float((1.0 + rets).prod() - 1.0)


def monthlytable(returns: TIMESERIES) -> pd.DataFrame:
    """
    Get a table of monthly returns.

    Args:
        returns: Series of individual returns.

    Returns:
        DataFrame with monthly returns, their STDev and YTD.
    """

    # Works better in the first month
    # Compute all the intramonth-returns, instead of reapplying some monthly resampling of the NAV
    r = pd.Series(returns)

    r = r.dropna()
    r.index = pd.DatetimeIndex(r.index)

    return_monthly = (
        r.groupby([r.index.year, r.index.month]).apply(_compound).unstack(level=1)
    )

    # make sure all months are in the table!
    frame = pd.DataFrame(index=return_monthly.index, columns=range(1, 13), data=np.NaN)
    frame[return_monthly.columns] = return_monthly

    frame = frame.rename(
        columns={month: calendar.month_abbr[month] for month in frame.columns}
    )

    ytd = frame.apply(_compound, axis=1)
    frame["STDev"] = np.sqrt(12) * frame.std(axis=1)
    # make sure that you don't include the column for the STDev in your computation
    frame["YTD"] = ytd
    frame.index.name = "Year"
    frame.columns.name = None
    # most recent years on top
    return frame.iloc[::-1]
