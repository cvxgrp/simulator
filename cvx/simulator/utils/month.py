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
# #    Copyright 2023 Thomas Schmelzer
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.
"""
Popular year vs month performance table.

Supports compounded and cumulative (i.e. fixed AUM) returns logic.
"""
from __future__ import annotations

import calendar
from enum import Enum

import numpy as np
import pandas as pd


def _compound_returns(returns):
    return (1.0 + returns).prod() - 1.0


def _cumulative_returns(returns):
    return returns.sum()


class Aggregate(Enum):
    COMPOUND = _compound_returns
    CUMULATIVE = _cumulative_returns


def monthlytable(returns: pd.Series, f: Aggregate) -> pd.DataFrame:
    """
    Get a table of monthly returns.

    Args:
        returns: Series of individual returns.
        f: Aggregate function to use.

    Returns:
        DataFrame with monthly returns, their STDev and YTD.
    """
    # Works better in the first month
    # Compute all the intramonth-returns
    # instead of reapplying some monthly resampling of the NAV
    r = pd.Series(returns)

    return_monthly = r.groupby([r.index.year, r.index.month]).apply(f)

    frame = return_monthly.unstack(level=1).rename(
        columns=lambda x: calendar.month_abbr[x]
    )

    ytd = frame.apply(f, axis=1)
    frame["STDev"] = np.sqrt(12) * frame.std(axis=1)
    # make sure that you don't include the column for the STDev in your computation
    frame["YTD"] = ytd
    frame.index.name = "Year"
    frame.columns.name = None
    # most recent years on top
    return frame.iloc[::-1]
