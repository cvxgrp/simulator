# -*- coding: utf-8 -*-
"""
Popular year vs month performance table.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd


def _compound(rets):
    """
    Helper function for compounded return calculation.
    """
    return (1.0 + rets).prod() - 1.0


def monthlytable(returns: pd.Series):
    """
    Get a table of monthly returns.

    Args:
        returns: Series of individual returns.

    Returns:
        DataFrame with monthly returns, their STDev and YTD.
    """

    # Works better in the first month
    # Compute all the intramonth-returns, instead of reapplying some monthly resampling of the NAV
    returns = returns.dropna()

    return_monthly = (
        returns.groupby([returns.index.year, returns.index.month])
        .apply(_compound)
        .unstack(level=1)
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
