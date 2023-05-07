# -*- coding: utf-8 -*-
"""
Popular year vs month performance table.

Supports compounded and cumulative (i.e. fixed AUM) returns logic.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd


def monthlytable(returns: pd.Series):
    """
    Get a table of monthly returns.

    Args:
        returns: Series of individual returns.

    Returns:
        DataFrame with monthly returns, their STDev and YTD.
    """

    def _compound(rets):
        """
        Helper function for compounded return calculation.

        Args:
            rets: Series of individual returns.

        Returns:
            Series of compounded returns.
        """
        return (1.0 + rets).prod() - 1.0

    # Works better in the first month
    # Compute all the intramonth-returns, instead of reapplying some monthly resampling of the NAV
    return_monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
        _compound
    )

    frame = return_monthly.unstack(level=1).rename(
        columns=lambda x: calendar.month_abbr[x]
    )

    ytd = frame.apply(_compound, axis=1)
    frame["STDev"] = np.sqrt(12) * frame.std(axis=1)
    # make sure that you don't include the column for the STDev in your computation
    frame["YTD"] = ytd
    frame.index.name = "Year"
    frame.columns.name = None
    # most recent years on top
    return frame.iloc[::-1]
