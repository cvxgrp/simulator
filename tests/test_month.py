# -*- coding: utf-8 -*-
"""
Test monthlytable.
"""
from __future__ import annotations

import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import quantstats as qs

from cvx.simulator.month import monthlytable


def test_table_compounded(resource_dir, returns):
    """
    Test year/month performance table correct.

    Using compounded logic.
    """
    series = returns.fillna(0.0)
    pd.testing.assert_frame_equal(
        monthlytable(series),
        pd.read_csv(resource_dir / "monthtable.csv", index_col=0),
        check_index_type=False,
    )

    # use quantstats
    ts1 = qs.stats.monthly_returns(series).iloc[::-1]["EOY"]
    ts1.index = [int(a) for a in ts1.index]

    # use cvxsimulaor
    ts2 = monthlytable(series)["YTD"]

    pd.testing.assert_series_equal(ts1, ts2, check_names=False, check_index_type=False)


def test_monthlytable():
    """
    Test case with monthly returns of 1%.
    """
    returns = pd.Series(
        [0.01] * 12,
        index=pd.date_range(start=datetime(2020, 1, 1), periods=12, freq="M"),
    )
    expected_output = pd.DataFrame(
        {
            "Jan": [0.01],
            "Feb": [0.01],
            "Mar": [0.01],
            "Apr": [0.01],
            "May": [0.01],
            "Jun": [0.01],
            "Jul": [0.01],
            "Aug": [0.01],
            "Sep": [0.01],
            "Oct": [0.01],
            "Nov": [0.01],
            "Dec": [0.01],
            "STDev": [0.0],
            "YTD": [0.1268250301319699],
        },
        index=[2020],
        columns=[calendar.month_abbr[i] for i in range(1, 13)] + ["STDev", "YTD"],
    )
    pd.testing.assert_frame_equal(
        monthlytable(returns),
        expected_output,
        check_names=False,
        check_index_type=False,
    )


def test_missing():
    """
    Test case with missing value for April
    """
    returns = pd.Series(
        [-0.01, 0.02, 0.03, np.nan, -0.01, 0.02, 0.03, -0.01, 0.02, 0.03, -0.01, 0.02],
        index=pd.date_range(start=datetime(2020, 1, 1), periods=12, freq="M"),
    )
    expected_output = pd.DataFrame(
        {
            "Jan": [-0.01],
            "Feb": [0.02],
            "Mar": [0.03],
            "Apr": [np.nan],
            "May": [-0.01],
            "Jun": [0.02],
            "Jul": [0.03],
            "Aug": [-0.01],
            "Sep": [0.02],
            "Oct": [0.03],
            "Nov": [-0.01],
            "Dec": [0.02],
            "STDev": [0.06161463816629651],
            "YTD": [0.13619569534908815],
        },
        index=[2020],
        columns=[calendar.month_abbr[i] for i in range(1, 13)] + ["STDev", "YTD"],
    )
    pd.testing.assert_frame_equal(
        monthlytable(returns),
        expected_output,
        check_names=False,
        check_index_type=False,
    )


def test_not_a_complete_year():
    """
    Test case with an incomplete year and a missing April
    """
    # We create a pd.Series with only 10 monthly returns and a missing April return
    returns = pd.Series(
        [-0.01, 0.02, 0.03, np.nan, -0.01, 0.02, 0.03, -0.01, 0.02, 0.03],
        index=pd.date_range(start=datetime(2020, 1, 1), periods=10, freq="M"),
    )

    # define the expected output
    expected_output = pd.DataFrame(
        {
            "Jan": [-0.01],
            "Feb": [0.02],
            "Mar": [0.03],
            "Apr": [np.nan],
            "May": [-0.01],
            "Jun": [0.02],
            "Jul": [0.03],
            "Aug": [-0.01],
            "Sep": [0.02],
            "Oct": [0.03],
            "Nov": [np.nan],
            "Dec": [np.nan],
            "STDev": [0.062449979983984036],
            "YTD": [0.12516903876915064],
        },
        index=[2020],
        columns=[calendar.month_abbr[i] for i in range(1, 13)] + ["STDev", "YTD"],
    )
    pd.testing.assert_frame_equal(
        monthlytable(returns),
        expected_output,
        check_names=False,
        check_index_type=False,
    )
