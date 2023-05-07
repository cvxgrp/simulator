# -*- coding: utf-8 -*-
"""
Test monthlytable.
"""
from __future__ import annotations

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
        check_index_type=False
    )

    # use quantstats
    ts1 = qs.stats.monthly_returns(series).iloc[::-1]['EOY']
    ts1.index = [int(a) for a in ts1.index]

    # use cvxsimulaor
    ts2 = monthlytable(series)["YTD"]

    pd.testing.assert_series_equal(ts1, ts2, check_names=False)
