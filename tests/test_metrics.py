# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.simulator.metrics import Metrics


def test_length_one():
    """
    Test that the standard deviation is nan if the length is one
    """
    profit = pd.Series(data=[1.0])
    m = Metrics(daily_profit=profit)

    assert np.isnan(m.std_profit)
    assert np.isnan(m.sr_profit)


def test_index_wrong_order():
    """
    Test that the index is in the correct order
    """
    profit = pd.Series(index=[5, 4], data=[2.0, 3.0])
    with pytest.raises(AssertionError):
        Metrics(profit)


def test_nan_value():
    """
    Test that nan values are not allowed
    """
    profit = pd.Series(index=[1, 2], data=[np.NaN, 2.0])
    with pytest.raises(AssertionError):
        Metrics(profit)


def test_profit(portfolio):
    """
    Test that the profit is computed correctly
    :param portfolio: the portfolio object (fixture)
    """
    m = Metrics(daily_profit=portfolio.profit)
    pd.testing.assert_series_equal(m.daily_profit, portfolio.profit)

    assert m.mean_profit == pytest.approx(-5.810981697171386)
    assert m.std_profit == pytest.approx(840.5615726803527)
    assert m.total_profit == pytest.approx(-3492.4000000000033)
    assert m.sr_profit == pytest.approx(-0.10974386369939439)
