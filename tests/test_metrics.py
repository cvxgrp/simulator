# -*- coding: utf-8 -*-
# from __future__ import annotations
import pytest


def test_profit(portfolio):
    """
    Test that the profit is computed correctly
    :param portfolio: the portfolio object (fixture)
    """
    # m = Metrics(daily_profit=portfolio.profit)
    # pd.testing.assert_series_equal(m.daily_profit, portfolio.profit)

    # assert m.mean_profit == pytest.approx(-5.810981697171386)
    # assert m.std_profit == pytest.approx(840.5615726803527)
    # assert m.total_profit == pytest.approx(-3492.4000000000033)
    # assert m.sr_profit == pytest.approx(-0.10974386369939439)

    assert portfolio.profit.mean() == pytest.approx(-5.810981697171386)
    assert portfolio.profit.std() == pytest.approx(840.5615726803527)
    assert portfolio.profit.sum() == pytest.approx(-3492.4000000000033)
    assert portfolio.profit.sharpe() == pytest.approx(-0.10974386369939439)
