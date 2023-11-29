import pandas as pd

from cvx.simulator.builder import _State


def test_trade(prices):
    s = _State(prices=prices)
    s._position = pd.Series({"A": 10, "B": 20, "C": -10})
    x = s.trade(pd.Series({"B": 25, "C": -15, "D": 40}))
    pd.testing.assert_series_equal(
        x, pd.Series({"A": -10.0, "B": 5.0, "C": -5.0, "D": 40.0})
    )


def test_trade_no_init_pos(prices):
    s = _State(prices=prices)
    x = s.trade(pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))
    pd.testing.assert_series_equal(x, pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))


def test_update(prices):
    s = _State(prices=prices.iloc[0])
    assert s.cash == 1e6
    s.update(position=pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))
    assert s.cash == 1206047.2
    assert s.value == -206047.2
    assert s.nav == 1e6
