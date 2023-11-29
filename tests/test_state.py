import pandas as pd

from cvx.simulator.builder import _State


def test_position_no_pos(prices):
    s = _State(prices=prices)
    x = s.position(assets=["AA", "BB"])
    pd.testing.assert_series_equal(x, pd.Series(index=["AA", "BB"], data=0.0))


def test_position_with_pos(prices):
    s = _State(prices=prices)
    s._position = pd.Series({"A": 10.0, "B": 20.0, "C": -10.0})
    x = s.position(assets=["A", "B"])
    pd.testing.assert_series_equal(x, pd.Series({"A": 10.0, "B": 20.0}))

    x = s.position(assets=["A", "D"])
    pd.testing.assert_series_equal(x, pd.Series({"A": 10.0, "D": 0.0}))


def test_trade(prices):
    s = _State(prices=prices)
    s._position = pd.Series({"A": 10, "B": 20, "C": -10})
    x = s.trade(pd.Series({"B": 25, "C": -15, "D": 40}))
    print(x)
    pd.testing.assert_series_equal(
        x, pd.Series({"A": -10.0, "B": 5.0, "C": -5.0, "D": 40.0})
    )


def test_trade_no_init_pos(prices):
    s = _State(prices=prices)
    x = s.trade(pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))
    print(x)
    pd.testing.assert_series_equal(x, pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))


def test_update(prices):
    s = _State(prices=prices.iloc[0])
    assert s.cash == 1e6
    s.update(position=pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))
    assert s.cash == 1206047.2

    print(s.weights)
    assert s.value == -206047.2
    assert s.nav == 1e6
