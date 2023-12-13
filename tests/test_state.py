import numpy as np
import pandas as pd

from cvx.simulator.state import State


def test_trade(prices):
    s = State(prices=prices[["A", "B", "C"]].iloc[0])
    s.time = prices.index[0]
    print(s.assets)

    s.position = np.array([10, 20, -10])
    print(s.position)

    x = pd.Series({"B": 25, "C": -15, "D": 40}).sub(s.position, fill_value=0)
    pd.testing.assert_series_equal(
        x, pd.Series({"A": -10.0, "B": 5.0, "C": -5.0, "D": 40.0})
    )


def test_trade_no_init_pos(prices):
    s = State(prices=prices.iloc[0])
    x = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}).sub(s.position, fill_value=0)
    pd.testing.assert_series_equal(x, pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))


def test_update(prices):
    s = State(prices=prices.iloc[0])
    assert s.cash == 1e6
    s.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    assert s.cash == 1206047.2
    assert s.value == -206047.2
    assert s.nav == 1e6
    assert s.gmv == 1670455.8


def test_state():
    prices = pd.Series(data=[2.0, 3.0])
    positions = pd.Series(data=[100, 300])
    cash = 400
    state = State(cash=cash, prices=prices)
    state.position = positions
    # value is the money in stocks
    assert state.value == 1100.0
    # nav is the value plus the cash
    assert state.nav == 400.0
    # weights are the positions divided by the value
    pd.testing.assert_series_equal(
        state.weights, pd.Series(data=[2.0 / 4.0, 9.0 / 4.0])
    )
    # leverage is the value divided by the nav
    assert state.leverage == 11.0 / 4.0
