from dataclasses import dataclass

import pandas as pd
import pytest

from cvx.simulator._abc.state import State


@dataclass
class TestState(State):
    @State.position.setter
    def position(self, position):
        self._position = position


@pytest.fixture()
def state(prices):
    return TestState(prices=prices.iloc[0])


def test_assets_partial(prices):
    state = TestState(prices=prices[["A", "B", "C"]].iloc[0])
    pd.testing.assert_index_equal(state.assets, pd.Index(["A", "B", "C"]))


def test_assets_full(prices):
    state = TestState(prices=prices.iloc[0])
    pd.testing.assert_index_equal(state.assets, prices.columns)


def test_trade_no_init_pos(state):
    x = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}).sub(state.position, fill_value=0)
    pd.testing.assert_series_equal(x, pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))


def test_gap(prices):
    state = TestState(prices=prices.iloc[0])
    assert state.days == 0
    state.time = prices.index[0]
    assert state.days == 0
    state.time = prices.index[1]
    assert state.days == 1


def test_gmv(state):
    assert state.gmv == 0.0


def test_trades(state):
    state._trades = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    pd.testing.assert_series_equal(
        state.trades, pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    )
    assert state.gross.sum() == -206047.2


def test_set_position(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    pd.testing.assert_series_equal(
        state.position, pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    )


def test_value(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    assert state.value == pytest.approx(-206047.2)
