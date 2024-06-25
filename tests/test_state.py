import numpy as np
import pandas as pd
import pytest

from cvx.simulator import State


@pytest.fixture()
def prices():
    """
    Fixture for the prices
    :param resource_dir: the resource directory (fixture)
    """
    return pd.DataFrame(
        columns=["A", "B", "C", "D"],
        index=pd.Index([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-30")]),
        data=[[1.0, 2.0, 3.0, 4.0], [1.3, 2.3, 2.7, 4.2]],
    )


@pytest.fixture()
def state(prices):
    state = State()
    state.prices = prices.iloc[0]
    return state


def test_assets_partial(prices):
    state = State()
    state.prices = prices[["A", "B", "C"]].iloc[0]
    pd.testing.assert_index_equal(state.assets, pd.Index(["A", "B", "C"]))


def test_assets_full(state, prices):
    pd.testing.assert_index_equal(state.assets, prices.columns)


def test_trade_no_init_pos(state):
    x = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}).sub(state.position, fill_value=0)
    pd.testing.assert_series_equal(
        x.dropna(), pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    )


def test_gap(state, prices):
    assert state.days == 0
    state.time = prices.index[0]
    assert state.days == 0
    state.time = prices.index[1]
    assert state.days == 29


def test_set_position(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    pd.testing.assert_series_equal(
        state.position.dropna(), pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    )


def test_mask():
    state = State()
    state.prices = pd.Series({"A": 1.0, "B": 2.0, "C": np.nan})
    np.testing.assert_array_equal(state.mask, np.array([True, True, False]))


def test_init(prices):
    state = State()
    state.aum = 1e4

    assert state.cash == 1e4
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == 0.0
    assert state.profit == 0.0
    assert state.gmv == 0.0
    assert state.leverage == 0.0

    state.prices = prices.iloc[0]

    assert state.cash == 1e4
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == 0.0
    assert state.profit == 0.0
    assert state.gmv == 0.0
    assert state.leverage == 0.0

    # update position and weights
    state.position = np.ones(len(state.assets))

    assert state.cash == 1e4 - prices.iloc[0].sum()
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == prices.iloc[0].sum()
    assert state.profit == 0.0
    pd.testing.assert_series_equal(state.trades, pd.Series(1.0, index=state.assets))

    pd.testing.assert_series_equal(
        state.weights, prices.iloc[0].div(state.nav), check_names=False
    )

    assert state.leverage == pytest.approx(state.weights.abs().sum())

    # update prices
    state.prices = prices.iloc[1]
    assert state.profit == 0.5
    assert state.cash == 1e4 - prices.iloc[0].sum()
    assert state.aum == 1e4 + state.profit
    assert state.nav == 1e4 + state.profit
    assert state.value == prices.iloc[1].sum()

    # update cash
    state.cash = 1.01 * state.cash
    assert state.cash == 1.01 * (1e4 - prices.iloc[0].sum())


def test_cash_set_aum(state):
    state.aum = 1e6
    assert state.aum == 1e6
    assert state.cash == 1e6
    assert state.nav == 1e6


def test_cash_set_cash(state):
    state.cash = 1e6
    assert state.aum == 1e6
    assert state.cash == 1e6
    assert state.nav == 1e6


def test_before_price():
    """
    Test an empty state before the prices are set.
    """
    state = State()
    pd.testing.assert_index_equal(state.assets, pd.Index([], dtype=str))
    np.testing.assert_array_equal(state.mask, np.array([]))
    pd.testing.assert_series_equal(state.prices, pd.Series(dtype=float))
    pd.testing.assert_series_equal(state.weights, pd.Series(dtype=float))
    pd.testing.assert_series_equal(
        state.position, pd.Series(dtype=float), check_index_type=False
    )
    pd.testing.assert_series_equal(state.cashposition, pd.Series(dtype=float))
    assert state.value == 0.0
    assert state.aum == 0.0
    assert state.cash == 0.0
    assert state.leverage == 0.0
    assert state.days == 0
    assert state.time is None
    assert state.gmv == 0.0
    assert state.nav == 0.0


def test_after_price(prices):
    """
    Test a state after the first prices are set.
    """
    state = State()
    state.prices = prices.iloc[0]
    state.time = prices.index[0]

    pd.testing.assert_index_equal(
        state.assets, pd.Index(["A", "B", "C", "D"], dtype=str)
    )
    np.testing.assert_array_equal(state.mask, np.array([True, True, True, True]))
    pd.testing.assert_series_equal(state.prices, prices.iloc[0])
    pd.testing.assert_series_equal(state.weights, pd.Series(index=["A", "B", "C", "D"]))
    pd.testing.assert_series_equal(
        state.position, pd.Series(index=["A", "B", "C", "D"])
    )
    pd.testing.assert_series_equal(
        state.cashposition, pd.Series(index=["A", "B", "C", "D"])
    )
    assert state.value == 0.0
    assert state.aum == 0.0
    assert state.cash == 0.0
    assert state.leverage == 0.0
    assert state.days == 0
    assert state.time == prices.index[0]
    assert state.gmv == 0.0
    assert state.nav == 0.0
