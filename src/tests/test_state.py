"""
Tests for the State class in the cvx.simulator package.

This module contains tests for the State class, which represents the current state
of a portfolio during simulation. The tests verify that the State correctly tracks
positions, prices, cash, and other portfolio metrics, and that it updates these
values correctly when the state changes.
"""

import numpy as np
import pandas as pd
import pytest

from cvx.simulator import State


@pytest.fixture()
def prices() -> pd.DataFrame:
    """
    Create a price data fixture for testing.

    This fixture creates a DataFrame with price data for four assets (A, B, C, D)
    at two time points (2020-01-01 and 2020-01-30).

    Returns
    -------
    pd.DataFrame
        DataFrame with price data for testing
    """
    return pd.DataFrame(
        columns=["A", "B", "C", "D"],
        index=pd.Index([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-30")]),
        data=[[1.0, 2.0, 3.0, 4.0], [1.3, 2.3, 2.7, 4.2]],
    )


@pytest.fixture()
def state(prices: pd.DataFrame) -> State:
    """
    Create a State fixture for testing.

    This fixture creates a State instance and initializes it with the first
    row of price data from the prices fixture.

    Parameters
    ----------
    prices : pd.DataFrame
        The price data fixture

    Returns
    -------
    State
        A State instance initialized with the first row of price data
    """
    state = State()
    state.prices = prices.iloc[0]
    return state


def test_assets_partial(prices: pd.DataFrame) -> None:
    """
    Test that the assets property works with a subset of assets.

    This test verifies that the State.assets property correctly returns
    the subset of assets for which prices have been set.

    Parameters
    ----------
    prices : pd.DataFrame
        The price data fixture
    """
    state = State()
    state.prices = prices[["A", "B", "C"]].iloc[0]
    pd.testing.assert_index_equal(state.assets, pd.Index(["A", "B", "C"]))


def test_assets_full(state: State, prices: pd.DataFrame) -> None:
    """
    Test that the assets property works with all assets.

    This test verifies that the State.assets property correctly returns
    all assets from the price data when all prices have been set.

    Parameters
    ----------
    state : State
        The State fixture
    prices : pd.DataFrame
        The price data fixture
    """
    pd.testing.assert_index_equal(state.assets, prices.columns)


def test_trade_no_init_pos(state: State) -> None:
    """
    Test trade calculation with no initial position.

    This test verifies that the State correctly calculates trades when
    there is no initial position (i.e., the initial position is effectively zero).

    Parameters
    ----------
    state : State
        The State fixture
    """
    x = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}).sub(state.position, fill_value=0)
    pd.testing.assert_series_equal(x.dropna(), pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))


def test_gap(state: State, prices: pd.DataFrame) -> None:
    """
    Test that the days property correctly calculates time gaps.

    This test verifies that the State.days property correctly calculates
    the number of days between consecutive time points.

    Parameters
    ----------
    state : State
        The State fixture
    prices : pd.DataFrame
        The price data fixture
    """
    assert state.days == 0
    state.time = prices.index[0]
    assert state.days == 0
    state.time = prices.index[1]
    assert state.days == 29


def test_set_position(state: State) -> None:
    """
    Test that the position property can be set correctly.

    This test verifies that the State.position property can be set
    with a pandas Series and that the values are correctly stored.

    Parameters
    ----------
    state : State
        The State fixture
    """
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    pd.testing.assert_series_equal(state.position.dropna(), pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))


def test_mask() -> None:
    """
    Test that the mask property correctly identifies valid prices.

    This test verifies that the State.mask property correctly identifies
    which assets have valid (non-NaN) prices.
    """
    state = State()
    state.prices = pd.Series({"A": 1.0, "B": 2.0, "C": np.nan})
    np.testing.assert_array_equal(state.mask, np.array([True, True, False]))


def test_init(prices: pd.DataFrame) -> None:
    """
    Test the initialization and updating of a State object.

    This comprehensive test verifies that:
    1. A new State with only AUM set has correct initial values
    2. Setting prices doesn't change the financial values
    3. Setting positions correctly updates cash, value, and weights
    4. Updating prices correctly calculates profit and updates AUM
    5. Setting cash directly works correctly

    Parameters
    ----------
    prices : pd.DataFrame
        The price data fixture
    """
    state = State()
    state.aum = 1e4

    # Test initial state with only AUM set
    assert state.cash == 1e4
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == 0.0
    assert state.profit == 0.0
    assert state.gmv == 0.0
    assert state.leverage == 0.0

    # Test state after setting prices
    state.prices = prices.iloc[0]

    assert state.cash == 1e4
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == 0.0
    assert state.profit == 0.0
    assert state.gmv == 0.0
    assert state.leverage == 0.0

    # Test state after updating position and weights
    state.position = np.ones(len(state.assets))

    assert state.cash == 1e4 - prices.iloc[0].sum()
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == prices.iloc[0].sum()
    assert state.profit == 0.0
    pd.testing.assert_series_equal(state.trades, pd.Series(1.0, index=state.assets))

    pd.testing.assert_series_equal(state.weights, prices.iloc[0].div(state.nav), check_names=False)

    assert state.leverage == pytest.approx(state.weights.abs().sum())

    # Test state after updating prices
    state.prices = prices.iloc[1]
    assert state.profit == 0.5
    assert state.cash == 1e4 - prices.iloc[0].sum()
    assert state.aum == 1e4 + state.profit
    assert state.nav == 1e4 + state.profit
    assert state.value == prices.iloc[1].sum()

    # Test state after updating cash
    state.cash = 1.01 * state.cash
    assert state.cash == 1.01 * (1e4 - prices.iloc[0].sum())


def test_cash_set_aum(state: State) -> None:
    """
    Test setting AUM directly.

    This test verifies that setting the AUM property directly correctly
    updates the AUM, cash, and NAV values.

    Parameters
    ----------
    state : State
        The State fixture
    """
    state.aum = 1e6
    assert state.aum == 1e6
    assert state.cash == 1e6
    assert state.nav == 1e6


def test_cash_set_cash(state: State) -> None:
    """
    Test setting cash directly.

    This test verifies that setting the cash property directly correctly
    updates the AUM, cash, and NAV values.

    Parameters
    ----------
    state : State
        The State fixture
    """
    state.cash = 1e6
    assert state.aum == 1e6
    assert state.cash == 1e6
    assert state.nav == 1e6


def test_before_price() -> None:
    """
    Test an empty state before the prices are set.

    This test verifies that a newly created State object has the expected
    default values for all properties before any prices are set.
    """
    state = State()
    pd.testing.assert_index_equal(state.assets, pd.Index([], dtype=str))
    np.testing.assert_array_equal(state.mask, np.array([]))
    pd.testing.assert_series_equal(state.prices, pd.Series(dtype=float))
    pd.testing.assert_series_equal(state.weights, pd.Series(dtype=float))
    pd.testing.assert_series_equal(state.position, pd.Series(dtype=float), check_index_type=False)
    pd.testing.assert_series_equal(state.cashposition, pd.Series(dtype=float))
    assert state.value == 0.0
    assert state.aum == 0.0
    assert state.cash == 0.0
    assert state.leverage == 0.0
    assert state.days == 0
    assert state.time is None
    assert state.gmv == 0.0
    assert state.nav == 0.0


def test_after_price(prices: pd.DataFrame) -> None:
    """
    Test a state after the first prices are set.

    This test verifies that a State object has the expected values for all
    properties after prices are set but before any positions are established.

    Parameters
    ----------
    prices : pd.DataFrame
        The price data fixture
    """
    state = State()
    state.prices = prices.iloc[0]
    state.time = prices.index[0]

    pd.testing.assert_index_equal(state.assets, pd.Index(["A", "B", "C", "D"], dtype=str))
    np.testing.assert_array_equal(state.mask, np.array([True, True, True, True]))
    pd.testing.assert_series_equal(state.prices, prices.iloc[0])
    pd.testing.assert_series_equal(state.weights, pd.Series(index=["A", "B", "C", "D"]))
    pd.testing.assert_series_equal(state.position, pd.Series(index=["A", "B", "C", "D"]))
    pd.testing.assert_series_equal(state.cashposition, pd.Series(index=["A", "B", "C", "D"]))
    assert state.value == 0.0
    assert state.aum == 0.0
    assert state.cash == 0.0
    assert state.leverage == 0.0
    assert state.days == 0
    assert state.time == prices.index[0]
    assert state.gmv == 0.0
    assert state.nav == 0.0
