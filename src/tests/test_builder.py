"""Tests for the Builder class in the cvx.simulator package.

This module contains tests for the Builder class, which is responsible for
creating portfolios by iterating through time and setting positions. The tests
verify that the Builder correctly handles position setting, weight calculations,
cash management, and portfolio construction.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cvxsimulator import Builder, Portfolio, interpolate


@pytest.fixture()
def builder(prices: pd.DataFrame) -> Builder:
    """Create a Builder fixture for testing.

    This fixture creates a Builder instance with the provided price data
    and a fixed initial AUM of 1,000,000.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    Returns:
    -------
    Builder
        A Builder instance initialized with the provided prices

    """
    return Builder(prices=prices, initial_aum=1e6)


def test_initial_cash(builder: Builder) -> None:
    """Test that the initial AUM is set correctly.

    This test verifies that the Builder correctly initializes with
    the specified initial AUM value.

    Parameters
    ----------
    builder : Builder
        The Builder fixture to test

    """
    assert builder.initial_aum == 1e6


def test_build_empty(builder: Builder, prices: pd.DataFrame) -> None:
    """Test that a newly built portfolio is empty with correct structure.

    This test verifies that a portfolio built from a new Builder (without
    any positions set) has the correct structure: same prices as input,
    NaN units, and zero profit.

    Parameters
    ----------
    builder : Builder
        The Builder fixture to test
    prices : pd.DataFrame
        The price data fixture used to initialize the Builder

    """
    portfolio = builder.build()
    pd.testing.assert_frame_equal(portfolio.prices, prices)
    pd.testing.assert_frame_equal(portfolio.units, np.nan * prices)
    pd.testing.assert_series_equal(portfolio.profit, pd.Series(index=prices.index, data=0.0, name="Profit"))


def test_set_position(prices: pd.DataFrame) -> None:
    """Test setting positions directly in the Builder.

    This test verifies that positions can be set directly in the Builder
    during iteration, and that the resulting portfolio has the expected NAV.

    Parameters
    ----------
    prices : pd.DataFrame
        The price data fixture

    """
    b = Builder(prices=prices[["B", "C"]].head(5), initial_aum=50000)
    for _times, state in b:
        b.position = state.nav / (state.prices * 2)
        assert np.allclose(b.position, state.nav / (state.prices * 2))
        b.aum = state.aum

    portfolio = b.build()
    assert isinstance(portfolio, Portfolio)

    assert portfolio.nav.to_numpy()[-1] == pytest.approx(49773.093729)


def test_set_weights(prices: pd.DataFrame) -> None:
    """Test setting weights in the Builder.

    This test verifies that portfolio weights can be set directly in the Builder
    during iteration, and that the resulting portfolio has the expected NAV.
    The test sets equal weights (50/50) for two assets.

    Parameters
    ----------
    prices : pd.DataFrame
        The price data fixture

    """
    b = Builder(prices=prices[["B", "C"]].head(5), initial_aum=50000)
    for _times, state in b:
        b.weights = np.array((0.5, 0.5))
        assert np.allclose(b.weights, np.array((0.5, 0.5)))
        assert np.allclose(state.weights.values, np.array((0.5, 0.5)))
        b.aum = state.aum  # cash - (state.trades * state.prices).sum()

    portfolio = b.build()
    assert portfolio.nav.to_numpy()[-1] == pytest.approx(49773.093729)


def test_set_cashpositions(prices: pd.DataFrame) -> None:
    """Test setting cash positions in the Builder.

    This test verifies that cash positions can be set directly in the Builder
    during iteration, and that the resulting portfolio has the expected NAV.
    The test allocates equal cash amounts to two assets.

    Parameters
    ----------
    prices : pd.DataFrame
        The price data fixture

    """
    b = Builder(prices=prices[["B", "C"]].head(5), initial_aum=50000)
    for _times, state in b:
        b.cashposition = np.ones((2,)) * state.nav / 2
        assert np.allclose(b.cashposition, np.ones((2,)) * state.nav / 2)
        # b.cash = state.cash - (state.trades * state.prices).sum()
        b.aum = state.aum

    portfolio = b.build()
    assert portfolio.nav.to_numpy()[-1] == pytest.approx(49773.093729)


def test_set_position_again(prices: pd.DataFrame) -> None:
    """Test setting positions directly in the Builder (duplicate test).

    This test is similar to test_set_position and verifies that positions
    can be set directly in the Builder during iteration, and that the
    resulting portfolio has the expected NAV.

    Parameters
    ----------
    prices : pd.DataFrame
        The price data fixture

    """
    b = Builder(prices=prices[["B", "C"]].head(5), initial_aum=50000)
    for _times, state in b:
        b.position = state.nav / (state.prices * 2)
        assert np.allclose(b.position, state.nav / (state.prices * 2))
        # b.cash = state.cash - (state.trades * state.prices).sum()
        b.aum = state.aum

    portfolio = b.build()
    assert portfolio.nav.to_numpy()[-1] == pytest.approx(49773.093729)


def test_weights_on_wrong_days(resource_dir: Any) -> None:
    """Test error handling when setting weights with invalid dimensions.

    This test verifies that the Builder correctly raises ValueError when:
    1. Setting weights with incorrect dimensions
    2. Setting cash positions with incorrect dimensions
    3. Setting positions with incorrect dimensions

    It also verifies that setting weights with correct dimensions works.

    Parameters
    ----------
    resource_dir : Any
        Fixture providing the path to test resources

    """
    prices = pd.read_csv(resource_dir / "priceNaN.csv", index_col=0, parse_dates=True, header=0).apply(interpolate)

    # there are no inner NaNs
    rng = np.random.default_rng(42)

    b = Builder(prices=prices, initial_aum=50000)

    for _t, _ in b:
        with pytest.raises(ValueError):
            b.weights = np.array((0.5, 0.25, 0.25))

        with pytest.raises(ValueError):
            # C is not there yet
            b.cashposition = np.array((5, 5, 5))

        with pytest.raises(ValueError):
            # C is not there yet
            b.position = np.array((5, 5, 5))

    for _t, state in b:
        # set the weights for all assets alive
        b.weights = rng.uniform(size=len(state.assets))


def test_iteration_state(builder: Builder) -> None:
    """Test the initial state during Builder iteration.

    This test verifies that the initial state of the portfolio during
    iteration has the expected properties: zero leverage, correct NAV,
    zero value, and empty positions.

    Parameters
    ----------
    builder : Builder
        The Builder fixture to test

    """
    for _t, state in builder:
        assert state.leverage == 0
        assert state.nav == 1e6
        assert state.value == 0.0
        pd.testing.assert_series_equal(state.weights, pd.Series(index=state.assets, data=np.nan), check_names=False)
        pd.testing.assert_series_equal(
            builder.position,
            pd.Series(index=state.assets, data=np.nan),
            check_names=False,
        )


def test_valid(builder: Builder) -> None:
    """Test that all assets in the Builder have valid price data.

    This test verifies that the Builder correctly identifies all assets
    as having valid price data (no NaN values in the middle of time series).

    Parameters
    ----------
    builder : Builder
        The Builder fixture to test

    """
    assert np.all(builder.valid)


def test_intervals(builder: Builder) -> None:
    """Test that the Builder correctly identifies first and last valid indices.

    This test verifies that the Builder's intervals property correctly
    identifies the first and last valid indices for each asset.

    Parameters
    ----------
    builder : Builder
        The Builder fixture to test

    """
    x = builder.intervals
    assert x["last"].loc["G"] == pd.Timestamp("2015-04-22")
