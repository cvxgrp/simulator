import numpy as np
import pandas as pd
import pytest

from cvx.simulator import EquityBuilder, EquityPortfolio, interpolate

# from cvx.simulator.interpolation import interpolate


@pytest.fixture()
def builder(prices):
    """
    Fixture for the builder
    :param prices: the prices frame (fixture)
    """
    return EquityBuilder(prices=prices)


def test_initial_cash(builder):
    """
    Test that the initial cash is 1e6
    :param builder: the builder object (fixture)
    """
    assert builder.initial_cash == 1e6


def test_build_empty(builder, prices):
    """
    Test that the portfolio is still empty
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    portfolio = builder.build()
    pd.testing.assert_frame_equal(portfolio.prices, prices)
    pd.testing.assert_frame_equal(portfolio.units, np.NaN * prices)
    pd.testing.assert_series_equal(
        portfolio.profit, pd.Series(index=prices.index, data=0.0)
    )


def test_set_position(prices):
    b = EquityBuilder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.position = state.nav / (state.prices * 2)
        assert np.allclose(b.position, state.nav / (state.prices * 2))

    portfolio = b.build()
    assert isinstance(portfolio, EquityPortfolio)

    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_set_weights(prices):
    """
    Test that the weights are set correctly
    :param prices: the prices frame (fixture)
    """
    b = EquityBuilder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.weights = np.array([0.5, 0.5])
        assert np.allclose(b.weights, np.array([0.5, 0.5]))
        assert np.allclose(state.weights.values, np.array([0.5, 0.5]))

    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_set_cashpositions(prices):
    """
    Test that the cashpositions are set correctly
    :param prices: the prices frame (fixture)
    """
    b = EquityBuilder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.cashposition = np.ones(2) * state.nav / 2
        assert np.allclose(b.cashposition, np.ones(2) * state.nav / 2)

    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_set_position_again(prices):
    b = EquityBuilder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.position = state.nav / (state.prices * 2)
        assert np.allclose(b.position, state.nav / (state.prices * 2))

    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_weights_on_wrong_days(resource_dir):
    prices = pd.read_csv(
        resource_dir / "priceNaN.csv", index_col=0, parse_dates=True, header=0
    ).apply(interpolate)

    # there are no inner NaNs

    b = EquityBuilder(prices=prices, initial_cash=50000)
    t = prices.index

    for t, state in b:
        with pytest.raises(ValueError):
            b.weights = np.array([0.5, 0.25, 0.25])

        with pytest.raises(ValueError):
            # C is not there yet
            b.cashposition = [5, 5, 5]

        with pytest.raises(ValueError):
            # C is not there yet
            b.position = [5, 5, 5]

    for t, state in b:
        # set the weights for all assets alive
        b.weights = np.random.rand(len(state.assets))


def test_iteration_state(builder):
    """
    iterate through builder and verify that the state is correct
    :param builder: the builder object (fixture)
    """
    for t, state in builder:
        assert state.leverage == 0
        assert state.nav == 1e6
        assert state.value == 0.0
        pd.testing.assert_series_equal(
            state.weights, pd.Series(index=state.assets, data=np.NaN), check_names=False
        )
        pd.testing.assert_series_equal(
            builder.position,
            pd.Series(index=state.assets, data=np.NaN),
            check_names=False,
        )