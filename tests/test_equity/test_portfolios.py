from __future__ import annotations

import pandas as pd
import pytest

from cvx.simulator.equity.builder import EquityBuilder


@pytest.fixture()
def portfolio(prices):
    """portfolio fixture"""
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    b = EquityBuilder(prices, initial_cash=1e6)

    for t, state in b:
        b.position = positions.loc[t[-1]]
        b.cash = state.cash

    return b.build()


def test_weights(portfolio):
    # in the portfolio we hold exactly one stock of each asset
    x1 = portfolio.weights.apply(lambda x: x * portfolio.nav)
    x2 = portfolio.prices.ffill()
    pd.testing.assert_frame_equal(x1, x2)


def test_portfolio_small(prices):
    builder = EquityBuilder(prices=prices[["A", "B"]], initial_cash=1e6)

    for _, state in builder:
        # hold one share in both assets
        builder.position = [1, 1]
        builder.cash = state.cash

    portfolio = builder.build()

    pd.testing.assert_series_equal(
        portfolio.nav,
        prices[["A", "B"]].sum(axis=1) + 1e6 - prices[["A", "B"]].iloc[0].sum(),
    )

    portfolio.cashflow.iloc[0] = pytest.approx(-975014.24)


def test_iter(prices):
    """
    test building a portfolio with only one asset and exactly 1 share.
    :param prices: the prices frame (fixture)
    """

    # Let's setup a portfolio with one asset: A
    b = EquityBuilder(prices[["A"]].dropna())

    # We now iterate through the underlying timestamps of the portfolio
    for times, state in b:
        # we set the position of A to 1.0
        b.position = pd.Series({"A": 1.0})
        b.cash = state.cash

    # we build the portfolio
    portfolio = b.build()

    # given our position is exactly one stock in A the price of A and the equity match
    pd.testing.assert_series_equal(
        portfolio.equity["A"], portfolio.prices["A"], check_names=False
    )

    # we only do one trade when we initialize the portfolio
    s = pd.Series(index=portfolio.index, data=0.0)
    s[s.index[0]] = 1.0

    # trades can either be measured in units or in currency units
    pd.testing.assert_series_equal(portfolio.trades_units["A"], s, check_names=False)
    pd.testing.assert_series_equal(
        portfolio.trades_currency["A"], s * portfolio.prices["A"], check_names=False
    )


def test_long_only(prices, resource_dir):
    """
    Test building a portfolio with two assets and only long positions
    :param prices: the prices frame (fixture)
    :param resource_dir: the resource directory (fixture)
    """
    # Let's setup a portfolio with two assets: A and B
    b = EquityBuilder(prices=prices[["A", "B"]], initial_cash=100000)

    # We now iterate through the underlying timestamps of the portfolio
    for times, state in b:
        # we set the position of A to 2.0 and B to 4.0
        b.position = pd.Series({"A": 2.0, "B": 4.0})
        b.cash = state.cash

    portfolio = b.build()

    # Our assets have hopefully increased in value
    # portfolio.equity.to_csv(resource_dir / "equity.csv")
    assert portfolio.equity.sum(axis=1).values[0] == pytest.approx(96595.48)
    assert portfolio.equity.sum(axis=1).values[-1] == pytest.approx(114260.54)
    pd.testing.assert_frame_equal(
        pd.read_csv(
            resource_dir / "equity.csv", index_col=0, header=0, parse_dates=True
        ),
        portfolio.equity,
    )

    # the (absolute) profit is the difference between nav and initial cash
    profit = (portfolio.nav - portfolio.cash).diff().fillna(0.0)

    # We don't need to set the initial cash to estimate the (absolute) profit
    # The daily profit is also the change in valuation of the previous position
    pd.testing.assert_series_equal(profit, portfolio.profit)

    # the investor has made approximately 17665 USD over the lifespan of the portfolio
    assert portfolio.profit.cumsum().values[-1] == pytest.approx(17665.06)

    # We assume the (retail) investor is allocating some capital C to his/her strategy.
    # Here we need enough capital to buy the initial position
    # portfolio.trades_currency.to_csv(resource_dir / "trades_usd.csv")
    pd.testing.assert_frame_equal(
        pd.read_csv(
            resource_dir / "trades_usd.csv", index_col=0, header=0, parse_dates=True
        ),
        portfolio.trades_currency,
    )

    # The NAV (net asset value) is cash + equity
    pd.testing.assert_series_equal(
        portfolio.nav, portfolio.cash + portfolio.equity.sum(axis=1)
    )


def test_portfolio(prices):
    """
    build portfolio from price
    """
    b = EquityBuilder(prices=prices)
    pd.testing.assert_frame_equal(b.prices, prices.ffill())

    for t, _ in b:
        # set the position
        b.position = pd.Series(index=prices.keys(), data=1000.0)

    portfolio = b.build()

    pd.testing.assert_frame_equal(
        portfolio.units,
        pd.DataFrame(index=prices.index, columns=prices.keys(), data=1000.0),
    )
