# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from cvx.simulator.builder import _State, builder
from cvx.simulator.portfolio import EquityPortfolio, Plot, diff


def test_state():
    prices = pd.Series(data=[2.0, 3.0])
    positions = pd.Series(data=[100, 300])
    cash = 400
    state = _State(cash=cash, prices=prices, position=positions)
    # value is the money in stocks
    assert state.value == 1100.0
    # nav is the value plus the cash
    assert state.nav == 1500.0
    # weights are the positions divided by the value
    pd.testing.assert_series_equal(
        state.weights, pd.Series(data=[2.0 / 15.0, 9.0 / 15.0])
    )
    # leverage is the value divided by the nav
    assert state.leverage == 11.0 / 15.0


def test_assets(portfolio, prices):
    """
    Test that the assets of the portfolio are the same as the columns of the prices
    :param portfolio: the portfolio object (fixture)
    :param prices: the prices frame (fixture)
    """
    assert set(portfolio.assets) == set(prices.columns)


def test_index(portfolio):
    """
    Test that the index of the portfolio is the same as the index of the prices
    :param portfolio: the portfolio object (fixture)
    """
    assert len(portfolio.index) == 602
    pd.testing.assert_index_equal(portfolio.index, portfolio.prices.index)


def test_prices(portfolio, prices):
    """
    Test that the prices of the portfolio are the same as the prices
    :param portfolio: the portfolio object (fixture)
    :param prices: the prices frame (fixture)
    """
    pd.testing.assert_frame_equal(portfolio.prices, prices)


def test_turnover(portfolio):
    v = portfolio.trades_stocks * portfolio.prices.ffill()
    pd.testing.assert_frame_equal(v, portfolio.trades_currency)
    pd.testing.assert_frame_equal(v.abs(), portfolio.turnover)


def test_get(portfolio):
    for time in portfolio.index:
        w = portfolio[time]
        assert isinstance(w, pd.Series)


def test_stocks(portfolio):
    """
    Test that the stocks of the portfolio have all been set to 1.0
    :param portfolio: the portfolio object (fixture)
    """
    stocks = pd.DataFrame(index=portfolio.index, columns=portfolio.assets, data=1.0)
    pd.testing.assert_frame_equal(portfolio.stocks, stocks)


def test_weights(portfolio):
    # in the portfolio we hold exactly one stock of each asset
    x1 = portfolio.weights.apply(lambda x: x * portfolio.nav)
    x2 = portfolio.prices.ffill()
    pd.testing.assert_frame_equal(x1, x2)


def test_drawdown(portfolio):
    """
    Test that the drawdown of the portfolio is zero
    :param portfolio: the portfolio object (fixture)
    """
    pd.testing.assert_series_equal(
        portfolio.highwater, portfolio.nav.expanding(min_periods=1).max()
    )

    drawdown = 1.0 - portfolio.nav / portfolio.highwater
    pd.testing.assert_series_equal(portfolio.drawdown, drawdown)


def test_iter(prices):
    """
    test building a portfolio with only one asset and exactly 1 share.
    :param prices: the prices frame (fixture)
    """

    # Let's setup a portfolio with one asset: A
    b = builder(prices[["A"]])

    # We now iterate through the underlying timestamps of the portfolio
    for times, _ in b:
        # we set the position of A to 1.0
        b[times[-1]] = pd.Series({"A": 1.0})

    # we build the portfolio
    portfolio = b.build()

    # given our position is exactly one stock in A the price of A and the equity match
    pd.testing.assert_series_equal(
        portfolio.equity["A"], portfolio.prices["A"], check_names=False
    )

    # we only do one trade when we initialize the portfolio
    s = pd.Series(index=portfolio.index, data=0.0)
    s[s.index[0]] = 1.0

    # trades can either be measured in stocks or in currency units
    pd.testing.assert_series_equal(portfolio.trades_stocks["A"], s, check_names=False)
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
    b = builder(prices=prices[["A", "B"]], initial_cash=100000)

    # We now iterate through the underlying timestamps of the portfolio
    for times, _ in b:
        # we set the position of A to 2.0 and B to 4.0
        b[times[-1]] = pd.Series({"A": 2.0, "B": 4.0})

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
    profit = (portfolio.nav - portfolio.initial_cash).diff().dropna()

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

    # The available cash is the initial cash - costs for trading, e.g.
    pd.testing.assert_series_equal(
        portfolio.cash,
        portfolio.initial_cash - portfolio.trades_currency.sum(axis=1).cumsum(),
    )

    # The NAV (net asset value) is cash + equity
    pd.testing.assert_series_equal(
        portfolio.nav, portfolio.cash + portfolio.equity.sum(axis=1)
    )


def test_long_short(prices, resource_dir):
    """
    Test building a portfolio with two assets and long and short positions
    :param prices: the prices frame (fixture)
    :param resource_dir: the resource directory (fixture)
    """
    # Let's setup a portfolio with two assets: A and B
    b = builder(prices=prices[["B", "C"]], initial_cash=20000)

    # We now iterate through the underlying timestamps of the portfolio
    for times, _ in b:
        # we set the position of B to 3.0 and C to -1.0
        b[times[-1]] = pd.Series({"B": 3.0, "C": -1.0})

    # we build the portfolio
    portfolio = b.build()

    # Our assets have hopefully increased in value
    assert portfolio.equity.sum(axis=1).values[0] == pytest.approx(7385.84)
    assert portfolio.equity.sum(axis=1).values[-1] == pytest.approx(30133.25)
    pd.testing.assert_frame_equal(
        pd.read_csv(
            resource_dir / "equity_ls.csv", index_col=0, header=0, parse_dates=True
        ),
        portfolio.equity,
    )

    # the (absolute) profit is the difference between nav and initial cash
    profit = (portfolio.nav - portfolio.initial_cash).diff().dropna()

    # We don't need to set the initial cash to estimate the (absolute) profit
    # The daily profit is also the change in valuation of the previous position
    pd.testing.assert_series_equal(profit, portfolio.profit)

    # the investor has made approximately 17665 USD over the lifespan of the portfolio
    assert portfolio.profit.cumsum().values[-1] == pytest.approx(22747.41)

    # portfolio.trades_currency.to_csv(resource_dir / "trades_usd_ls.csv")
    pd.testing.assert_frame_equal(
        pd.read_csv(
            resource_dir / "trades_usd_ls.csv", index_col=0, header=0, parse_dates=True
        ),
        portfolio.trades_currency,
    )

    # The available cash is the initial cash - costs for trading, e.g.
    pd.testing.assert_series_equal(
        portfolio.cash,
        portfolio.initial_cash - portfolio.trades_currency.sum(axis=1).cumsum(),
    )

    # The NAV (net asset value) is cash + equity
    pd.testing.assert_series_equal(
        portfolio.nav, portfolio.cash + portfolio.equity.sum(axis=1)
    )


def test_add(prices, resource_dir):
    """
    Tests the addition of two portfolios
    TODP: Currently only tests the positions of the portfolios
    """
    index_left = pd.DatetimeIndex(
        [pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")]
    )
    index_right = pd.DatetimeIndex(
        [
            pd.Timestamp("2013-01-02"),
            pd.Timestamp("2013-01-03"),
            pd.Timestamp("2013-01-04"),
        ]
    )

    pos_left = pd.DataFrame(data={"A": [0, 1], "C": [3, 3]}, index=index_left)
    pos_right = pd.DataFrame(data={"A": [1, 1, 2], "B": [2, 3, 4]}, index=index_right)

    port_left = EquityPortfolio(
        prices.loc[pos_left.index][pos_left.columns], stocks=pos_left
    )
    port_right = EquityPortfolio(
        prices.loc[pos_right.index][pos_right.columns], stocks=pos_right
    )

    pd.testing.assert_frame_equal(pos_left, port_left.stocks)
    pd.testing.assert_frame_equal(pos_right, port_right.stocks)

    port_add = port_left + port_right
    print(port_add.stocks)
    print(port_add.prices)

    www = pd.read_csv(resource_dir / "positions.csv", index_col=0, parse_dates=[0])
    pd.testing.assert_frame_equal(www, port_add.stocks, check_freq=False)


def test_duplicates():
    """
    test for duplicates in the index
    """
    prices = pd.DataFrame(index=[1, 1], columns=["A"])
    with pytest.raises(AssertionError):
        builder(prices=prices)

    prices = pd.DataFrame(index=[1], columns=["A"])
    position = pd.DataFrame(index=[1, 1], columns=["A"])

    with pytest.raises(AssertionError):
        EquityPortfolio(prices=prices, stocks=position)


def test_monotonic():
    """
    test for monotonic index
    """
    prices = pd.DataFrame(index=[2, 1], columns=["A"])
    with pytest.raises(AssertionError):
        builder(prices=prices)


def test_portfolio(prices):
    """
    build portfolio from price
    """
    b = builder(prices=prices)
    pd.testing.assert_frame_equal(b.prices, prices.ffill())

    for t, _ in b:
        # set the position
        b[t[-1]] = pd.Series(index=prices.keys(), data=1000.0)

    portfolio = b.build()

    pd.testing.assert_frame_equal(
        portfolio.stocks,
        pd.DataFrame(index=prices.index, columns=prices.keys(), data=1000.0),
    )


def test_multiply(portfolio):
    """
    Test multiplication of portfolio
    :param portfolio: the portfolio object (fixture)
    """
    double = portfolio * 2.0
    pd.testing.assert_frame_equal(2.0 * portfolio.stocks, double.stocks)


def test_multiply_r(portfolio):
    """
    Test multiplication of portfolio
    :param portfolio: the portfolio object (fixture)
    """
    double = 2.0 * portfolio
    pd.testing.assert_frame_equal(2.0 * portfolio.stocks, double.stocks)


def test_truncate(portfolio):
    """
    Test truncation of portfolio
    :param portfolio: the portfolio object (fixture)
    """
    p = portfolio.truncate(before=portfolio.index[100])
    assert set(p.index) == set(portfolio.index[100:])
    assert p.initial_cash == portfolio.nav.values[100]
    assert p.nav.values[-1] == portfolio.nav.values[-1]


def test_diff(portfolio):
    """
    Test diff of portfolio
    :param portfolio: the portfolio object (fixture)
    """
    p = diff(portfolio, portfolio)
    assert p.initial_cash == 1e6
    assert p.trading_cost_model is None

    pd.testing.assert_frame_equal(p.stocks, 0.0 * portfolio.stocks)


def test_resample(prices):
    """
    Test resampling of portfolio
    :param prices:
    :return:
    """
    b = builder(prices=prices)
    # pd.testing.assert_frame_equal(b.prices, prices.ffill())

    for time, state in b:
        # each day we do a one-over-N rebalancing
        b[time[-1]] = 1.0 / len(b.assets) * state.nav / state.prices

    portfolio = b.build()

    # only now we resample the portfolio
    p = portfolio.resample(rule="M")

    # check the last few rows
    p = p.truncate(before=p.index[590])
    assert np.linalg.norm(p.trades_stocks.iloc[1:].values) == pytest.approx(
        0.0, abs=1e-12
    )


def test_quantstats(portfolio):
    """
    Test quantstats
    :param portfolio: the portfolio object (fixture)
    """

    portfolio.nav.sharpe()
    portfolio.metrics(mode="full")


def test_plots(portfolio):
    portfolio.plots(mode="full", show=False)


def test_plot(portfolio):
    portfolio.plot(kind=Plot.DRAWDOWN, show=False)
    portfolio.plot(kind=Plot.MONTHLY_HEATMAP, show=False)


def test_html(portfolio, tmp_path):
    portfolio.html(output=tmp_path / "test.html")
    assert os.path.exists(tmp_path / "test.html")


def test_snapshot(portfolio):
    portfolio.snapshot(show=False, fontname=None)


def test_plot_enum(portfolio):
    for plot in Plot:
        print("********************************************************************")
        print(plot)
        try:
            plot.plot(portfolio.nav.pct_change().dropna(), show=False, fontname=None)
        except Exception as e:
            print(e)
            pass


def test_rolling_betas(portfolio):
    portfolio.plot(
        kind=Plot.ROLLING_BETA,
        benchmark=0.5 * portfolio.nav.pct_change().dropna(),
        fontname=None,
        show=False,
    )
