import pandas as pd
import numpy as np

import pytest

from cvx.simulator.builder import _State
from cvx.simulator.builder import builder
from cvx.simulator.portfolio import EquityPortfolio


def test_state():
    prices = pd.Series(data=[2.0, 3.0])
    positions = pd.Series(data=[100, 300])
    cash = 400
    state = _State(cash=cash, prices=prices, position=positions)
    assert state.value == 1100.0
    assert state.nav == 1500.0
    pd.testing.assert_series_equal(state.weights, pd.Series(data=[2.0/15.0, 9.0/15.0]))
    assert state.leverage == 11.0/15.0


def test_assets(portfolio):
    assert set(portfolio.assets) == {'A', 'B', 'C', 'D', 'E', 'F', 'G'}


def test_index(portfolio):
    assert len(portfolio.index) == 602
    pd.testing.assert_index_equal(portfolio.index, portfolio.prices.index)


def test_prices(portfolio, prices):
    pd.testing.assert_frame_equal(portfolio.prices, prices)


def test_stocks(portfolio):
    stocks = pd.DataFrame(index=portfolio.index, columns=portfolio.assets, data=1.0)
    pd.testing.assert_frame_equal(portfolio.stocks, stocks)


def test_iter(prices):
    # construct a portfolio with only one asset
    b = builder(prices[["A"]])
    assert set(b.assets) == {"A"}

    # don't change the position at all and loop through the entire history
    for times, _ in b:
        b[times[-1]] = pd.Series({"A": 1.0})

    portfolio = b.build()

    # given our position is exactly one stock in A the price of A and the equity match
    pd.testing.assert_series_equal(portfolio.equity["A"], portfolio.prices["A"], check_names=False)

    # we only do one trade when we initialize the portfolio
    s = pd.Series(index=portfolio.index, data=0.0)
    s[s.index[0]] = 1.0

    # trades can either be measured in stocks or in currency units
    pd.testing.assert_series_equal(portfolio.trades_stocks["A"], s, check_names=False)
    pd.testing.assert_series_equal(portfolio.trades_currency["A"], s * portfolio.prices["A"], check_names=False)


def test_long_only(prices, resource_dir):
    # Let's setup a portfolio with two assets: A and B
    b = builder(prices=prices[["A", "B"]], initial_cash=100000)
    assert set(b.assets) == {"A", "B"}
    assert len(b.index) == 602

    pd.testing.assert_index_equal(b.index, b.prices.index)

    # We now iterate through the underlying timestamps of the portfolio
    for times, _ in b:
        b[times[-1]] = pd.Series({"A": 2.0, "B": 4.0})

    portfolio = b.build()
    # Our assets have hopefully increased in value
    # portfolio.equity.to_csv(resource_dir / "equity.csv")
    assert portfolio.equity.sum(axis=1).values[0] == pytest.approx(96595.48)
    assert portfolio.equity.sum(axis=1).values[-1] == pytest.approx(114260.54)
    pd.testing.assert_frame_equal(pd.read_csv(resource_dir / "equity.csv", index_col=0, header=0, parse_dates=True),
                                  portfolio.equity)

    # the (absolute) profit is the difference between nav and initial cash
    profit = (portfolio.nav - portfolio.initial_cash).diff().dropna()
    
    # We don't need to set the initial cash to estimate the (absolute) profit
    # The daily profit is also the change in valuation of the previous position
    pd.testing.assert_series_equal(profit, portfolio.profit)

    # the investor has made approximately 17665 USD over the lifespan of the portfolio
    assert portfolio.profit.cumsum().values[-1] == pytest.approx(17665.06)

    # We assume the (retail) investor is allocating some capital C to his/her strategy.
    # Here we need enough capital to buy the initial position
    #portfolio.trades_currency.to_csv(resource_dir / "trades_usd.csv")
    pd.testing.assert_frame_equal(pd.read_csv(resource_dir / "trades_usd.csv", index_col=0, header=0, parse_dates=True),
                                  portfolio.trades_currency)

    # The available cash is the initial cash - costs for trading, e.g.
    pd.testing.assert_series_equal(portfolio.cash, portfolio.initial_cash -portfolio.trades_currency.sum(axis=1).cumsum())

    # The NAV (net asset value) is cash + equity
    pd.testing.assert_series_equal(portfolio.nav, portfolio.cash + portfolio.equity.sum(axis=1))


def test_long_short(prices, resource_dir):
    # Let's setup a portfolio with two assets: B and C
    b = builder(prices=prices[["B", "C"]], initial_cash=20000)
    assert set(b.assets) == {"B", "C"}
    assert len(b.index) == 602

    pd.testing.assert_index_equal(b.index, b.prices.index)

    # We now iterate through the underlying timestamps of the portfolio
    for times, _ in b:
        b[times[-1]] = pd.Series({"B": 3.0, "C": -1.0})

    portfolio = b.build()

    # Our assets have hopefully increased in value
    #portfolio.equity.to_csv(resource_dir / "equity_ls.csv")
    assert portfolio.equity.sum(axis=1).values[0] == pytest.approx(7385.84)
    assert portfolio.equity.sum(axis=1).values[-1] == pytest.approx(30133.25)
    pd.testing.assert_frame_equal(pd.read_csv(resource_dir / "equity_ls.csv", index_col=0, header=0, parse_dates=True),
                                  portfolio.equity)


    # the (absolute) profit is the difference between nav and initial cash
    profit = (portfolio.nav - portfolio.initial_cash).diff().dropna()

    # We don't need to set the initial cash to estimate the (absolute) profit
    # The daily profit is also the change in valuation of the previous position
    pd.testing.assert_series_equal(profit, portfolio.profit)

    # the investor has made approximately 17665 USD over the lifespan of the portfolio
    assert portfolio.profit.cumsum().values[-1] == pytest.approx(22747.41)

    #portfolio.trades_currency.to_csv(resource_dir / "trades_usd_ls.csv")
    pd.testing.assert_frame_equal(pd.read_csv(resource_dir / "trades_usd_ls.csv", index_col=0, header=0, parse_dates=True),
                                  portfolio.trades_currency)

    # The available cash is the initial cash - costs for trading, e.g.
    pd.testing.assert_series_equal(portfolio.cash, portfolio.initial_cash - portfolio.trades_currency.sum(axis=1).cumsum())

    # The NAV (net asset value) is cash + equity
    pd.testing.assert_series_equal(portfolio.nav, portfolio.cash + portfolio.equity.sum(axis=1))


def test_add(prices, resource_dir):
    """
    Tests the addition of two portfolios
    TODP: Currently only tests the positions of the portfolios
    """
    index_left = pd.DatetimeIndex([pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-02")])
    index_right = pd.DatetimeIndex([pd.Timestamp("2013-01-02"), pd.Timestamp("2013-01-03"), pd.Timestamp("2013-01-04")])

    pos_left = pd.DataFrame(data={"A": [0, 1], "C": [3, 3]}, index=index_left)
    pos_right = pd.DataFrame(data={"A": [1, 1, 2], "B": [2, 3, 4]}, index=index_right)

    port_left = EquityPortfolio(prices.loc[pos_left.index][pos_left.columns], stocks=pos_left)
    port_right = EquityPortfolio(prices.loc[pos_right.index][pos_right.columns], stocks=pos_right)

    pd.testing.assert_frame_equal(pos_left, port_left.stocks)
    pd.testing.assert_frame_equal(pos_right, port_right.stocks)

    port_add = port_left + port_right
    www = pd.read_csv(resource_dir / "positions.csv", index_col=0, parse_dates=[0])
    pd.testing.assert_frame_equal(www, port_add.stocks, check_freq=False)






def test_duplicates():
    """
    duplicate in index
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
    index not increasing
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

    for t,_ in b:
        # set the position
        b[t[-1]] = pd.Series(index=prices.keys(), data=1000.0)

    portfolio = b.build()

    pd.testing.assert_frame_equal(
        portfolio.stocks,
        pd.DataFrame(index=prices.index, columns=prices.keys(), data=1000.0),
    )


def test_multiply(portfolio):
    double = portfolio*2.0
    pd.testing.assert_frame_equal(2.0*portfolio.stocks, double.stocks)


def test_multiply_r(portfolio):
    double = 2.0*portfolio
    pd.testing.assert_frame_equal(2.0*portfolio.stocks, double.stocks)


def test_truncate(portfolio):
    p = portfolio.truncate(before=portfolio.index[100])
    assert set(p.index) == set(portfolio.index[100:])
    assert p.initial_cash == portfolio.nav.values[100]
    assert p.nav.values[-1] == portfolio.nav.values[-1]


def test_resample(prices):
    b = builder(prices=prices)
    pd.testing.assert_frame_equal(b.prices, prices.ffill())

    for time, state in b:
        # each day we do a one-over-N rebalancing
        b[time[-1]] = 1.0 / len(b.assets) * state.nav / state.prices

    portfolio = b.build()

    # only now we reample the portfolio
    p = portfolio.resample(rule="M", truncate=False)


    # check the last few rows
    p = p.truncate(before=p.index[590])
    assert np.linalg.norm(p.trades_stocks.iloc[1:].values) == pytest.approx(0.0, abs=1e-12)
