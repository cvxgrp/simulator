import pandas as pd
import pytest

from cvx.simulator.portfolio import build_portfolio, _State


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
    pd.testing.assert_frame_equal(portfolio.prices, prices.ffill())


def test_stocks(portfolio):
    stocks = pd.DataFrame(index=portfolio.index, columns=portfolio.assets, data=1.0)
    pd.testing.assert_frame_equal(portfolio.stocks, stocks)


def test_iter(prices):
    # construct a portfolio with only one asset
    portfolio = build_portfolio(prices[["A"]])
    assert set(portfolio.assets) == {"A"}

    # initialize the position of the asset
    portfolio.stocks["A"].loc[portfolio.index[0]] = 1.0

    # don't change the position at all and loop through the entire history
    for before, now, _ in portfolio:
        portfolio[now] = portfolio[before]

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
    portfolio = build_portfolio(prices=prices[["A", "B"]], initial_cash=100000)
    assert set(portfolio.assets) == {"A", "B"}
    assert len(portfolio.index) == 602

    pd.testing.assert_index_equal(portfolio.index, portfolio.prices.index)

    # We initialize the position (but not the cash) with 2 stocks in A and 4 stocks in B
    portfolio.stocks.loc[portfolio.index[0], "A"] = 2.0
    portfolio.stocks.loc[portfolio.index[0], "B"] = 4.0

    # We now iterate through the underlying timestamps of the portfolio
    for before, now, _ in portfolio:
        # before is t_{i-1} and now is t_{i}
        portfolio[now] = portfolio[before]

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
    portfolio = build_portfolio(prices=prices[["B", "C"]], initial_cash=20000)
    assert set(portfolio.assets) == {"B", "C"}
    assert len(portfolio.index) == 602

    pd.testing.assert_index_equal(portfolio.index, portfolio.prices.index)

    # We initialize the position (but not the cash) with 2 stocks in A and 4 stocks in B
    portfolio.stocks.loc[portfolio.index[0], "B"] =  3.0
    portfolio.stocks.loc[portfolio.index[0], "C"] = -1.0

    # We now iterate through the underlying timestamps of the portfolio
    for before, now, _ in portfolio:
        # before is t_{i-1} and now is t_{i}
        portfolio[now] = portfolio[before]

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

    port_left = build_portfolio(prices, stocks=pos_left)
    port_right = build_portfolio(prices, stocks=pos_right)

    pd.testing.assert_frame_equal(pos_left, port_left.stocks)
    pd.testing.assert_frame_equal(pos_right, port_right.stocks)

    port_add = port_left + port_right
    www = pd.read_csv(resource_dir / "positions.csv", index_col=0, parse_dates=[0])
    pd.testing.assert_frame_equal(www, port_add.stocks, check_freq=False)


def test_head(prices):
    portfolio = build_portfolio(prices=prices[["B", "C"]].head(2), initial_cash=20000)

    for before, now, state in portfolio:
        # before is t_{i-1} and now is t_{i}
        assert before == portfolio.index[0]
        assert now == portfolio.index[1]
        assert state.nav == 20000.0
        assert state.cash == 20000.0
        assert state.value == 0.0


def test_set_weights(prices):
    portfolio = build_portfolio(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for before, now, state in portfolio:
        portfolio.set_weights(time=now, weights=pd.Series(index=["B","C"], data=0.5))

    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_set_cashpositions(prices):
    portfolio = build_portfolio(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for before, now, state in portfolio:
        portfolio.set_cashposition(time=now, cashposition=pd.Series(index=["B", "C"], data=state.nav / 2))

    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)

        #portfolio[now] = portfolio[before]

def test_duplicates():
    """
    duplicate in index
    """
    prices = pd.DataFrame(index=[1, 1], columns=["A"])
    with pytest.raises(AssertionError):
        build_portfolio(prices=prices)

    prices = pd.DataFrame(index=[1], columns=["A"])
    position = pd.DataFrame(index=[1, 1], columns=["A"])

    with pytest.raises(AssertionError):
        build_portfolio(prices=prices, stocks=position)


def test_monotonic():
    """
    index not increasing
    """
    prices = pd.DataFrame(index=[2, 1], columns=["A"])
    with pytest.raises(AssertionError):
        build_portfolio(prices=prices)

def test_portfolio(prices):
    """
    build portfolio from price
    """
    portfolio = build_portfolio(prices=prices)
    pd.testing.assert_frame_equal(portfolio.prices, prices.ffill())

    for t in portfolio.index:
        # set the position
        portfolio[t] = pd.Series(index=prices.keys(), data=1000.0)

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
