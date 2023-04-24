import pandas as pd
import pytest

from cvx.simulator.portfolio import build_portfolio
from cvx.simulator.metrics import Metrics


@pytest.fixture()
def prices(resource_dir):
    return pd.read_csv(resource_dir / "price.csv", index_col=0, parse_dates=True, header=0)

@pytest.fixture()
def portfolio(prices):
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    return build_portfolio(prices, stocks=positions)

#@pytest.fixture()
#def portfolio(prices):
#    return build_portfolio(prices=prices)


def test_assets(portfolio):
    assert set(portfolio.assets) == {'A', 'B', 'C', 'D', 'E', 'F', 'G'}


def test_index(portfolio):
    assert len(portfolio.index) == 602


def test_iter(prices):
    portfolio = build_portfolio(prices)
    portfolio.stocks.loc[portfolio.index[0], "A"] = 1.0
    for before, now in portfolio:
        portfolio[now] = portfolio[before]

    pd.testing.assert_series_equal(portfolio.equity.sum(axis=1), portfolio.prices["A"], check_names=False)

    s = pd.Series(index=portfolio.index, data=0.0)
    s[s.index[0]] = 1.0

    pd.testing.assert_series_equal(portfolio.trades_stocks["A"], s, check_names=False)
    pd.testing.assert_series_equal(portfolio.trades_currency["A"], s * portfolio.prices["A"], check_names=False)


def test_set_stocks(portfolio):
    portfolio[portfolio.index[0]] = pd.Series(index=portfolio.assets, data=0.0)
    pd.testing.assert_series_equal(portfolio[portfolio.index[0]], pd.Series(index=portfolio.assets, data=0.0),
                                   check_names=False)


def test_cash(prices):
    portfolio = build_portfolio(prices=prices)
    portfolio.stocks.loc[portfolio.index[0], "A"] = 2.0
    portfolio.stocks.loc[portfolio.index[0], "B"] = 4.0

    for before, now in portfolio:
        portfolio[now] = portfolio[before]

    assert portfolio.nav(initial_cash=100000).values[-1] == pytest.approx(117665.06)
    assert portfolio.equity.sum(axis=1).values[-1] == pytest.approx(114260.54)
    assert portfolio.cash(initial_cash=100000).values[0] == pytest.approx(117665.06 - 114260.54)
    assert portfolio.cash(initial_cash=100000).values[-1] == pytest.approx(117665.06 - 114260.54)
    assert portfolio.profit.cumsum().values[-1] == pytest.approx(17665.06)
    assert portfolio.equity.sum(axis=1).diff().cumsum().values[-1] == pytest.approx(17665.06)


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

#def test_profit(portfolio):
#    print(portfolio.profit)
#    assert False

def test_profit(portfolio):
    print(portfolio.profit)
    m = Metrics(daily_profit=portfolio.profit)
    pd.testing.assert_series_equal(m.daily_profit, portfolio.profit)

    assert m.mean_profit == pytest.approx(-5.810981697171386)
    assert m.std_profit == pytest.approx(840.5615726803527)
    assert m.total_profit == pytest.approx(-3492.4000000000033)
    assert m.sr_profit == pytest.approx(-0.10974386369939439)
