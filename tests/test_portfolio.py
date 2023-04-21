import pandas as pd
import pytest

from cvx.simulator.EquityPortfolio import build_portfolio



@pytest.fixture()
def prices(resource_dir):
    return pd.read_csv(resource_dir / "price.csv", index_col=0, parse_dates=True, header=0)


@pytest.fixture()
def portfolio(prices):
    return build_portfolio(prices=prices)


def test_assets(portfolio):
    assert set(portfolio.assets) == {'A', 'B', 'C', 'D', 'E', 'F', 'G'}


def test_index(portfolio):
    assert len(portfolio.index) == 602


def test_iter(portfolio):
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


def test_cash(portfolio):
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


