import pandas as pd
import pytest

from cvx.simulator.EquityPortfolio import build


@pytest.fixture()
def prices(resource_dir):
    return pd.read_csv(resource_dir / "price.csv", index_col=0, parse_dates=True, header=0)


@pytest.fixture()
def portfolio(prices):
    return build(prices=prices)


def test_assets(portfolio):
    assert set(portfolio.assets) == {'A', 'B', 'C', 'D', 'E', 'F', 'G'}


def test_index(portfolio):
    assert len(portfolio.index) == 602


def test_iter(portfolio):
    portfolio.stocks.loc[portfolio.index[0], "A"] = 1.0

    for before, now in portfolio:
        portfolio.stocks.loc[now] = portfolio.stocks.loc[before]

    pd.testing.assert_series_equal(portfolio.equity.sum(axis=1), portfolio.prices["A"], check_names=False)

    s = pd.Series(index=portfolio.index, data=0.0)
    s[s.index[0]] = 1.0

    pd.testing.assert_series_equal(portfolio.trades_stocks["A"], s, check_names=False)
    pd.testing.assert_series_equal(portfolio.trades_currency["A"], s*portfolio.prices["A"], check_names=False)


def test_set_stocks(portfolio):
    portfolio[portfolio.index[0]] = pd.Series(index = portfolio.assets, data=0.0)
    pd.testing.assert_series_equal(portfolio[portfolio.index[0]], pd.Series(index = portfolio.assets, data=0.0), check_names=False)


def test_cash(portfolio):
    portfolio.stocks.loc[portfolio.index[0], "A"] = 2.0
    portfolio.stocks.loc[portfolio.index[0], "B"] = 4.0

    for before, now in portfolio:
        portfolio.stocks.loc[now] = portfolio.stocks.loc[before]

    assert portfolio.nav(100000).values[-1] == pytest.approx(117665.06)
