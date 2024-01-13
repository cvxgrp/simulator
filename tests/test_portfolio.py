import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from cvx.simulator.portfolio import Portfolio
from cvx.simulator.utils.quantstats.plot import Plot


@pytest.fixture()
def nav():
    return pd.read_csv(
        Path(__file__).parent / "resources" / "nav.csv",
        index_col=0,
        parse_dates=True,
        header=0,
    ).squeeze()


@pytest.fixture()
def portfolio(prices, nav):
    """portfolio fixture"""

    units = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)

    return Portfolio(prices=prices, units=units, aum=nav)


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
    v = portfolio.trades_units * portfolio.prices
    pd.testing.assert_frame_equal(v, portfolio.trades_currency)
    pd.testing.assert_frame_equal(v.abs(), portfolio.turnover)


def test_turnover_relative(portfolio):
    v = portfolio.trades_units * portfolio.prices
    pd.testing.assert_frame_equal(v.div(portfolio.nav, 0), portfolio.turnover_relative)


def test_get(portfolio):
    for time in portfolio.index:
        w = portfolio[time]
        assert isinstance(w, pd.Series)


def test_units(portfolio):
    """
    Test that the units of the portfolio have all been set to 1.0
    :param portfolio: the portfolio object (fixture)
    """
    stocks = pd.DataFrame(index=portfolio.index, columns=portfolio.assets, data=1.0)
    pd.testing.assert_frame_equal(portfolio.units, stocks)


def test_returns(portfolio):
    """
    Test that the returns of the portfolio are the same as the returns of the prices
    :param portfolio: the portfolio object (fixture)
    """
    pd.testing.assert_frame_equal(portfolio.returns, portfolio.prices.pct_change())


def test_cashposition(portfolio):
    """
    Test that the cashposition of the portfolio is the same as the cashposition of the prices
    :param portfolio: the portfolio object (fixture)
    """
    pd.testing.assert_frame_equal(
        portfolio.cashposition, portfolio.prices * portfolio.units
    )


def test_drawdown(portfolio):
    """
    Test that the drawdown of the portfolio is zero
    :param portfolio: the portfolio object (fixture)
    """
    pd.testing.assert_series_equal(
        portfolio.highwater,
        portfolio.nav.expanding(min_periods=1).max(),
        check_names=False,
    )

    drawdown = 1.0 - portfolio.nav / portfolio.highwater
    pd.testing.assert_series_equal(portfolio.drawdown, drawdown, check_names=False)


def test_monotonic():
    """
    test for monotonic index
    """
    prices = pd.DataFrame(index=[2, 1], columns=["A"])
    with pytest.raises(AssertionError):
        Portfolio(prices=prices, units=prices, aum=1e6)


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
    # snapshot returns a matplotlib figure
    # You need to call show() to display it
    xxx = pd.Series(index=portfolio.index, data=0.0)
    portfolio.snapshot(benchmark=xxx)
    plt.show()
    # or
    # portfolio.snapshot(benchmark=xxx).show()


def test_plot_enum(portfolio):
    for plot in Plot:
        print("********************************************************************")
        print(plot)
        try:
            plot.plot(portfolio.nav.pct_change().dropna(), show=False, fontname=None)
        except Exception as e:
            print(e)
            pass


def test_equity(portfolio):
    """
    Test that the equity of the portfolio is the same as the prices * units
    :param portfolio: the portfolio object (fixture)
    """
    pd.testing.assert_frame_equal(portfolio.equity, portfolio.prices * portfolio.units)


def test_profit(portfolio):
    """
    Test that the profit is computed correctly
    :param portfolio: the portfolio object (fixture)
    """
    pd.testing.assert_series_equal(
        portfolio.profit,
        portfolio.equity.sum(axis=1).diff().fillna(0.0),
        check_names=False,
    )


def test_rolling_betas(portfolio):
    portfolio.plot(
        kind=Plot.ROLLING_BETA,
        benchmark=0.5 * portfolio.nav.pct_change().dropna(),
        fontname=None,
        show=False,
    )


def test_profit_metrics(portfolio):
    """
    Test that the profit is computed correctly
    :param portfolio: the portfolio object (fixture)
    """
    assert portfolio.profit.mean() == pytest.approx(-5.801328903654639)
    assert portfolio.profit.std() == pytest.approx(839.8620124674756)
    assert portfolio.profit.sum() == pytest.approx(-3492.4000000000033)
    assert portfolio.profit.sharpe() == pytest.approx(-0.10965282385614909)
