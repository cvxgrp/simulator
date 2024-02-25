import pandas as pd
import pytest

from cvx.simulator.builder import Builder
from cvx.simulator.portfolio import Portfolio


def test_portfolio_cumulated(prices):
    builder = Builder(prices=prices[["A", "B"]].tail(10), initial_aum=1e6)

    for t, state in builder:
        # hold one share in both assets
        builder.cashposition = [1e5, 4e5]

        # costs are 5 bps of the traded value
        costs = 0.0005 * (state.trades.abs() * state.prices).sum()

        # reduce the available aum by the costs
        builder.aum = state.aum - costs

    portfolio = builder.build()

    pd.testing.assert_series_equal(portfolio.nav, builder.aum)

    # assert sharpe(portfolio.nav.pct_change().dropna()) == pytest.approx(3.891806571531769, abs=1e-3)
    assert portfolio.nav.iloc[-1] == pytest.approx(1015576.0104632963, abs=1e-3)


def test_from_cash_position_prices(prices):
    cashpos = pd.DataFrame(index=prices.index, columns=prices.columns, data=1e5)

    portfolio = Portfolio.from_cashpos_prices(
        prices=prices, cashposition=cashpos, aum=1e6
    )

    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(
        portfolio.nav, profit.cumsum() + portfolio.aum, check_names=False
    )


def test_from_cash_returns(prices):
    returns = prices.pct_change().fillna(0.0)
    cashpos = pd.DataFrame(index=prices.index, columns=prices.columns, data=1e5)

    portfolio = Portfolio.from_cashpos_returns(
        returns=returns, cashposition=cashpos, aum=1e6
    )

    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(
        portfolio.nav, profit.cumsum() + portfolio.aum, check_names=False
    )

    print(portfolio.weights)
