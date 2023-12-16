import numpy as np
import pandas as pd
import pytest

from cvx.simulator.equity.builder import EquityBuilder


@pytest.fixture()
def index():
    return pd.DatetimeIndex(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"])


@pytest.fixture()
def columns():
    return ["A", "B"]


@pytest.fixture()
def prices(index, columns):
    return pd.DataFrame(
        index=index,
        columns=columns,
        data=([np.nan, 200], [100, 200], [100, 200], [100, np.nan]),
    )


def f(n):
    return np.ones(n) / n


def test_case(columns, index, prices):
    b = EquityBuilder(prices=prices, initial_cash=2000)
    print(f"\n{b.prices.values}")

    assert np.all(b.valid)

    frame = pd.DataFrame(
        index=columns,
        columns=["first", "last"],
        data=([[index[1], index[3]], [index[0], index[2]]]),
    )
    pd.testing.assert_frame_equal(frame, b.intervals)

    # run a 1/n portfolio on those prices
    for t, state in b:
        b.weights = f(n=len(state.assets))

    # build the portfolio
    portfolio = b.build()

    # 1) first buy 10 of B  2) buy 10 of A and sell 5 of B  3) do nothing 4) your units in B are now worthless
    stocks = pd.DataFrame(
        index=index,
        columns=columns,
        data=([np.nan, 10.0], [10.0, 5.0], [10.0, 5.0], [10.0, np.nan]),
    )
    pd.testing.assert_frame_equal(stocks, b.units)

    # equity is the value of your units,
    # 1) 2000 in B 2) 1000 in A and 1000 in B 3) 1000 in A and 1000 in B 4) 1000 in A, B worthless
    equity = pd.DataFrame(
        index=index,
        columns=columns,
        data=([np.nan, 2000.0], [1000.0, 1000.0], [1000.0, 1000.0], [1000.0, np.nan]),
    )
    pd.testing.assert_frame_equal(equity, portfolio.equity)

    # trades are the currency value of the trades,
    # 1) buy 2000 of B 2) buy 1000 of A and sell 1000 of B 3) do nothing 4) position in B is worthless
    trades = pd.DataFrame(
        index=index,
        columns=columns,
        data=([np.nan, 2000.0], [1000.0, -1000.0], [0.0, 0.0], [0.0, np.nan]),
    )
    pd.testing.assert_frame_equal(trades, portfolio.trades_currency)

    # weights are the portfolio weights, 1) 100% B 2) 50% A, 50% B 3) 50% A, 50% B 4) 100% A because B is worthless
    weights = pd.DataFrame(
        index=index,
        columns=columns,
        data=([np.nan, 1.0], [0.5, 0.5], [0.5, 0.5], [1.0, np.nan]),
    )
    pd.testing.assert_frame_equal(weights, portfolio.weights)

    # cash is the cash balance, all initial cash is in stock at all times
    pd.testing.assert_series_equal(
        portfolio.cash, pd.Series(index=index, data=[0.0, 0.0, 0.0, 0.0])
    )
    # nav is the net asset value equity and cash, 1) 2000 2) 2000 3) 2000 4) 1000
    pd.testing.assert_series_equal(
        portfolio.nav, pd.Series(index=index, data=[2000.0, 2000.0, 2000.0, 1000.0])
    )
