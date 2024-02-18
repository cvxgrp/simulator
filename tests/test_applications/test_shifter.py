# from cvx.simulator import Builder
import pandas as pd

from cvx.simulator import Builder
from cvx.simulator.shifter import shift


def test_portfolio_cumulated(prices):
    builder = Builder(prices=prices[["A", "B"]].tail(10), initial_aum=1e6)

    for t, state in builder:
        # hold one share in both assets
        builder.cashposition = [1e5, 4e5]

        # reduce the available aum by the costs
        builder.aum = state.aum  # - costs

    portfolio = builder.build()
    print(portfolio.weights)
    print(portfolio.aum)

    ppp = shift(portfolio=portfolio, periods=2)
    print(ppp.weights)
    print(ppp.aum)

    pd.testing.assert_frame_equal(portfolio.weights.shift(2), ppp.weights)
