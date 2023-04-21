# That’s great!
# The simulator should always be completely agnostic as to the trading policy.
# You should even demonstrate this with silly policies.
# Like here’s one:  Each day choose names from the universe at random.
# Buy one (say 0.1 of your portfolio wealth) and short one the same amount.
# Not a good strategy, but a valid one.
# Of course the simulate will terminate if you go bust (which seems likely).
from pathlib import Path

import numpy as np
import pandas as pd

from cvx.simulator.EquityPortfolio import build_portfolio

if __name__ == '__main__':
    prices = pd.read_csv(Path("resources") / "price.csv", index_col=0, parse_dates=True, header=0).ffill()
    portfolio = build_portfolio(prices=prices)

    # we assume a constant portfolio size throughout the backtest
    initial_cash = 1e6

    for before, now in portfolio:
        # pick two assets at random
        pair = np.random.choice(portfolio.assets, 2)
        # prices for the pair
        prices = portfolio.prices.loc[now][pair]
        # compute the pair
        stocks = pd.Series(index=portfolio.assets, data=0.0)
        stocks[pair] = [1e6, -1e6]/prices.values

        portfolio[now] = stocks

    print(portfolio.stocks)
    print(portfolio.trades_currency.sum(axis=1))
    print(portfolio.nav(initial_cash=initial_cash))


