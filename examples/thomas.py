import pandas as pd
pd.options.plotting.backend = "plotly"

import yfinance as yf
import quantstats as qs

from cvx.simulator.portfolio import build_portfolio
from cvx.simulator.metrics import Metrics

if __name__ == '__main__':

    data = yf.download(tickers = "SPY AAPL GOOG MSFT",  # list of tickers
                       period = "10y",                  # time period
                       interval = "1d",                 # trading interval
                       prepost = False,                 # download pre/post market hours data?
                       repair = True)                   # repair obvious price errors e.g. 100x?

    prices = data["Adj Close"]
    capital = 1e6

    portfolio = build_portfolio(prices=prices, initial_cash=capital)

    for _, now, state in portfolio:
        # each day we invest a quarter of the capital in the assets
        portfolio[now] = 0.125 * state.nav / state.prices

    #fig = portfolio.cash.plot()
    #fig.show()

    qs.plots.snapshot(portfolio.nav, title="1/n portfolio")
    print(Metrics(portfolio.profit).sr_profit)

    # show sharpe ratio
    print(qs.stats.sharpe(portfolio.nav.pct_change()))
    print(qs.stats.sharpe(portfolio.profit))
    print(portfolio.nav.min())
    print(portfolio.nav.max())
    print(qs.stats.sharpe(portfolio.nav))

    print(Metrics(portfolio.nav.pct_change().dropna()).sr_profit)
    print(qs.stats.sharpe(portfolio.nav.pct_change().dropna()))
    print(qs.stats.sharpe(portfolio.nav))