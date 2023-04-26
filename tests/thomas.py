import yfinance as yf

from cvx.simulator.portfolio import build_portfolio

if __name__ == '__main__':

    data = yf.download(tickers = "SPY AAPL GOOG MSFT",  # list of tickers
                       period = "10y",                   # time period
                       interval = "1d",                 # trading interval
                       prepost = False,                 # download pre/post market hours data?
                       repair = True)                   # repair obvious price errors e.g. 100x?

    prices = data["Adj Close"]
    capital = 1e6

    portfolio = build_portfolio(prices=prices, initial_cash=capital)

    print(portfolio._state.cash)

    for before, now, snapshot in portfolio:
        # each day we invest a quarter of the capital in the assets
        portfolio[now] = 0.25 * snapshot.nav / portfolio.prices.loc[now]
