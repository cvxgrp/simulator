from dataclasses import dataclass

import numpy as np
import pandas as pd


def build(prices, capital):
    stocks = pd.DataFrame(index=prices.index, columns=prices.columns, data=np.NaN)
    return _EquityPortfolio(prices=prices, capital=capital, stocks=stocks)


@dataclass(frozen=True)
class _EquityPortfolio:
    prices: pd.DataFrame
    capital: float
    stocks: pd.DataFrame

    @property
    def index(self):
        return self.prices.index

    @property
    def assets(self):
        return self.prices.columns

    def __iter__(self):
        for before, now in zip(self.index[:-1], self.index[1:]):
            self.stocks.loc[now] = self.stocks.loc[before]
            yield before, now

    def __setitem__(self, key, value):
        assert isinstance(value, pd.Series)
        self.stocks.loc[key, value.index] = value

    def __getitem__(self, item):
        assert item in self.index
        return self.stocks.loc[item]

    @property
    def value(self):
        return self.prices * self.stocks

    @property
    def trades_stocks(self):
        t = self.stocks.diff()
        t.loc[self.index[0]] = self.stocks.loc[self.index[0]]
        return t

    @property
    def trades_currency(self):
        return self.trades_stocks * self.prices



if __name__ == '__main__':
    index = pd.date_range('2021-01-01', periods=8, freq='D')

    prices=pd.DataFrame(columns=["A","B"], index=index, data=np.random.rand(8,2))
    portfolio = build(prices, capital=1e6)

    # set the initial position outside the loop
    portfolio.stocks.loc[index[0]] = pd.Series(index=["A","B"], data=[3.0, 4.0])

    for before, now in portfolio:
        portfolio[now] = pd.Series(index=["A"], data=[5.0])

    print(portfolio.stocks)
    print(portfolio.value.sum(axis=1))
    print(portfolio.trades_currency)
    print(portfolio.trades_stocks)