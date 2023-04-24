from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_portfolio(prices, stocks=None):
    assert isinstance(prices, pd.DataFrame)

    if stocks is None:
        stocks = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    assert set(stocks.index).issubset(set(prices.index))
    assert set(stocks.columns).issubset(set(prices.columns))

    prices = prices[stocks.columns].loc[stocks.index]
    return _EquityPortfolio(stocks=stocks, prices=prices)


@dataclass(frozen=True)
class _EquityPortfolio:
    prices: pd.DataFrame
    stocks: pd.DataFrame

    @property
    def index(self):
        return self.prices.index

    @property
    def assets(self):
        return self.prices.columns

    def __iter__(self):
        for before, now in zip(self.index[:-1], self.index[1:]):
            yield before, now

    def __setitem__(self, key, value):
        assert isinstance(value, pd.Series)
        self.stocks.loc[key, value.index] = value

    def __getitem__(self, item):
        assert item in self.index
        return self.stocks.loc[item]

    @property
    def equity(self):
        return (self.prices * self.stocks).ffill()

    @property
    def trades_stocks(self):
        t = self.stocks.diff()
        t.loc[self.index[0]] = self.stocks.loc[self.index[0]]
        return t.fillna(0.0)

    @property
    def trades_currency(self):
        return self.trades_stocks * self.prices.ffill()

    def cash(self, initial_cash):
        return -self.trades_currency.sum(axis=1).cumsum() + initial_cash

    def nav(self, initial_cash):
        return self.equity.sum(axis=1) + self.cash(initial_cash)

    @property
    def profit(self):
        """
        Profit
        """
        price_changes = self.prices.ffill().diff()
        previous_stocks = self.stocks.shift(1).fillna(0.0)
        return (previous_stocks * price_changes).dropna(axis=0, how="all").sum(axis=1)

    def __mul__(self, scalar):
        """
        Multiplies positions by a scalar
        """
        return _EquityPortfolio(prices=self.prices, stocks=self.stocks * scalar)

    def __add__(self, port_new):
        """
        Adds two portfolios together
        """
        assert isinstance(port_new, _EquityPortfolio)

        assets = self.assets.union(port_new.assets)
        index = self.index.union(port_new.index)

        left = pd.DataFrame(index=index, columns=assets)
        left.update(self.stocks)
        # this is a problem...
        left = left.fillna(0.0)

        right = pd.DataFrame(index=index, columns=assets)
        right.update(port_new.stocks)
        right = right.fillna(0.0)

        positions = left + right

        prices_left = self.prices.combine_first(port_new.prices)
        prices_right = port_new.prices.combine_first(self.prices)

        pd.testing.assert_frame_equal(prices_left, prices_right)

        return build_portfolio(prices=prices_right, stocks=positions)


if __name__ == '__main__':
    index = pd.date_range('2021-01-01', periods=8, freq='D')

    prices=pd.DataFrame(columns=["A","B"], index=index, data=np.random.rand(8,2))
    portfolio = build_portfolio(prices)

    # set the initial position outside the loop
    portfolio.stocks.loc[index[0]] = pd.Series(index=["A","B"], data=[3.0, 4.0])

    for before, now in portfolio:
        portfolio[now] = portfolio[before]
        portfolio[now] = pd.Series(index=["A"], data=[5.0])

    print(portfolio.stocks)
    print(portfolio.equity.sum(axis=1))
    print(portfolio.trades_currency)
    print(portfolio.trades_stocks)
    print(portfolio.cash(initial_cash=10))
