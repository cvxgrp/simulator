from dataclasses import dataclass
import pandas as pd


def build_portfolio(prices, stocks=None, initial_cash=1e6):
    assert isinstance(prices, pd.DataFrame)

    if stocks is None:
        stocks = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0, dtype=float)

    assert set(stocks.index).issubset(set(prices.index))
    assert set(stocks.columns).issubset(set(prices.columns))

    prices = prices[stocks.columns].loc[stocks.index]
    return _EquityPortfolio(stocks=stocks, prices=prices, initial_cash=float(initial_cash))


@dataclass(frozen=True)
class _EquityPortfolio:
    prices: pd.DataFrame
    stocks: pd.DataFrame
    initial_cash: float = 1e6

    @property
    def index(self):
        return self.prices.index

    @property
    def assets(self):
        return self.prices.columns

    def __iter__(self):
        for before, now in zip(self.index[:-1], self.index[1:]):
            self.stocks.loc[now] = self.stocks.loc[before]
            nav = self.nav[now]
            value = self.equity.loc[now]
            cash = nav - value
            yield before, now, nav, cash
            # easy to return equity...

    def __setitem__(self, key, value):
        assert isinstance(value, pd.Series)
        self.stocks.loc[key, value.index] = value

    def __getitem__(self, item):
        assert item in self.index
        return self.stocks.loc[item]

    @property
    def equity(self):
        # same as a cash position
        return (self.prices * self.stocks).ffill()

    @property
    def trades_stocks(self):
        t = self.stocks.diff()
        t.loc[self.index[0]] = self.stocks.loc[self.index[0]]
        return t.fillna(0.0)

    @property
    def trades_currency(self):
        return self.trades_stocks * self.prices.ffill()

    @property
    def cash(self):
        return -self.trades_currency.sum(axis=1).cumsum() + self.initial_cash

    @property
    def nav(self):
        return self.equity.sum(axis=1) + self.cash

    #def returns(self, initial_cash=1):
    #    return self.profit / initial_cash

    #def accumulated(self, initial_cash=0):
    #    return self.profit.cumsum() + initial_cash

    #def compounded(self, initial_cash=1):
    #    return (1.0 + self.returns(initial_cash=initial_cash)).cumprod()

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

