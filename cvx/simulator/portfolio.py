from dataclasses import dataclass
import pandas as pd

from cvx.simulator.trading_costs import TradingCostModel


@dataclass(frozen=True)
class EquityPortfolio:
    prices: pd.DataFrame
    stocks: pd.DataFrame
    trading_cost_model: TradingCostModel = None
    initial_cash: float = 1e6

    def __post_init__(self):
        assert self.prices.index.is_monotonic_increasing
        assert self.prices.index.is_unique
        assert self.stocks.index.is_monotonic_increasing
        assert self.stocks.index.is_unique

        assert set(self.stocks.index).issubset(set(self.prices.index))
        assert set(self.stocks.columns).issubset(set(self.prices.columns))

    @property
    def index(self):
        return self.prices.index

    @property
    def assets(self):
        return self.prices.columns

    @property
    def weights(self):
        return self.equity / self.nav

    def __getitem__(self, item):
        assert item in self.index
        return self.stocks.loc[item]

    @property
    def trading_costs(self):
        # return a frame of all zeros
        if self.trading_cost_model is None:
            return 0.0 * self.prices

        return self.trading_cost_model.eval(self.prices, self.trades_stocks)

    @property
    def equity(self) -> pd.DataFrame:
        # same as a cash position
        return (self.prices * self.stocks).ffill()

    @property
    def trades_stocks(self) -> pd.DataFrame:
        t = self.stocks.diff()
        t.loc[self.index[0]] = self.stocks.loc[self.index[0]]
        return t.fillna(0.0)

    @property
    def trades_currency(self) -> pd.DataFrame:
        return self.trades_stocks * self.prices.ffill()

    @property
    def cash(self) -> pd.Series:
        return self.initial_cash - self.trades_currency.sum(axis=1).cumsum() - self.trading_costs.sum(axis=1).cumsum()

    @property
    def nav(self) -> pd.Series:
        return self.equity.sum(axis=1) + self.cash

    @property
    def profit(self) -> pd.Series:
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
        return EquityPortfolio(prices=self.prices, stocks=self.stocks * scalar, initial_cash=self.initial_cash * scalar, trading_cost_model=self.trading_cost_model)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, port_new):
        """
        Adds two portfolios together
        """
        assert isinstance(port_new, EquityPortfolio)

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

        return EquityPortfolio(prices=prices_right, stocks=positions,
                               initial_cash=self.initial_cash + port_new.initial_cash,
                               trading_cost_model=self.trading_cost_model)
