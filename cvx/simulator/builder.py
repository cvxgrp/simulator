from dataclasses import dataclass, field
import pandas as pd

from cvx.simulator.portfolio import EquityPortfolio
from cvx.simulator.trading_costs import TradingCostModel


@dataclass
class _State:
    prices: pd.Series = None
    position: pd.Series = None
    cash: float = 1e6

    @property
    def value(self):
        return (self.prices * self.position).sum()

    @property
    def nav(self):
        return self.value + self.cash

    @property
    def weights(self):
        return (self.prices * self.position)/self.nav

    @property
    def leverage(self):
        return self.weights.abs().sum()

    def update(self, position, model=None, **kwargs):
        trades = position - self.position
        self.position = position
        self.cash -= (trades * self.prices).sum()

        if model is not None:
            self.cash -= model.eval(self.prices,  trades=trades, **kwargs).sum()

        return self


def builder(prices, initial_cash=1e6, trading_cost_model=None):
    assert isinstance(prices, pd.DataFrame)
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique

    stocks = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0, dtype=float)

    trading_cost_model = trading_cost_model
    return _Builder(stocks=stocks, prices=prices.ffill(), initial_cash=float(initial_cash),
                    trading_cost_model=trading_cost_model)


@dataclass(frozen=True)
class _Builder:
    prices: pd.DataFrame
    stocks: pd.DataFrame
    trading_cost_model: TradingCostModel
    initial_cash: float = 1e6
    _state: _State = field(default_factory=_State)

    def __post_init__(self):
        self._state.position = self.stocks.loc[self.index[0]]
        self._state.prices = self.prices.loc[self.index[0]]
        self._state.cash = self.initial_cash - self._state.value

    @property
    def index(self):
        return self.prices.index

    @property
    def assets(self):
        return self.prices.columns

    def set_weights(self, time, weights):
        """
        Set the position via weights (e.g. fractions of the nav)

        :param time: time
        :param weights: series of weights
        """
        self[time] = (self._state.nav * weights) / self._state.prices

    def set_cashposition(self, time, cashposition):
        """
        Set the position via cash positions (e.g. USD invested per asset)

        :param time: time
        :param cashposition: series of cash positions
        """
        self[time] = cashposition / self._state.prices

    def set_position(self, time, position):
        """
        Set the position via number of assets (e.g. number of stocks)

        :param time: time
        :param position: series of number of stocks
        """
        self[time] = position

    def __iter__(self):
        for t in self.index[1:]:
            # valuation of the current position
            self._state.prices = self.prices.loc[t]

            # this is probably very slow...
            # portfolio = EquityPortfolio(prices=self.prices.truncate(after=now), stocks=self.stocks.truncate(after=now), initial_cash=self.initial_cash, trading_cost_model=self.trading_cost_model)

            yield self.index[self.index <= t], self._state

    def __setitem__(self, key, position):
        assert isinstance(position, pd.Series)
        assert set(position.index).issubset(set(self.assets))

        self.stocks.loc[key, position.index] = position
        self._state.update(position, model=self.trading_cost_model)

    def __getitem__(self, item):
        assert item in self.index
        return self.stocks.loc[item]

    def build(self):
        return EquityPortfolio(prices=self.prices, stocks=self.stocks, initial_cash=self.initial_cash, trading_cost_model=self.trading_cost_model)

