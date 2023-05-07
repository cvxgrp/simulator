from dataclasses import dataclass, field
import pandas as pd

from cvx.simulator.trading_costs import LinearCostModel, TradingCostModel


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


def build_portfolio(prices, stocks=None, initial_cash=1e6, trading_cost_model=None):
    assert isinstance(prices, pd.DataFrame)

    if stocks is None:
        stocks = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0, dtype=float)

    assert set(stocks.index).issubset(set(prices.index))
    assert set(stocks.columns).issubset(set(prices.columns))

    prices = prices[stocks.columns].loc[stocks.index]

    trading_cost_model = trading_cost_model or LinearCostModel(name="LinearCostModel cost model")
    return _EquityPortfolio(stocks=stocks, prices=prices.ffill(), initial_cash=float(initial_cash),
                            trading_cost_model=trading_cost_model)


@dataclass(frozen=True)
class _EquityPortfolio:
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
        for before, now in zip(self.index[:-1], self.index[1:]):
            # valuation of the current position
            self._state.prices = self.prices.loc[now]

            yield before, now, self._state

    def __setitem__(self, key, position):
        assert isinstance(position, pd.Series)
        assert set(position.index).issubset(set(self.assets))

        self.stocks.loc[key, position.index] = position
        self._state.update(position, model=self.trading_cost_model)

    def __getitem__(self, item):
        assert item in self.index
        return self.stocks.loc[item]

    @property
    def trading_costs(self) -> pd.DataFrame:
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

        return build_portfolio(prices=prices_right, stocks=positions,
                               initial_cash=self.initial_cash + port_new.initial_cash)
