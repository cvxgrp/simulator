# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator, Optional, Tuple

import pandas as pd

from cvx.simulator.portfolio import EquityPortfolio
from cvx.simulator.trading_costs import TradingCostModel


@dataclass
class _State:
    """The _State class defines a state object used to keep track of the current
    state of the portfolio.

    Attributes:

    prices: a pandas Series object containing the stock prices of the current
    portfolio state

    position: a pandas Series object containing the current holdings of the portfolio

    cash: the amount of cash available in the portfolio.

    By default, prices and position are set to None, while cash is set to 1 million.
    These attributes can be updated and accessed through setter and getter methods
    """

    prices: pd.Series = None
    position: pd.Series = None
    cash: float = 1e6

    @property
    def value(self) -> float:
        """
        The value property computes the value of the portfolio at the current
        time taking into account the current holdings and current stock prices.
        If the value cannot be computed due to missing positions
        (they might be still None), zero is returned instead.
        """
        try:
            return float((self.prices * self.position).sum())
        except TypeError:
            return 0.0

    @property
    def nav(self) -> float:
        """
        The nav property computes the net asset value (NAV) of the portfolio,
        which is the sum of the current value of the
        portfolio as determined by the value property,
        and the current amount of cash available in the portfolio.
        """
        return self.value + self.cash

    @property
    def weights(self) -> pd.Series:
        """
        The weights property computes the weighting of each asset in the current
        portfolio as a fraction of the total portfolio value (nav).

        Returns:

        a pandas series object containing the weighting of each asset as a
        fraction of the total portfolio value. If the positions are still
        missing, then a series of zeroes is returned.
        """
        try:
            return (self.prices * self.position) / self.nav
        except TypeError:
            return 0 * self.prices

    @property
    def leverage(self) -> float:
        """
        The `leverage` property computes the leverage of the portfolio,
        which is the sum of the absolute values of the portfolio weights.
        """
        return float(self.weights.abs().sum())

    @property
    def position_robust(self) -> pd.Series:
        """
        The position_robust property returns the current position of the
        portfolio or a series of zeroes if the position is still missing.
        """
        if self.position is None:
            self.position = 0.0 * self.prices

        return self.position

    def update(
        self,
        position: pd.Series,
        model: Optional[TradingCostModel] = None,
        **kwargs: Any,
    ) -> _State:
        """
        The update method updates the current state of the portfolio with the
        new input position. It calculates the trades made based on the new
        and the previous position, updates the internal position and
        cash attributes, and applies any trading costs according to a model parameter.

        The method takes three input parameters:

        position: a pandas series object representing the new position of the
        portfolio.

        model: an optional trading cost model (e.g. slippage, fees) to be
        incorporated into the update. If None, no trading costs will be applied.

        **kwargs: additional keyword arguments to pass into the trading cost
        model.

        Returns:
        self: the _State instance with the updated position and cash.

        Updates:

        trades: the difference between positions in the old and new portfolio.
        position: the new position of the portfolio.
        cash: the new amount of cash in the portfolio after any trades and trading costs are applied.

        Note that the method does not return any value: instead,
        it updates the internal state of the _State instance.
        """
        trades = position - self.position_robust

        self.position = position
        self.cash -= (trades * self.prices).sum()

        if model is not None:
            self.cash -= model.eval(self.prices, trades=trades, **kwargs).sum()

        # builder is frozen, so we can't construct a new state
        return self


def builder(
    prices: pd.DataFrame,
    weights: Optional[pd.DataFrame] = None,
    market_cap: Optional[pd.DataFrame] = None,
    trade_volume: Optional[pd.DataFrame] = None,
    initial_cash: float = 1e6,
    trading_cost_model: Optional[TradingCostModel] = None,
    max_cap_fraction: Optional[float] = None,
    min_cap_fraction: Optional[float] = None,
    max_trade_fraction: Optional[float] = None,
    min_trade_fraction: Optional[float] = None,
) -> _Builder:
    """The builder function creates an instance of the _Builder class, which
    is used to construct a portfolio of assets. The function takes in a pandas
    DataFrame of historical prices for the assets in the portfolio, optional
    weights for each asset, an initial cash value, and a trading cost model.
    The function first asserts that the prices DataFrame has a monotonic
    increasing and unique index. It then creates a DataFrame of zeros to hold
    the number of shares of each asset owned at each time step. The function
    initializes a _Builder object with the stocks DataFrame, the prices
    DataFrame (forward-filled), the initial cash value, and the trading cost
    model. If weights are provided, they are set for each time step using
    set_weights method of the _Builder object. The final output is the
    constructed _Builder object."""

    assert isinstance(prices, pd.DataFrame)
    assert prices.index.is_monotonic_increasing
    assert prices.index.is_unique

    stocks = pd.DataFrame(
        index=prices.index, columns=prices.columns, data=0.0, dtype=float
    )

    builder = _Builder(
        stocks=stocks,
        prices=prices.ffill(),
        initial_cash=float(initial_cash),
        trading_cost_model=trading_cost_model,
        market_cap=market_cap,
        trade_volume=trade_volume,
        max_cap_fraction=max_cap_fraction,
        min_cap_fraction=min_cap_fraction,
        max_trade_fraction=max_trade_fraction,
        min_trade_fraction=min_trade_fraction,
    )

    if weights is not None:
        for t, state in builder:
            builder.set_weights(time=t[-1], weights=weights.loc[t[-1]])

    return builder


@dataclass(frozen=True)
class _Builder:
    prices: pd.DataFrame
    stocks: pd.DataFrame
    trading_cost_model: Optional[TradingCostModel] = None
    initial_cash: float = 1e6
    _state: _State = field(default_factory=_State)
    market_cap: pd.DataFrame = None
    trade_volume: pd.DataFrame = None
    max_cap_fraction: Optional[float] = None
    min_cap_fraction: Optional[float] = None
    max_trade_fraction: Optional[float] = None
    min_trade_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        """
        The __post_init__ method is a special method of initialized instances
        of the _Builder class and is called after initialization.
        It sets the initial amount of cash in the portfolio to be equal to the input initial_cash parameter.

        The method takes no input parameter. It initializes the cash attribute in the internal
        _State object with the initial amount of cash in the portfolio, self.initial_cash.

        Note that this method is often used in Python classes for additional initialization routines
        that can only be performed after the object is fully initialized. __post_init__
        is called automatically after the object initialization.
        """
        self._state.cash = self.initial_cash

    @property
    def index(self) -> pd.DatetimeIndex:
        """A property that returns the index of the portfolio,
        which is the time period for which the portfolio data is available.

        Returns: pd.Index: A pandas index representing the
        time period for which the portfolio data is available.

        Notes: The function extracts the index of the prices dataframe,
        which represents the time periods for which data is available for the portfolio.
        The resulting index will be a pandas index object
        with the same length as the number of rows in the prices dataframe."""

        return pd.DatetimeIndex(self.prices.index)

    @property
    def assets(self) -> pd.Index:
        """A property that returns a list of the assets held by the portfolio.

        Returns: list: A list of the assets held by the portfolio.

        Notes: The function extracts the column names of the prices dataframe,
        which correspond to the assets held by the portfolio.
        The resulting list will contain the names of all assets
        held by the portfolio, without any duplicates."""
        return self.prices.columns

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change().dropna(axis=0, how="all")

    def cov(
        self, **kwargs: Any
    ) -> Generator[Tuple[datetime, pd.DataFrame], None, None]:
        # You can do much better using volatility adjusted returns rather than returns
        cov = self.returns.ewm(**kwargs).cov()
        cov = cov.dropna(how="all", axis=0)
        for t in cov.index.get_level_values(level=0).unique():
            yield t, cov.loc[t, :, :]

        # {t: cov.loc[t, :, :] for t in cov.index.get_level_values('date').unique()}

    def set_weights(self, time: datetime, weights: pd.Series) -> None:
        """
        Set the position via weights (e.g. fractions of the nav)

        :param time: time
        :param weights: series of weights
        """
        assert isinstance(weights, pd.Series), "weights must be a pandas Series"
        self[time] = (self._state.nav * weights) / self._state.prices

    def set_cashposition(self, time: datetime, cashposition: pd.Series) -> None:
        """
        Set the position via cash positions (e.g. USD invested per asset)

        :param time: time
        :param cashposition: series of cash positions
        """
        assert isinstance(
            cashposition, pd.Series
        ), "cashposition must be a pandas Series"
        self[time] = cashposition / self._state.prices

    def set_position(self, time: datetime, position: pd.Series) -> None:
        """
        Set the position via number of assets (e.g. number of stocks)

        :param time: time
        :param position: series of number of stocks
        """
        assert isinstance(position, pd.Series), "position must be a pandas Series"
        self[time] = position

    def __iter__(self) -> Generator[Tuple[pd.DatetimeIndex, _State], None, None]:
        """
        The __iter__ method allows the object to be iterated over in a for loop,
        yielding time and the current state of the portfolio.
        The method yields a list of dates seen so far
        (excluding the first date) and returns a tuple
        containing the list of dates and the current portfolio state.

        Yield:

        interval: a pandas DatetimeIndex object containing the dates seen so far.

        state: the current state of the portfolio,
        taking into account the stock prices at each interval.
        """
        for t in self.index:
            # valuation of the current position
            self._state.prices = self.prices.loc[t]
            yield self.index[self.index <= t], self._state

    def __setitem__(self, time: datetime, position: pd.Series) -> None:
        """
        The method __setitem__ updates the stock data in the dataframe for a specific time index
        with the input position. It first checks that position is a valid input,
        meaning it is a pandas Series object and has its index within the assets of the dataframe.
        The method takes two input parameters:

        time: time index for which to update the stock data
        position: pandas series object containing the updated stock data

        Returns: None

        Updates:
        the stock data of the dataframe at the given time index with the input position
        the internal state of the portfolio with the updated position, taking into account the trading cost model

        Raises:
        AssertionError: if the input position is not a pandas Series object
        or its index is not a subset of the assets of the dataframe.
        """
        assert isinstance(position, pd.Series)
        assert set(position.index).issubset(set(self.assets))

        if self.market_cap is not None:
            # compute capitalization of desired position
            cap = position * self._state.prices
            # compute relative capitalization
            rel_cap = cap / self.market_cap.loc[time]
            # clip relative capitalization
            rel_cap.clip(
                lower=self.min_cap_fraction, upper=self.max_cap_fraction, inplace=True
            )
            # move back to capitalization
            cap = rel_cap * self.market_cap.loc[time]
            # compute position
            position = cap / self._state.prices

        if self.trade_volume is not None:
            trade = position - self._state.position_robust

            # move to trade in USD
            trade = trade * self._state.prices
            # compute relative trade volume
            rel_trade = trade / self.trade_volume.loc[time]
            # clip relative trade volume
            rel_trade.clip(
                lower=self.min_trade_fraction,
                upper=self.max_trade_fraction,
                inplace=True,
            )
            # move back to trade
            trade = rel_trade * self.trade_volume.loc[time]
            # move back to trade in number of stocks
            trade = trade / self._state.prices
            # compute position
            position = self._state.position_robust + trade

        self.stocks.loc[time, position.index] = position
        self._state.update(position, model=self.trading_cost_model)

    def __getitem__(self, time: datetime) -> pd.Series:
        """The __getitem__ method retrieves the stock data for a specific time in the dataframe.
        It returns the stock data for that time. The method takes one input parameter:

        time: the time index for which to retrieve the stock data
        Returns: stock data for the input time

        Note that the input time must be in the index of the dataframe, otherwise a KeyError will be raised.
        """
        return self.stocks.loc[time]

    def build(self) -> EquityPortfolio:
        """A function that creates a new instance of the EquityPortfolio
        class based on the internal state of the Portfolio builder object.

        Returns: EquityPortfolio: A new instance of the EquityPortfolio class
        with the attributes (prices, stocks, initial_cash, trading_cost_model) as specified in the Portfolio builder.

        Notes: The function simply creates a new instance of the EquityPortfolio
        class with the attributes (prices, stocks, initial_cash, trading_cost_model) equal
        to the corresponding attributes in the Portfolio builder object.
        The resulting EquityPortfolio object will have the same state as the Portfolio builder from which it was built.
        """

        return EquityPortfolio(
            prices=self.prices,
            stocks=self.stocks,
            initial_cash=self.initial_cash,
            trading_cost_model=self.trading_cost_model,
        )
