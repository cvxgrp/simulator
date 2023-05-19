from dataclasses import dataclass, field
import pandas as pd

from cvx.simulator.portfolio import EquityPortfolio
from cvx.simulator.trading_costs import TradingCostModel


@dataclass
class _State:
    """The _State class defines a state object used to keep track of the current state of the portfolio.
    Attributes:

    prices: a pandas Series object containing the stock prices of the current portfolio state
    position: a pandas Series object containing the current holdings of the portfolio
    cash: the amount of cash available in the portfolio.

    By default, prices and position are set to None, while cash is set to 1 million.
    These attributes can be updated and accessed through setter and getter methods
    """
    prices: pd.Series = None
    position: pd.Series = None
    cash: float = 1e6

    @property
    def value(self):
        """
        The value method computes the value of the portfolio at the current time,
        taking into account the current holdings and current stock prices.


        Returns:

        the value of the portfolio as a float. If the value cannot be computed due to missing position
        (they might be still None), zero is returned instead.

        The method is a decorator-based getter method, using the @property decorator to access
        the value attribute as if it were a method. The value of the portfolio is computed as the
        product of the current stock prices and the current holdings, summed together.
        Note that if the current position is missing, the multiplication will raise a TypeError.
        In this case, zero is returned as the value of the portfolio.
        """
        # Note that the first position may not exist yet.
        try:
            return (self.prices * self.position).sum()
        except TypeError:
            return 0.0

    @property
    def nav(self):
        """
        The nav method computes the net asset value (NAV) of the portfolio,
        which is the sum of the current value of the
        portfolio as determined by the value method,
        and the current amount of cash available in the portfolio.

        Returns:

        The net asset value of the portfolio as a float, computed as the sum of the current
        value of the portfolio and the cash available in the portfolio.
        The method is a decorator-based getter method using the @property decorator
        to access the NAV attribute as if it were a method.
        The NAV is computed as the sum of the current value (as computed by the value method)
        and the amount of cash available.
        """
        return self.value + self.cash

    @property
    def weights(self):
        """
        The weights method computes the weighting of each asset in the current portfolio
        as a fraction of the total portfolio value.

        Returns:

        a pandas series object containing the weighting of each asset as a fraction
        of the total portfolio value. If the positions are still missing, then a series of zeroes is returned.
        The method is a decorator-based getter method using the @property decorator to access
        the weights attribute as if it were a method. The weighting for each asset is computed
        as the product of the current position and the current stock prices,
        divided by the net asset value of the portfolio (as computed by the nav method).
        Note that this weighting represents the fraction of the portfolio value allocated to a particular asset.
        If the portfolio weighting cannot be computed due to missing positions,
        then a series of zeroes with the same size as the prices attribute is returned instead.
        """
        try:
            return (self.prices * self.position)/self.nav
        except TypeError:
            return 0 * self.prices

    @property
    def leverage(self):
        """
        The `leverage` method computes the leverage of the portfolio,
        which is the sum of the absolute values of the portfolio weights.

        Returns:
        - the portfolio leverage as a float, computed as the sum of the
        absolute values of the portfolio weights.

        The method is a decorator-based getter method using the
        `@property` decorator to access the leverage attribute
        as if it were a method. The portfolio leverage is computed as
        the sum of the absolute values of the portfolio weights,
        which represents the total exposure of the portfolio to multiple assets.
        """
        return self.weights.abs().sum()

    def update(self, position, model=None, **kwargs):
        """
        The update method updates the current state of the portfolio with the new input position.
        It calculates the trades made based on the new and the previous position,
        updates the internal position and cash attributes,
        and applies any applicable trading costs according to model parameter.

        The method takes three input parameters:

        position: a pandas series object representing the new position of the portfolio.
        model: an optional trading cost model (e.g. slippage, fees) to be incorporated into the update.
        If None, no trading costs will be applied.
        **kwargs: additional keyword arguments to pass into the trading cost model.

        Returns:
        self: the _State instance with the updated position and cash.

        Updates:

        trades: the difference between positions in the old and new portfolio.
        position: the new position of the portfolio.
        cash: the new amount of cash in the portfolio after any trades and trading costs are applied.

        Note that the method does not return any value: instead,
        it updates the internal state of the _State instance.
        """
        if self.position is None:
            trades = position
        else:
            trades = position - self.position

        self.position = position
        self.cash -= (trades * self.prices).sum()

        if model is not None:
            self.cash -= model.eval(self.prices,  trades=trades, **kwargs).sum()

        return self


def builder(prices, weights=None, initial_cash=1e6, trading_cost_model=None):
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

    stocks = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0, dtype=float)

    trading_cost_model = trading_cost_model
    builder = _Builder(stocks=stocks, prices=prices.ffill(), initial_cash=float(initial_cash),
                    trading_cost_model=trading_cost_model)

    if weights is not None:
        for t, state in builder:
            builder.set_weights(time=t[-1], weights=weights.loc[t[-1]])

    return builder


@dataclass(frozen=True)
class _Builder:
    prices: pd.DataFrame
    stocks: pd.DataFrame
    trading_cost_model: TradingCostModel
    initial_cash: float = 1e6
    _state: _State = field(default_factory=_State)

    def __post_init__(self):
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
    def index(self):
        """ A property that returns the index of the portfolio,
        which is the time period for which the portfolio data is available.

        Returns: pd.Index: A pandas index representing the
        time period for which the portfolio data is available.

        Notes: The function extracts the index of the prices dataframe,
        which represents the time periods for which data is available for the portfolio.
        The resulting index will be a pandas index object
        with the same length as the number of rows in the prices dataframe. """

        return self.prices.index

    @property
    def assets(self):
        """ A property that returns a list of the assets held by the portfolio.

        Returns: list: A list of the assets held by the portfolio.

        Notes: The function extracts the column names of the prices dataframe,
        which correspond to the assets held by the portfolio.
        The resulting list will contain the names of all assets
        held by the portfolio, without any duplicates. """
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

    def __setitem__(self, time, position):
        """
        The method __setitem__ updates the stock data in the dataframe for a specific time index with the input position. It first checks that position is a valid input, meaning it is a pandas Series object and has its index within the assets of the dataframe.
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

        self.stocks.loc[time, position.index] = position
        self._state.update(position, model=self.trading_cost_model)

    def __getitem__(self, time):
        """ The __getitem__ method retrieves the stock data for a specific time in the dataframe.
        It returns the stock data for that time. The method takes one input parameter:

        time: the time index for which to retrieve the stock data
        Returns: stock data for the input time

        Note that the input time must be in the index of the dataframe, otherwise a KeyError will be raised.
        """
        return self.stocks.loc[time]

    def build(self):
        """ A function that creates a new instance of the EquityPortfolio
        class based on the internal state of the Portfolio builder object.

        Returns: EquityPortfolio: A new instance of the EquityPortfolio class
        with the attributes (prices, stocks, initial_cash, trading_cost_model) as specified in the Portfolio builder.

        Notes: The function simply creates a new instance of the EquityPortfolio
        class with the attributes (prices, stocks, initial_cash, trading_cost_model) equal
        to the corresponding attributes in the Portfolio builder object.
        The resulting EquityPortfolio object will have the same state as the Portfolio builder from which it was built. """

        return EquityPortfolio(prices=self.prices,
                               stocks=self.stocks,
                               initial_cash=self.initial_cash,
                               trading_cost_model=self.trading_cost_model)

