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
        """
        The timestamps in the portfolio (index in prices frame)

        Returns:
             Index of timestamps
        """
        return self.prices.index

    @property
    def assets(self):
        """
        The assets in the portfolio (columns in prices frame)

        Returns:
            Index of assets.
        """
        return self.prices.columns

    @property
    def weights(self):
        """
        Frame of relative weights (e.g. value / nav)
        """
        return self.equity / self.nav

    def __getitem__(self, time):
        return self.stocks.loc[time]

    @property
    def trading_costs(self):
        # return a frame of all zeros
        if self.trading_cost_model is None:
            return 0.0 * self.prices

        return self.trading_cost_model.eval(self.prices, self.trades_stocks)

    @property
    def equity(self) -> pd.DataFrame:
        """ A property that returns a pandas dataframe
        representing the equity positions of the portfolio,
        which is the value of each asset held by the portfolio.
        Returns: pd.DataFrame: A pandas dataframe representing
        the equity positions of the portfolio.

        Notes: The function calculates the equity of the portfolio
        by multiplying the current prices of each asset
        by the number of shares held by the portfolio.
        The resulting values are filled forward to account
        for any missing data or NaN values.
        The equity dataframe will have the same dimensions
        as the prices and stocks dataframes. """

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
        """ Returns a pandas series representing the total value
        of the portfolio's investments and cash.

        Returns: pd.Series: A pandas series representing the
                            total value of the portfolio's investments and cash.
        """

        return self.equity.sum(axis=1) + self.cash

    @property
    def profit(self) -> pd.Series:
        """ A property that returns a pandas series representing the
        profit gained or lost in the portfolio based on changes in asset prices.

        Returns: pd.Series: A pandas series representing the profit
        gained or lost in the portfolio based on changes in asset prices.

        Notes: The calculation is based on the difference between
        the previous and current prices of the assets in the portfolio,
        multiplied by the number of stocks in each asset previously held.
        """

        price_changes = self.prices.ffill().diff()
        previous_stocks = self.stocks.shift(1).fillna(0.0)
        return (previous_stocks * price_changes).dropna(axis=0, how="all").sum(axis=1)

    @property
    def highwater(self) -> pd.Series:
        """ A function that returns a pandas series representing
        the high-water mark of the portfolio, which is the highest point
        the portfolio value has reached over time.

        Returns: pd.Series: A pandas series representing the
        high-water mark of the portfolio.

        Notes: The function performs a rolling computation based on
        the cumulative maximum of the portfolio's value over time,
        starting from the beginning of the time period being considered.
        Min_periods argument is set to 1 to include the minimum period of one day.
        The resulting series will show the highest value the portfolio has reached at each point in time. """
        return self.nav.expanding(min_periods=1).max()

    @property
    def drawdown(self) -> pd.Series:
        """ A property that returns a pandas series representing the
        drawdown of the portfolio, which measures the decline
        in the portfolio's value from its (previously) highest
        point to its current point.

        Returns: pd.Series: A pandas series representing the
        drawdown of the portfolio.

        Notes: The function calculates the ratio of the portfolio's current value
        vs. its current high-water-mark and then subtracting the result from 1.
        A positive drawdown means the portfolio is currently worth
        less than its high-water mark. A drawdown of 0.1 implies that the nav is currently 0.9 times the high-water mark """
        return 1.0 - self.nav / self.highwater

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
