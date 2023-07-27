# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Tuple

import pandas as pd
import quantstats as qs

from cvx.simulator.grid import iron_frame
from cvx.simulator.trading_costs import TradingCostModel

qs.extend_pandas()


class Plot(Enum):
    DAILY_RETURNS = 1
    DISTRIBUTION = 2
    DRAWDOWN = 3
    DRAWDOWNS_PERIODS = 4
    EARNINGS = 5
    HISTOGRAM = 6
    LOG_RETURNS = 7
    MONTHLY_HEATMAP = 8
    # see issue: https://github.com/ranaroussi/quantstats/issues/276
    MONTHLY_RETURNS = 9
    RETURNS = 10
    ROLLING_BETA = 11
    ROLLING_SHARPE = 12
    ROLLING_SORTINO = 13
    ROLLING_VOLATILITY = 14
    YEARLY_RETURNS = 15

    #
    # def __call__(self, returns, **kwargs):
    #     return self._func(returns=returns, **kwargs)

    def plot(self, returns: pd.DataFrame, **kwargs: Any) -> Any:
        func = getattr(qs.plots, self.name.lower())
        return func(returns=returns, **kwargs)


def diff(
    portfolio1: EquityPortfolio,
    portfolio2: EquityPortfolio,
    initial_cash: float = 1e6,
    trading_cost_model: Optional[TradingCostModel] = None,
) -> EquityPortfolio:
    # check both portfolios are on the same price grid
    pd.testing.assert_frame_equal(portfolio1.prices, portfolio2.prices)

    stocks = portfolio1.stocks - portfolio2.stocks

    return EquityPortfolio(
        prices=portfolio1.prices,
        stocks=stocks,
        initial_cash=initial_cash,
        trading_cost_model=trading_cost_model,
    )


@dataclass(frozen=True)
class EquityPortfolio:
    """A class that represents an equity portfolio
    and contains dataframes for prices and stock holdings,
    as well as optional parameters for trading cost models
    and initial cash values.

    Attributes:
        prices (pd.DataFrame): A pandas dataframe representing
        the prices of various assets held by the portfolio over time.
        stocks (pd.DataFrame): A pandas dataframe representing the number of shares
        held for each asset in the portfolio over time.
        trading_cost_model (TradingCostModel): An optional trading cost model
        to use when trading assets in the portfolio.
        initial_cash (float): An optional scalar float representing the initial
        cash value available for the portfolio.

    Notes: The EquityPortfolio class is designed to represent
    a portfolio of assets where only equity positions are held.
    The prices and stocks dataframes are assumed to have the same
    index object representing the available time periods for which data is available.
    If no trading cost model is provided, the trading_cost_model attribute
    will be set to None by default.
    If no initial cash value is provided, the initial_cash attribute
    will be set to a default value of 1,000,000."""

    prices: pd.DataFrame
    stocks: pd.DataFrame
    trading_cost_model: Optional[TradingCostModel] = None
    initial_cash: float = 1e6

    def __post_init__(self) -> None:
        """A class method that performs input validation after object initialization.
        Notes: The post_init method is called after an instance of the EquityPortfolio class has been initialized,
        and performs a series of input validation checks to ensure that the prices
        and stocks dataframes are in the expected format
        with no duplicates or missing data,
        and that the stocks dataframe represents valid equity positions
        for the assets held in the portfolio.
        Specifically, the method checks that both the prices and stocks dataframes
        have a monotonic increasing and unique index,
        and that the index and columns of the stocks dataframe are subsets
        of the index and columns of the prices dataframe, respectively.
        If any of these checks fail, an assertion error will be raised."""

        assert self.prices.index.is_monotonic_increasing
        assert self.prices.index.is_unique
        assert self.stocks.index.is_monotonic_increasing
        assert self.stocks.index.is_unique

        assert set(self.stocks.index).issubset(set(self.prices.index))
        assert set(self.stocks.columns).issubset(set(self.prices.columns))

    @property
    def index(self) -> pd.DatetimeIndex:
        """A property that returns the index of the EquityPortfolio instance,
        which is the time period for which the portfolio data is available.

        Returns: pd.Index: A pandas index representing the time period for which the
        portfolio data is available.

        Notes: The function extracts the index of the prices dataframe,
        which represents the time periods for which data is available for the portfolio.
        The resulting index will be a pandas index object with the same length
        as the number of rows in the prices dataframe."""
        return pd.DatetimeIndex(self.prices.index)

    @property
    def assets(self) -> pd.Index:
        """A property that returns a list of the assets held by the EquityPortfolio object.

        Returns: list: A list of the assets held by the EquityPortfolio object.

        Notes: The function extracts the column names of the prices dataframe,
        which correspond to the assets held by the EquityPortfolio object.
        The resulting list will contain the names of all assets held by the portfolio, without any duplicates.
        """
        return self.prices.columns

    @property
    def weights(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing
        the weights of various assets in the portfolio.

        Returns: pd.DataFrame: A pandas dataframe representing the weights
        of various assets in the portfolio.

        Notes: The function calculates the weights of various assets
        in the portfolio by dividing the equity positions
        for each asset (as represented in the equity dataframe)
        by the total portfolio value (as represented in the nav dataframe).
        Both dataframes are assumed to have the same dimensions.
        The resulting dataframe will show the relative weight
        of each asset in the portfolio at each point in time."""
        return self.equity.apply(lambda x: x / self.nav)

    def __getitem__(self, time: datetime) -> pd.Series:
        """The `__getitem__` method retrieves the stock data for a specific time in the dataframe.
        It returns the stock data for that time.

        The method takes one input parameter:
        - `time`: the time index for which to retrieve the stock data

        Returns:
        - stock data for the input time

        Note that the input time must be in the index of the dataframe,
        otherwise a KeyError will be raised."""
        return self.stocks.loc[time]

    @property
    def trading_costs(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe
        representing the trading costs incurred by the portfolio due to trades made.

        Returns: pd.DataFrame: A pandas dataframe representing the trading
        costs incurred by the portfolio due to trades made.

        Notes: The function calculates the trading costs using the specified
        trading cost model (if available) and the prices and trading
        data represented by the prices and trades_stocks
        dataframes, respectively. If no trading cost model is provided,
        a dataframe with all zeros will be returned.
        The resulting dataframe will have the same dimensions as
        the prices and trades_stocks dataframes,
        showing the trading costs incurred at each point in time for each asset traded.
        """
        if self.trading_cost_model is None:
            return 0.0 * self.prices

        return self.trading_cost_model.eval(self.prices, self.trades_stocks)

    @property
    def equity(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe
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
        as the prices and stocks dataframes."""

        return (self.prices * self.stocks).ffill()

    @property
    def trades_stocks(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing the trades made in the portfolio in terms of stocks.

        Returns: pd.DataFrame: A pandas dataframe representing the trades made in the portfolio in terms of stocks.

        Notes: The function calculates the trades made by the portfolio by taking
        the difference between the current and previous values of the stocks dataframe.
        The resulting values will represent the number of shares of each asset
        bought or sold by the portfolio at each point in time.
        The resulting dataframe will have the same dimensions
        as the stocks dataframe, with NaN values filled with zeros."""
        t = self.stocks.diff()
        t.loc[self.index[0]] = self.stocks.loc[self.index[0]]
        return t.fillna(0.0)

    @property
    def trades_currency(self) -> pd.DataFrame:
        """A property that returns a pandas dataframe representing
        the trades made in the portfolio in terms of currency.

        Returns: pd.DataFrame: A pandas dataframe representing the trades made in the portfolio in terms of currency.

        Notes: The function calculates the trades made in currency by multiplying
        the number of shares of each asset bought or sold (as represented in the trades_stocks dataframe)
        with the current prices of each asset (as represented in the prices dataframe).
        Uses pandas ffill() method to forward fill NaN values in the prices dataframe.
        The resulting dataframe will have the same dimensions as the stocks and prices dataframes.
        """
        return self.trades_stocks * self.prices.ffill()

    @property
    def turnover(self) -> pd.DataFrame:
        return self.trades_currency.abs()

    @property
    def cash(self) -> pd.Series:
        """A property that returns a pandas series representing the cash on hand in the portfolio.

        Returns: pd.Series: A pandas series representing the cash on hand in the portfolio.

        Notes: The function calculates the cash available in the portfolio by subtracting
        the sum of trades currency and cumulative trading costs from the initial cash value specified
        when constructing the object. Uses pandas cumsum() method
        to calculate the cumulative sum of trading costs and
        trades currency along the time axis.
        The resulting series will show how much money is available for further trades at each point in time.
        """
        return (
            self.initial_cash
            - self.trades_currency.sum(axis=1).cumsum()
            - self.trading_costs.sum(axis=1).cumsum()
        )

    @property
    def nav(self) -> pd.Series:
        """Returns a pandas series representing the total value
        of the portfolio's investments and cash.

        Returns: pd.Series: A pandas series representing the
                            total value of the portfolio's investments and cash.
        """
        return self.equity.sum(axis=1) + self.cash

    @property
    def profit(self) -> pd.Series:
        """A property that returns a pandas series representing the
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
        """A function that returns a pandas series representing
        the high-water mark of the portfolio, which is the highest point
        the portfolio value has reached over time.

        Returns: pd.Series: A pandas series representing the
        high-water mark of the portfolio.

        Notes: The function performs a rolling computation based on
        the cumulative maximum of the portfolio's value over time,
        starting from the beginning of the time period being considered.
        Min_periods argument is set to 1 to include the minimum period of one day.
        The resulting series will show the highest value the portfolio has reached at each point in time.
        """
        return self.nav.expanding(min_periods=1).max()

    @property
    def drawdown(self) -> pd.Series:
        """A property that returns a pandas series representing the
        drawdown of the portfolio, which measures the decline
        in the portfolio's value from its (previously) highest
        point to its current point.

        Returns: pd.Series: A pandas series representing the
        drawdown of the portfolio.

        Notes: The function calculates the ratio of the portfolio's current value
        vs. its current high-water-mark and then subtracting the result from 1.
        A positive drawdown means the portfolio is currently worth
        less than its high-water mark. A drawdown of 0.1 implies that the nav is currently 0.9 times the high-water mark
        """
        return 1.0 - self.nav / self.highwater

    def __mul__(self, scalar: float) -> "EquityPortfolio":
        """A method that allows multiplication of the EquityPortfolio object with a scalar constant.

        Args: scalar: A scalar constant that multiplies the number of shares
        of each asset held in the EquityPortfolio object.

        Returns: EquityPortfolio: A new EquityPortfolio object multiplied by the scalar constant.

        Notes: The mul method allows multiplication of an EquityPortfolio object
        with a scalar constant to increase or decrease
        the number of shares held for each asset in the portfolio accordingly.
        The method returns a new EquityPortfolio object with the same prices
        and trading cost model as the original object,
        and with the number of shares for each asset multiplied by the scalar constant
        (as represented in the stocks dataframe).
        Additionally, the initial cash value is multiplied
        by the scalar to maintain the same cash-to-equity ratio as the original portfolio.
        """
        assert scalar > 0
        return EquityPortfolio(
            prices=self.prices,
            stocks=self.stocks * scalar,
            initial_cash=self.initial_cash * scalar,
            trading_cost_model=self.trading_cost_model,
        )

    def __rmul__(self, scalar: float) -> EquityPortfolio:
        """A method that allows multiplication of the EquityPortfolio object with a scalar constant in a reversed order.

        Args: scalar: A scalar constant that multiplies the EquityPortfolio object in a reversed order.

        Returns: EquityPortfolio: A new EquityPortfolio object multiplied by the scalar constant.

        Notes: The rmul method allows multiplication of a scalar
        constant with an EquityPortfolio object in a reversed order"""
        return self.__mul__(scalar)

    def __add__(self, port_new: EquityPortfolio) -> EquityPortfolio:
        """
        A method that allows addition of two EquityPortfolio objects.
        """
        # check if the other object is an EquityPortfolio object
        assert isinstance(port_new, EquityPortfolio)

        # make sure the prices are aligned for overlapping points
        prices_left = self.prices.combine_first(port_new.prices)
        prices_right = port_new.prices.combine_first(self.prices)
        pd.testing.assert_frame_equal(prices_left, prices_right)

        # bring both portfolios to the finer grid
        left = self.reset_prices(prices=prices_left)
        right = port_new.reset_prices(prices=prices_left)

        # just make sure the left and right portfolio are now on exactly the same grid
        pd.testing.assert_index_equal(left.index, right.index)

        # add the stocks
        positions = left.stocks.add(right.stocks, fill_value=0.0)

        # make sure the trading cost models are the same
        return EquityPortfolio(
            prices=prices_right,
            stocks=positions,
            initial_cash=self.initial_cash + port_new.initial_cash,
            trading_cost_model=self.trading_cost_model,
        )

    def reset_prices(self, prices: pd.DataFrame) -> EquityPortfolio:
        """
        A method that constructs an EquityPortfolio object using finer prices.
        """
        # extract the relevant columns from prices
        p = prices[self.assets]

        # the prices need to contain the original index
        assert set(self.index).issubset(set(prices.index))

        # build a frame for the stocks
        stocks = pd.DataFrame(index=prices.index, columns=self.assets)

        # only forward fill stocks on the subgrid induced by the original index
        sub = stocks.truncate(before=self.index[0], after=self.index[-1])
        sub.update(self.stocks)
        sub = sub.ffill()

        # outside the original index, the stocks are zero
        stocks.update(sub)
        stocks = stocks.fillna(0.0)

        return EquityPortfolio(
            prices=p,
            stocks=stocks,
            initial_cash=self.initial_cash,
            trading_cost_model=self.trading_cost_model,
        )

    def truncate(
        self, before: Optional[datetime] = None, after: Optional[datetime] = None
    ) -> EquityPortfolio:
        """
        The truncate method truncates the prices DataFrame, stocks DataFrame
        and the cash series of an EquityPortfolio object.
        The method also optionally accepts a before and/or after argument
        to specify a date range for truncation.

        The method returns a new EquityPortfolio object which is a truncated version
        of the original object, with the same trading cost model
        and initial cash value. The stocks DataFrame is truncated
        using the same before and after arguments and the prices DataFrame
        is truncated similarly. The cash value is truncated
        to match the new date range and the first value of the
        truncated cash series is used as the initial cash value for the new object.

        Note that this method does not modify the original EquityPortfolio object,
        but rather returns a new object.
        :param before:
        :param after:
        :return:
        """
        return EquityPortfolio(
            prices=self.prices.truncate(before=before, after=after),
            stocks=self.stocks.truncate(before=before, after=after),
            trading_cost_model=self.trading_cost_model,
            initial_cash=self.nav.truncate(before=before, after=after).values[0],
        )

    # @property
    # def start(self):
    #     """first index with a profit that is not zero"""
    #     return self.profit.ne(0).idxmax()

    def resample(self, rule: Any) -> EquityPortfolio:
        """The resample method resamples an EquityPortfolio object to a new frequency
        specified by the rule argument.
        A new EquityPortfolio object is created with the original prices
        DataFrame and the resampled stocks DataFrame. The objects trading cost model and initial cash value
        are also copied into the new object.

        Note that the resample method does not modify the original EquityPortfolio object,
        but rather returns a new object.
        """
        # iron out the stocks index
        stocks = iron_frame(self.stocks, rule=rule)

        return EquityPortfolio(
            prices=self.prices,
            stocks=stocks,
            trading_cost_model=self.trading_cost_model,
            initial_cash=self.initial_cash,
        )

    def metrics(
        self,
        benchmark: Any = None,
        rf: float = 0.0,
        display: bool = True,
        mode: str = "basic",
        sep: bool = False,
        compound: bool = True,
        periods_per_year: int = 252,
        prepare_returns: bool = True,
        match_dates: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        The metrics method calculates the performance metrics of an EquityPortfolio object.

        :param kwargs:
        :return:
        """
        return qs.reports.metrics(
            returns=self.nav.pct_change().dropna(),
            benchmark=benchmark,
            rf=rf,
            display=display,
            mode=mode,
            sep=sep,
            compounded=compound,
            periods_per_year=periods_per_year,
            prepare_returns=prepare_returns,
            match_dates=match_dates,
            **kwargs,
        )

    def plots(
        self,
        benchmark: Any = None,
        grayscale: bool = False,
        figsize: Tuple[int, int] = (8, 5),
        mode: str = "basic",
        compounded: bool = True,
        periods_per_year: int = 252,
        prepare_returns: bool = True,
        match_dates: bool = True,
        **kwargs: Any,
    ) -> Any:
        return qs.reports.plots(
            returns=self.nav.pct_change().dropna(),
            benchmark=benchmark,
            grayscale=grayscale,
            figsize=figsize,
            mode=mode,
            compounded=compounded,
            periods_per_year=periods_per_year,
            prepare_returns=prepare_returns,
            match_dates=match_dates,
            **kwargs,
        )

    def plot(self, kind: Plot, **kwargs: Any) -> Any:
        return kind.plot(returns=self.nav.pct_change().dropna(), **kwargs)

    def html(
        self,
        benchmark: Any = None,
        rf: float = 0.0,
        grayscale: bool = False,
        title: str = "Strategy Tearsheet",
        output: Any = None,
        compounded: bool = True,
        periods_per_year: int = 252,
        download_filename: str = "quantstats-tearsheet.html",
        figfmt: str = "svg",
        template_path: Any = None,
        match_dates: bool = True,
        **kwargs: Any,
    ) -> Any:
        return qs.reports.html(
            returns=self.nav.pct_change().dropna(),
            benchmark=benchmark,
            rf=rf,
            grayscale=grayscale,
            title=title,
            output=output,
            compounded=compounded,
            periods_per_year=periods_per_year,
            download_filename=download_filename,
            figfmt=figfmt,
            template_path=template_path,
            match_dates=match_dates,
            **kwargs,
        )

    def snapshot(
        self,
        grayscale: bool = False,
        figsize: Tuple[int, int] = (10, 8),
        title: str = "Portfolio Summary",
        fontname: str = "Arial",
        lw: float = 1.5,
        mode: str = "comp",
        subtitle: bool = True,
        savefig: Any = None,
        show: bool = True,
        log_scale: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        The snapshot method creates a snapshot of the performance of an EquityPortfolio object.

        :param grayscale:
        :param figsize:
        :param title:
        :param fontname:
        :param lw:
        :param mode:
        :param subtitle:
        :param savefig:
        :param show:
        :param log_scale:
        :param kwargs:
        :return:
        """
        return qs.plots.snapshot(
            returns=self.nav.pct_change().dropna(),
            grayscale=grayscale,
            figsize=figsize,
            title=title,
            fontname=fontname,
            lw=lw,
            mode=mode,
            subtitle=subtitle,
            savefig=savefig,
            show=show,
            log_scale=log_scale,
            **kwargs,
        )
