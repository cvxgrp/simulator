import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    """
    Metrics class for a portfolio
    """
    daily_profit: pd.Series
    days_per_year: int = 252

    def __post_init__(self):
        # check there are no NaNs in the profit series
        assert not self.daily_profit.isnull().values.any()
        assert self.days_per_year > 0
        assert self.daily_profit.index.is_monotonic_increasing

    @property
    def total_profit(self):
        """
        Computes total profit of the portfolio
        """
        return self.daily_profit.sum()

    @property
    def mean_profit(self):
        """
        Computes mean daily profits
        """
        return self.daily_profit.mean()

    @property
    def std_profit(self):
        """
        Computes standard deviation of daily profits
        """
        return self.daily_profit.std()

    @property
    def sr_profit(self):
        """
        Computes the Sharpe ratio of daily profits
        """
        # print(self.mean_profit)
        return self.mean_profit * np.sqrt(self.days_per_year) \
               / self.std_profit