import numpy as np
import pandas as pd


def _periods(ts):
    """
    compute the number of periods in a time series
    """
    series = pd.Series(data=ts.index)
    return 365 * 24 * 60 * 60 / (series.diff().dropna().mean().total_seconds())


def sharpe(ts, n=None):
    """
    compute the sharpe ratio of a time series
    """
    std = ts.std()
    if std > 0:
        n = n or _periods(ts)
        return (ts.mean() / std) * np.sqrt(n)
    else:
        return np.inf
