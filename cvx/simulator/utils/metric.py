from math import sqrt


def sharpe(ts, n=252):
    """
    compute the sharpe ratio of a time series
    """
    return ts.mean() / ts.std() * sqrt(n)
