from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


def synthetic_returns(
    prices: pd.DataFrame, information_ratio: float, forward_smoothing: int
) -> pd.DataFrame:
    """
    prices: a DataFrame of prices
    information_ratio: the desired information ratio of the synthetic returns

    returns: a DataFrame of "synthetic return predictions" computed as
    alpha*(returns+noise), where alpha=var_r / (var_r + var_eps); this is the
    coefficient that minimize the variance of the prediction error under the
    above model.
    """
    rng = np.random.default_rng(1)

    returns = prices.pct_change()
    returns = returns.rolling(forward_smoothing).mean().shift(-(forward_smoothing - 1))
    var_r = returns.var()

    alpha = information_ratio**2
    var_eps = var_r * (1 - alpha) / alpha
    noise = rng.normal(0, np.sqrt(var_eps), size=returns.shape)
    returns = alpha * (returns + noise)

    return returns


@dataclass(frozen=True)
class OptimizationInput:
    """
    At time t, we have data from t-lookback to t-1.
    """

    mean: pd.Series
    covariance: pd.DataFrame
    risk_target: float

    @property
    def n_assets(self) -> int:
        return len(self.mean)


def basic_markowitz(inputs: OptimizationInput) -> np.ndarray:
    """Compute the basic Markowitz portfolio weights."""

    mu, Sigma = inputs.mean.values, inputs.covariance.values

    w = cp.Variable(inputs.n_assets)
    c = cp.Variable()
    objective = mu @ w

    chol = np.linalg.cholesky(Sigma)
    constraints = [
        cp.sum(w) + c == 1,
        cp.norm2(chol.T @ w) <= inputs.risk_target,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    return w.value, c.value
