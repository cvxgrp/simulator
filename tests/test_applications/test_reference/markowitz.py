"""Toying with markowitz portfolio optimization."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


def synthetic_returns(prices: pd.DataFrame, information_ratio: float, forward_smoothing: int) -> pd.DataFrame:
    """Generate synthetic return predictions based on historical prices.

    This function creates synthetic return predictions by combining actual returns
    with random noise. The weighting of returns vs. noise is determined by the
    information ratio parameter.

    Parameters
    ----------
    prices : pd.DataFrame
        A DataFrame of historical asset prices
    information_ratio : float
        The desired information ratio of the synthetic returns
    forward_smoothing : int
        Number of periods to use for forward-looking rolling mean

    Returns:
    -------
    pd.DataFrame
        A DataFrame of synthetic return predictions computed as
        alpha*(returns+noise), where alpha=var_r / (var_r + var_eps); this is the
        coefficient that minimizes the variance of the prediction error

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
    """Input data for Markowitz portfolio optimization.

    This class encapsulates the data needed for Markowitz portfolio optimization:
    expected returns (mean), covariance matrix, and risk target.

    At time t, we have data from t-lookback to t-1.

    Attributes:
    ----------
    mean : pd.Series
        Expected returns for each asset
    covariance : pd.DataFrame
        Covariance matrix of asset returns
    risk_target : float
        Target level of portfolio risk

    """

    mean: pd.Series
    covariance: pd.DataFrame
    risk_target: float

    @property
    def n_assets(self) -> int:
        """Get the number of assets in the optimization problem.

        Returns:
        -------
        int
            The number of assets, determined by the length of the mean vector

        """
        return len(self.mean)


def basic_markowitz(inputs: OptimizationInput) -> np.ndarray | None:
    """Compute the basic Markowitz portfolio weights.

    This function solves the Markowitz portfolio optimization problem:
    maximize expected return subject to a risk constraint.

    The optimization problem is:
    maximize mu^T w
    subject to:
        sum(w) + c = 1
        ||chol^T w|| <= risk_target

    where w are the portfolio weights, c is the cash weight,
    mu is the vector of expected returns, and chol is the Cholesky
    decomposition of the covariance matrix.

    Parameters
    ----------
    inputs : OptimizationInput
        Input data for the optimization, including expected returns,
        covariance matrix, and risk target

    Returns:
    -------
    tuple[np.ndarray, float]
        A tuple containing:
        - The optimal portfolio weights as a numpy array
        - The optimal cash weight as a float

    """
    mu, sigma = inputs.mean.to_numpy(), inputs.covariance.to_numpy()

    w = cp.Variable(inputs.n_assets)
    c = cp.Variable()
    objective = mu @ w

    chol = np.linalg.cholesky(sigma)
    constraints = [
        cp.sum(w) + c == 1,
        cp.norm2(chol.T @ w) <= inputs.risk_target,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    return w.value
