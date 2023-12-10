from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


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
