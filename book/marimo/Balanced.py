"""1/n portfolio strategies implemented using the CVX simulator.

This module demonstrates various implementations of the 1/n portfolio strategy,
where equal weights are assigned to all assets in the portfolio. It explores
different approaches to constructing and optimizing such portfolios:

1. Simple equal weighting with daily rebalancing
2. Formulating the problem as a convex optimization using different objectives:
   - Minimization of the Euclidean norm of weights
   - Minimization of the infinity norm of weights
   - Maximization of the entropy of weights
   - Minimization of tracking error to a 1/n target
3. Sparse updating strategy that only rebalances when the portfolio drifts
   significantly from the target weights

Each approach is implemented and visualized to compare performance.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.io as pio
    import polars as pl

    pd.options.plotting.backend = "plotly"

    # Ensure Plotly works with Marimo
    pio.renderers.default = "plotly_mimetype"

    path = mo.notebook_location() / "public" / "stock-prices.csv"

    # from cvxsimulator.builder import polars2pandas
    date_col = "date"
    dframe = pl.read_csv(str(path), try_parse_dates=True)

    dframe = dframe.with_columns(pl.col(date_col).cast(pl.Datetime("ns")))
    dframe = dframe.with_columns([pl.col(col).cast(pl.Float64) for col in dframe.columns if col != date_col])
    prices = dframe.to_pandas().set_index(date_col)

    from cvxsimulator import Builder


@app.cell
def _():
    """Display the introduction and overview of the 1/n portfolio family.

    This cell provides an introduction to the 1/n portfolio strategy and outlines
    the different approaches that will be explored in this notebook:
    - Vanilla implementation with daily rebalancing
    - Formulation as convex optimization problems with different objectives
    - Sparse updating strategy

    """
    mo.md(
        r"""
    # The 1/n family

    We play with the $1/n$ portfolio. We start with a vanilla implementation using daily rebalancing.
    Every portfolio should be the solution of a convex optimization problem,
    see https://www.linkedin.com/pulse/stock-picking-convex-programs-thomas-schmelzer.
    We do that and show methods to construct the portfolio with

    * the minimization of the Euclidean norm of the weights.
    * the minimization of the $\infty$ norm of the weights.
    * and the maximization of the Entropy of the weights.
    * the minimization of the tracking error to an $1/n$ portfolio.

    We also play with sparse updates, e.g. rather than rebalancing daily,
    we act only once the deviation of our drifted portfolio got too large from the target $1/n$ portfolio.

    This problem has been discussed https://www.linkedin.com/feed/update/urn:li:activity:7149432321388064768/
    """
    )
    return


@app.cell
def _():
    """Implement a simple 1/n portfolio strategy with daily rebalancing.

    This cell creates a portfolio that allocates equal weight to each asset
    and rebalances daily to maintain these equal weights. It then builds the
    portfolio and displays a snapshot of its performance.

    """
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _n = len(_state.assets)
        _builder.weights = np.ones(_n) / _n
        _builder.aum = _state.aum

    _portfolio = _builder.build()
    _portfolio.snapshot()
    return


@app.cell
def _():
    """Display the introduction to the cvxpy implementation section.

    This cell introduces the concept of formulating the 1/n portfolio problem
    as a convex optimization problem using cvxpy. It explains the benefits of
    this approach, particularly for adding constraints and understanding the
    relationship to Tikhonov regularization.

    """
    mo.md(
        r"""
    ## With cvxpy

    Formulating the problem above as a convex program is most useful when additional constraints have
    to be reflected. It also helps to link the 1/n portfolio to Tikhonov regularization and interpret
    its solution as a cornercase for more complex portfolios we are building
    """
    )
    return


@app.cell
def _():
    """Import the cvxpy library for convex optimization.

    This cell imports the cvxpy library, which will be used to formulate
    and solve the 1/n portfolio optimization problems in subsequent cells.

    Returns:
    -------
    tuple
        A tuple containing the cvxpy module

    """
    import cvxpy as cp

    return (cp,)


@app.cell
def _(mo):
    """Display the introduction to the Euclidean norm minimization approach.

    This cell introduces the first convex optimization approach: minimizing
    the Euclidean norm of the weight vector. This approach produces the same
    results as the simple 1/n portfolio but opens the door to more complex
    convex optimization formulations.

    Parameters
    ----------
    mo : marimo.Module
        The marimo module object

    """
    mo.md(
        r"""
    ### Minimization of the Euclidean norm

    We minimize the Euclidean norm of the weight vector. Same results as above but we
    open door to the world of convex paradise.
    """
    )
    return


@app.cell
def _(cp):
    """Implement a 1/n portfolio using Euclidean norm minimization.

    This cell creates a portfolio by formulating and solving a convex optimization
    problem that minimizes the Euclidean norm (L2 norm) of the weight vector,
    subject to the constraints that weights are non-negative and sum to 1.
    This approach produces the same results as the simple 1/n portfolio.

    Parameters
    ----------
    Builder : class
        The Builder class from cvx.simulator
    cp : module
        The cvxpy module
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    """
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _n = len(_state.assets)
        _weights = cp.Variable(_n)
        _objective = cp.norm(_weights, 2)
        _constraints = [_weights >= 0, cp.sum(_weights) == 1]
        cp.Problem(objective=cp.Minimize(_objective), constraints=_constraints).solve(solver=cp.CLARABEL)
        _builder.weights = _weights.value
        _builder.aum = _state.aum
    _portfolio = _builder.build()
    _portfolio.snapshot()
    return


@app.cell
def _(mo):
    """Display the introduction to the infinity norm minimization approach.

    This cell introduces the second convex optimization approach: minimizing
    the infinity norm of the weight vector. This approach is based on an idea
    by Vladimir Markov and provides another way to formulate the 1/n portfolio
    problem.

    Parameters
    ----------
    mo : marimo.Module
        The marimo module object

    """
    mo.md(
        r"""
    ### Minimization of the $\infty$ norm

    Based on an idea by Vladimir Markov
    """
    )
    return


@app.cell
def _(cp):
    """Implement a 1/n portfolio using infinity norm minimization.

    This cell creates a portfolio by formulating and solving a convex optimization
    problem that minimizes the infinity norm (L∞ norm) of the weight vector,
    subject to the constraints that weights are non-negative and sum to 1.
    This approach, based on Vladimir Markov's idea, is another way to achieve
    equal weighting across assets.

    Parameters
    ----------
    Builder : class
        The Builder class from cvx.simulator
    cp : module
        The cvxpy module
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    """
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _n = len(_state.assets)
        _weights = cp.Variable(_n)
        _objective = cp.norm_inf(_weights)
        _constraints = [_weights >= 0, cp.sum(_weights) == 1]
        cp.Problem(objective=cp.Minimize(_objective), constraints=_constraints).solve(solver=cp.CLARABEL)
        _builder.weights = _weights.value
        _builder.aum = _state.aum
    _portfolio = _builder.build()
    _portfolio.snapshot()
    return


@app.cell
def _():
    """Display the introduction to the entropy maximization approach.

    This cell introduces the third convex optimization approach: maximizing
    the entropy of the weight vector. This approach provides yet another way
    to formulate the 1/n portfolio problem and arrives at the same result
    as the previous methods.

    Parameters
    ----------
    mo : marimo.Module
        The marimo module object

    """
    mo.md(
        r"""
    ### Maximization of the entropy

    One can also maximize the entropy to arrive at the same result
    """
    )
    return


@app.cell
def _(cp):
    """Implement a 1/n portfolio using entropy maximization.

    This cell creates a portfolio by formulating and solving a convex optimization
    problem that maximizes the entropy of the weight vector, subject to the
    constraints that weights are non-negative and sum to 1. Entropy maximization
    is a well-known approach that leads to equal weighting when no other
    information is available.

    Parameters
    ----------
    Builder : class
        The Builder class from cvx.simulator
    cp : module
        The cvxpy module
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    """
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _n = len(_state.assets)
        _weights = cp.Variable(_n)
        _objective = cp.sum(cp.entr(_weights))
        _constraints = [_weights >= 0, cp.sum(_weights) == 1]
        cp.Problem(objective=cp.Maximize(_objective), constraints=_constraints).solve(solver=cp.CLARABEL)
        _builder.weights = _weights.value
        _builder.aum = _state.aum
    _portfolio = _builder.build()
    _portfolio.snapshot()
    return


@app.cell
def _():
    """Display the introduction to the tracking error minimization approach.

    This cell introduces the fourth convex optimization approach: minimizing
    the tracking error to a 1/n target portfolio. This approach explicitly
    formulates the problem as minimizing the distance to the equal-weight
    portfolio.

    Parameters
    ----------
    mo : marimo.Module
        The marimo module object

    """
    mo.md(r"""### Minimization of the tracking error""")
    return


@app.cell
def _(cp):
    """Implement a 1/n portfolio using tracking error minimization.

    This cell creates a portfolio by formulating and solving a convex optimization
    problem that minimizes the tracking error (Euclidean distance) between the
    portfolio weights and a target 1/n portfolio, subject to the constraints
    that weights are non-negative and sum to 1. This approach explicitly
    targets the equal-weight portfolio.

    Parameters
    ----------
    Builder : class
        The Builder class from cvx.simulator
    cp : module
        The cvxpy module
    np : module
        The numpy module
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    """
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _n = len(_state.assets)
        _weights = cp.Variable(_n)
        _objective = cp.norm(_weights - np.ones(_n) / _n, 2)
        _constraints = [_weights >= 0, cp.sum(_weights) == 1]
        cp.Problem(objective=cp.Minimize(_objective), constraints=_constraints).solve(solver=cp.CLARABEL)
        _builder.weights = _weights.value
        _builder.aum = _state.aum
    _portfolio = _builder.build()
    _portfolio.snapshot()
    return


@app.cell
def _():
    """Display the introduction to the sparse updates approach.

    This cell introduces a more practical approach to implementing the 1/n
    portfolio strategy: using sparse updates instead of daily rebalancing.
    This approach acknowledges that in practice, frequent rebalancing is
    costly and unnecessary, and that small deviations from the target
    weights can be tolerated.

    Parameters
    ----------
    mo : marimo.Module
        The marimo module object

    """
    mo.md(
        r"""
    ## With sparse updates

    In practice we do not want to rebalance the portfolio every day. We tolerate our portfolio
    is not an exact $1/n$ portfolio. We may expect slightly weaker results
    """
    )
    return


@app.cell
def _():
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _n = len(_state.assets)
        target = np.ones(_n) / _n
        drifted = _state.weights[_state.assets].fillna(0.0)
        delta = (target - drifted).abs().sum()
        if delta > 0.2:
            _builder.weights = target
        else:
            _builder.position = _state.position
        _builder.aum = _state.aum
    _portfolio = _builder.build()
    _portfolio.snapshot()
    return


if __name__ == "__main__":
    app.run()
