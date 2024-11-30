import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
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
def __():
    import numpy as np
    import pandas as pd

    from cvx.simulator import Builder

    return Builder, np, pd


@app.cell
def __(pd):
    # load prices from flat csv file
    prices = pd.read_csv(
        "data/stock-prices.csv", header=0, index_col=0, parse_dates=True
    )
    return (prices,)


@app.cell
def __(Builder, assets, builder, n, np, prices, state):
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _assets = state.assets
        _n = len(assets)
        builder.weights = np.ones(n) / n
        builder.aum = state.aum
    portfolio = builder.build()
    return (portfolio,)


@app.cell
def __(portfolio):
    portfolio.snapshot(aggregate=True)
    return


@app.cell
def __(mo):
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
def __():
    import cvxpy as cp

    return (cp,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Minimization of the Euclidean norm

        We minimize the Euclidean norm of the weight vector. Same results as above but we
        open door to the world of convex paradise.
        """
    )
    return


@app.cell
def __(Builder, assets, builder, cp, n, prices, state, weights):
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _assets = state.assets
        _n = len(assets)
        _weights = cp.Variable(n)
        _objective = cp.norm(weights, 2)
        _constraints = [weights >= 0, cp.sum(weights) == 1]
        cp.Problem(objective=cp.Minimize(_objective), constraints=_constraints).solve(
            solver=cp.CLARABEL
        )
        builder.weights = weights.value
        builder.aum = state.aum
    portfolio_1 = builder.build()
    portfolio_1.snapshot(aggregate=True)
    return (portfolio_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Minimization of the $\infty$ norm

        Based on an idea by Vladimir Markov
        """
    )
    return


@app.cell
def __(Builder, assets, builder, cp, n, prices, state, weights):
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _assets = state.assets
        _n = len(assets)
        _weights = cp.Variable(n)
        _objective = cp.norm_inf(weights)
        _constraints = [weights >= 0, cp.sum(weights) == 1]
        cp.Problem(objective=cp.Minimize(_objective), constraints=_constraints).solve(
            solver=cp.CLARABEL
        )
        builder.weights = weights.value
        builder.aum = state.aum
    portfolio_2 = builder.build()
    portfolio_2.snapshot(aggreagate=True)
    return (portfolio_2,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Maximization of the entropy

        One can also maximize the entropy to arrive at the same result
        """
    )
    return


@app.cell
def __(Builder, assets, builder, cp, n, prices, state, weights):
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _assets = state.assets
        _n = len(assets)
        _weights = cp.Variable(n)
        _objective = cp.sum(cp.entr(weights))
        _constraints = [weights >= 0, cp.sum(weights) == 1]
        cp.Problem(objective=cp.Maximize(_objective), constraints=_constraints).solve(
            solver=cp.CLARABEL
        )
        builder.weights = weights.value
        builder.aum = state.aum
    portfolio_3 = builder.build()
    portfolio_3.snapshot()
    return (portfolio_3,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Minimization of the tracking error
        """
    )
    return


@app.cell
def __(Builder, assets, builder, cp, n, np, prices, state, weights):
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _assets = state.assets
        _n = len(assets)
        _weights = cp.Variable(n)
        _objective = cp.norm(weights - np.ones(n) / n, 2)
        _constraints = [weights >= 0, cp.sum(weights) == 1]
        cp.Problem(objective=cp.Minimize(_objective), constraints=_constraints).solve(
            solver=cp.CLARABEL
        )
        builder.weights = weights.value
        builder.aum = state.aum
    portfolio_4 = builder.build()
    portfolio_4.snapshot(aggregate=True)
    return (portfolio_4,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## With sparse updates

        In practice we do not want to rebalance the portfolio every day. We tolerate our portfolio
        is not an exact $1/n$ portfolio. We may expect slightly weaker results
        """
    )
    return


@app.cell
def __(Builder, assets, builder, n, np, prices, state):
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _, _state in _builder:
        _assets = state.assets
        _n = len(assets)
        target = np.ones(n) / n
        drifted = state.weights[assets].fillna(0.0)
        delta = (target - drifted).abs().sum()
        if delta > 0.2:
            builder.weights = target
        else:
            builder.position = state.position
        builder.aum = state.aum
    portfolio_5 = builder.build()
    portfolio_5.snapshot(aggregate=True)
    return delta, drifted, portfolio_5, target


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
