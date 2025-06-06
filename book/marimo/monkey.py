"""monkey portfolios."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Monkey portfolios""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell
def _(pd):
    from cvx.simulator import Builder

    pd.options.plotting.backend = "plotly"
    return (Builder,)


@app.cell
def _(mo, pd):
    prices = pd.read_csv(mo.notebook_location() / "data" / "stock-prices.csv", header=0, index_col=0, parse_dates=True)
    return (prices,)


@app.cell
def _(Builder, np, prices):
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    np.random.seed(42)
    for _time, _state in _builder:
        _n = len(_state.assets)
        _w = np.random.rand(_n)
        _w = _w / np.sum(_w)
        assert np.all(_w >= 0)
        assert np.allclose(np.sum(_w), 1)
        _builder.weights = _w
        _builder.aum = _state.aum

    _portfolio = _builder.build()
    _portfolio.snapshot()
    return


@app.cell
def _(Builder, np, prices):
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    np.random.seed(42)
    for _time, _state in _builder:
        _n = len(_state.assets)
        _w = np.random.rand(_n)
        _w = _w / np.sum(_w)
        assert np.all(_w >= 0)
        assert np.allclose(np.sum(_w), 1)
        _builder.weights = _w
        _builder.aum = _state.aum

    _portfolio = _builder.build()
    _portfolio.snapshot()

    return


if __name__ == "__main__":
    app.run()
