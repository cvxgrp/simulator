"""monkey portfolios."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Monkey portfolios""")
    return


@app.cell
async def _():
    try:
        import sys

        if "pyodide" in sys.modules:
            import micropip

            await micropip.install("cvxsimulator")

    except ImportError:
        pass

    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl

    pd.options.plotting.backend = "plotly"

    return mo, np, pl


@app.cell
def _(pd):
    from cvx.simulator import Builder

    pd.options.plotting.backend = "plotly"
    return (Builder,)


@app.cell
def _(mo, pl):
    from cvx.simulator.builder import polars2pandas

    # Step 1: Read the CSV, parse dates
    prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock-prices.csv"), try_parse_dates=True)

    prices = polars2pandas(prices)
    print(prices)
    print(prices.dtypes)
    print(prices.index.dtype)
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
