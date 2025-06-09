"""monkey portfolios."""

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

    # Initialize random number generator once to be used by all cells
    rng = np.random.default_rng(42)


@app.cell
def _():
    mo.md(r"""# Monkey portfolios""")
    return


@app.cell
def _():
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _time, _state in _builder:
        _n = len(_state.assets)
        _w = rng.random(_n)
        _w = _w / np.sum(_w)
        assert np.all(_w >= 0)
        assert np.allclose(np.sum(_w), 1)
        _builder.weights = _w
        _builder.aum = _state.aum

    _portfolio = _builder.build()
    _portfolio.snapshot()
    return


@app.cell
def _():
    _builder = Builder(prices=prices, initial_aum=1000000.0)
    for _time, _state in _builder:
        _n = len(_state.assets)
        _w = rng.random(_n)
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
