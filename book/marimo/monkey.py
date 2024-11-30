import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(r"""# Monkey portfolios""")
    return


@app.cell
def __(__file__):
    from pathlib import Path

    import numpy as np
    import pandas as pd

    folder = Path(__file__).parent
    return Path, folder, np, pd


@app.cell
def __(pd):
    from cvx.simulator import Builder

    pd.options.plotting.backend = "plotly"
    return (Builder,)


@app.cell
def __(folder, pd):
    prices = pd.read_csv(
        folder / "data" / "stock-prices.csv", header=0, index_col=0, parse_dates=True
    )
    return (prices,)


@app.cell
def __(Builder, np, prices):
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
    _portfolio.snapshot(aggregate=True)
    return


@app.cell
def __(Builder, np, prices):
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
    _portfolio.snapshot(aggregate=True)

    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
