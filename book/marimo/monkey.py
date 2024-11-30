import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Monkey portfolios
        """
    )
    return


@app.cell
def __():
    import numpy as np
    import pandas as pd

    return np, pd


@app.cell
def __(pd):
    from cvx.simulator import Builder

    pd.options.plotting.backend = "plotly"
    return (Builder,)


@app.cell
def __(pd):
    prices = pd.read_csv(
        "data/stock-prices.csv", header=0, index_col=0, parse_dates=True
    )
    return (prices,)


@app.cell
def __(Builder, n, np, prices, state, w):
    b = Builder(prices=prices, initial_aum=1000000.0)
    np.random.seed(42)
    for _time, _state in b:
        _n = len(_state.assets)
        _w = np.random.rand(n)
        _w = w / np.sum(w)
        assert np.all(_w >= 0)
        assert np.allclose(np.sum(_w), 1)
        b.weights = w
        b.aum = _state.aum
    return (b,)


@app.cell
def __(b):
    _portfolio = b.build()
    _portfolio.nav.plot()
    return


@app.cell
def __(Builder, n, np, prices, state, w):
    b_1 = Builder(prices=prices, initial_aum=1000000.0)
    np.random.seed(42)
    for _time, _state in b_1:
        _n = len(state.assets)
        _w = np.random.rand(n)
        _w = w / np.sum(w)
        assert np.all(_w >= 0)
        assert np.allclose(np.sum(_w), 1)
        b_1.weights = w
        b_1.aum = state.aum
    return (b_1,)


@app.cell
def __(b_1):
    _portfolio = b_1.build()
    _portfolio.snapshot(aggregate=True)
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
