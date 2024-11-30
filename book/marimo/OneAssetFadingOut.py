import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(r"""# One asset fading out""")
    return


@app.cell
def __(__file__):
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from cvx.simulator import Builder

    folder = Path(__file__).parent
    return Builder, Path, folder, np, pd


@app.cell
def __(folder, np, pd):
    # two assets, A and B, constant price for A=100 and B=200
    prices = pd.read_csv(
        folder / "data" / "prices.csv", header=0, index_col=0, parse_dates=True
    )
    prices.loc["2022-01-03", "B"] = np.nan
    prices.loc["2022-01-04", "B"] = np.nan
    prices
    return (prices,)


@app.cell
def __(mo):
    mo.md(r"""## Iterate""")
    return


@app.cell
def __(Builder, np, prices):
    _builder = Builder(prices=prices, initial_aum=2000)

    for t, _state in _builder:
        _builder.weights = np.ones(len(_state.assets)) / len(_state.assets)
        _builder.aum = _state.aum

    portfolio = _builder.build()
    return portfolio, t


@app.cell
def __(portfolio):
    portfolio.prices
    return


@app.cell
def __(portfolio):
    portfolio.nav
    return


@app.cell
def __(portfolio):
    portfolio.weights
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
