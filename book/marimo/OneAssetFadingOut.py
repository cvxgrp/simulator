import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # One asset fading out
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
def __(np, pd):
    # two assets, A and B, constant price for A=100 and B=200
    prices = pd.read_csv("data/prices.csv", header=0, index_col=0, parse_dates=True)
    prices.loc["2022-01-03", "B"] = np.NaN
    prices.loc["2022-01-04", "B"] = np.NaN
    prices
    return (prices,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Iterate
        """
    )
    return


@app.cell
def __(Builder, np, prices):
    b = Builder(prices=prices, initial_aum=2000)

    for t, state in b:
        b.weights = np.ones(len(state.assets)) / len(state.assets)
        b.aum = state.aum
    return b, state, t


@app.cell
def __(b):
    b.units
    return


@app.cell
def __(b):
    b.prices
    return


@app.cell
def __(b):
    portfolio = b.build()
    return (portfolio,)


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
