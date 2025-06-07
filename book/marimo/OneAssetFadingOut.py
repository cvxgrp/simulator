"""Demonstration of portfolio behavior when an asset fades out.

This module demonstrates how the CVX simulator handles the case where an asset
becomes unavailable (fades out) during the simulation period. It creates a simple
portfolio with two assets, A and B, where B becomes unavailable after a certain date.
The portfolio is rebalanced to maintain equal weights among the available assets.

The notebook shows:
1. How to create a price dataset with missing values
2. How to build a portfolio with the Builder class
3. How the portfolio automatically adjusts when an asset fades out
4. How to examine the resulting portfolio's prices, NAV, and weights
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(mo):
    """Display the title of the notebook.

    Parameters
    ----------
    mo : marimo.Module
        The marimo module object

    """
    mo.md(r"""# One asset fading out""")
    return


@app.cell
async def _():
    """Import required libraries and modules.

    This cell imports the necessary libraries and modules for the simulation:
    - marimo: For notebook functionality
    - numpy: For numerical operations
    - pandas: For data manipulation
    - Builder: From cvx.simulator for portfolio simulation

    Returns
    -------
    tuple
        A tuple containing the imported modules (Builder, mo, np, pd)

    """
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

    from cvx.simulator import Builder

    return Builder, mo, np, pd, pl


@app.cell
def _(mo, np, pl):
    from cvx.simulator.builder import polars2pandas

    # Step 1: Read the CSV, parse dates
    prices = pl.read_csv(str(mo.notebook_location() / "public" / "prices.csv"), try_parse_dates=True)

    prices = polars2pandas(prices)
    print(prices)
    print(prices.dtypes)
    print(prices.index.dtype)

    prices.loc["2022-01-03", "B"] = np.nan
    prices.loc["2022-01-04", "B"] = np.nan
    print(prices)
    return (prices,)


@app.cell
def _(mo):
    mo.md(r"""## Iterate""")
    return


@app.cell
def _(Builder, np, prices):
    """Build a portfolio with equal weights that adapts to an asset fading out.

    This cell:
    1. Creates a Builder instance with the modified price data
    2. Iterates through time, setting equal weights for all available assets
       at each time step (automatically adapting when asset B fades out)
    3. Builds and returns the final portfolio

    Parameters
    ----------
    Builder : class
        The Builder class from cvx.simulator
    np : module
        The numpy module
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns,
        where asset B fades out (has NaN values) on certain dates

    Returns
    -------
    tuple
        A tuple containing the built portfolio

    """
    _builder = Builder(prices=prices, initial_aum=2000)

    for _t, _state in _builder:
        _builder.weights = np.ones(len(_state.assets)) / len(_state.assets)
        _builder.aum = _state.aum

    portfolio = _builder.build()
    return (portfolio,)


@app.cell
def _(portfolio):
    """Display the prices DataFrame from the portfolio.

    This cell shows the price data in the portfolio, including the
    missing values for asset B after it fades out.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object built by the previous cell

    """
    print(portfolio.prices)
    return


@app.cell
def _(portfolio):
    print(portfolio.nav)
    return


@app.cell
def _(portfolio):
    """Display the portfolio weights over time.

    This cell shows how the portfolio weights evolve over time,
    particularly how the weights adjust when asset B fades out
    (shifting from 50/50 to 100% in asset A).

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object built by the previous cell

    """
    print(portfolio.weights)
    return


if __name__ == "__main__":
    app.run()
