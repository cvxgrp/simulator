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

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.io as pio
    import polars as pl

    pd.options.plotting.backend = "plotly"

    # Ensure Plotly works with Marimo
    pio.renderers.default = "plotly_mimetype"

    path = mo.notebook_location() / "public" / "prices.csv"

    # from cvxsimulator.builder import polars2pandas
    date_col = "date"
    dframe = pl.read_csv(str(path), try_parse_dates=True)

    dframe = dframe.with_columns(pl.col(date_col).cast(pl.Datetime("ns")))
    dframe = dframe.with_columns([pl.col(col).cast(pl.Float64) for col in dframe.columns if col != date_col])
    prices = dframe.to_pandas().set_index(date_col)

    prices.loc["2022-01-03", "B"] = np.nan
    prices.loc["2022-01-04", "B"] = np.nan

    from cvxsimulator import Builder


@app.cell
def _():
    """Display the title of the notebook.

    Parameters
    ----------
    mo : marimo.Module
        The marimo module object

    """
    mo.md(r"""# One asset fading out""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Iterate""")
    return


@app.cell
def _():
    """Build a portfolio with equal weights that adapts to an asset fading out.

    This cell:
    1. Creates a Builder instance with the modified price data
    2. Iterates through time, setting equal weights for all available assets
       at each time step (automatically adapting when asset B fades out)
    3. Builds and returns the final portfolio

    Returns:
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


if __name__ == "__main__":
    app.run()
