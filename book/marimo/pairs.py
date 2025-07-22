"""pairs trading."""

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

    from loguru import logger

    from cvxsimulator import Builder

    # Initialize random number generator once to be used by all cells
    rng = np.random.default_rng(42)


@app.cell
def _():
    """Display the title of the notebook.

    Parameters
    ----------
    mo : marimo.Module
        The marimo module object

    """
    mo.md(r"""# Almost pairs trading""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This little exercise goes back to an idea by Stephen Boyd:

    The simulator should always be completely agnostic as to the trading policy.
    You should even demonstrate this with silly policies.
    Like here’s one:  Each day choose names from the universe at random.
    Buy one (say 0.1 of your portfolio wealth) and short one the same amount.
    Not a good strategy, but a valid one.
    Of course the simulate will terminate if you go bust (which seems likely).
    """
    )
    return


@app.cell
def _():
    """Implement the pairs trading strategy and build the portfolio.

    This cell:
    1. Loads price data from a CSV file
    2. Creates a Builder instance with the price data and initial AUM
    3. Implements the pairs trading strategy:
       - For each time step, randomly selects two assets
       - Goes long one asset and short the other with equal dollar amounts
       - Allocates 10% of the portfolio to this pair
    4. Applies transaction costs (1 bps of traded volume)
    5. Builds and returns the final portfolio


    Returns:
    -------
    tuple
        A tuple containing the built portfolio

    """
    logger.info("Load prices")
    logger.info("Build portfolio")
    b = Builder(prices=prices, initial_aum=1e6)

    for _t, state in b:
        assert state.nav > 0, "Game over"

        # pick two assets at random
        pair = rng.choice(state.assets, 2, replace=False)

        # compute the pair
        units = pd.Series(index=state.assets, data=0.0)
        units[pair] = [state.nav, -state.nav] / state.prices[pair].to_numpy()
        b.position = 0.1 * units

        # 1 bps of the traded volume (measured in USD) are paid as fee
        costs = 0.0001 * (state.trades.abs() * state.prices).sum()
        b.aum = state.aum - costs

    portfolio = b.build()
    return (portfolio,)


@app.cell
def _(portfolio):
    """Plot the portfolio's net asset value (NAV) over time.

    This cell visualizes the performance of the pairs trading strategy
    by plotting the portfolio's NAV over the simulation period.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio object built by the previous cell

    """
    portfolio.nav.plot()
    return


if __name__ == "__main__":
    app.run()
