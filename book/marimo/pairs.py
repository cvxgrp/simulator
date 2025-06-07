"""pairs trading."""

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
async def _():
    """Import required libraries and modules.

    This cell imports the necessary libraries and modules for the pairs trading simulation:
    - marimo: For notebook functionality
    - numpy: For numerical operations
    - pandas: For data manipulation
    - loguru: For logging
    - Builder: From cvx.simulator for portfolio simulation

    Returns
    -------
    tuple
        A tuple containing the imported modules (Builder, logger, mo, np, pd)

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

    from loguru import logger

    from cvx.simulator import Builder

    return Builder, logger, mo, np, pd, pl


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
def _(Builder, logger, np, pd, prices):
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

    Parameters
    ----------
    Builder : class
        The Builder class from cvx.simulator
    logger : Logger
        The logger instance for logging information
    np : module
        The numpy module
    pd : module
        The pandas module
    prices: pd.DataFrame
        The prices

    Returns
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
        pair = np.random.choice(state.assets, 2, replace=False)

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
