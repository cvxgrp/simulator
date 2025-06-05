import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(mo):
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
    import marimo as mo
    import numpy as np
    import pandas as pd
    from loguru import logger

    from cvx.simulator import Builder

    return Builder, logger, mo, np, pd


@app.cell
def _(Builder, logger, mo, np, pd):
    logger.info("Load prices")
    prices = pd.read_csv(
        str(mo.notebook_location() / "data" / "stock-prices.csv"), index_col=0, parse_dates=True, header=0
    )

    logger.info("Build portfolio")
    b = Builder(prices=prices, initial_aum=1e6)

    for t, state in b:
        assert state.nav > 0, "Game over"

        # pick two assets at random
        pair = np.random.choice(state.assets, 2, replace=False)

        # compute the pair
        units = pd.Series(index=state.assets, data=0.0)
        units[pair] = [state.nav, -state.nav] / state.prices[pair].values
        b.position = 0.1 * units

        # 1 bps of the traded volume (measured in USD) are paid as fee
        costs = 0.0001 * (state.trades.abs() * state.prices).sum()
        b.aum = state.aum - costs

    portfolio = b.build()
    return (portfolio,)


@app.cell
def _(portfolio):
    portfolio.nav.plot()
    return


if __name__ == "__main__":
    app.run()
