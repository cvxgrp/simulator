# The simulator should always be completely agnostic as to the trading policy.
# You should even demonstrate this with silly policies.
# Like here’s one:  Each day choose names from the universe at random.
# Buy one (say 0.1 of your portfolio wealth) and short one the same amount.
# Not a good strategy, but a valid one.
# Of course the simulate will terminate if you go bust (which seems likely).
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from cvx.simulator.portfolio import build_portfolio
from cvx.simulator.trading_costs import LinearCostModel

pd.options.plotting.backend = "plotly"

if __name__ == '__main__':
    logger.info("Load prices")
    prices = pd.read_csv(Path("data") / "price.csv", index_col=0, parse_dates=True, header=0).ffill()

    # Let's pay a fee of 10 bps every time
    logger.info("Build trading cost model")
    linearCostModel = LinearCostModel(name="LinearCostModel cost model", factor=0.0010)

    logger.info("Build portfolio")
    portfolio = build_portfolio(prices=prices, initial_cash=1e6, trading_cost_model=linearCostModel)

    for before, now, snapshot in portfolio:
        assert snapshot.nav > 0, "Game over"
        logger.info(f"Nav: {snapshot.nav}")
        logger.info(f"Cash: {snapshot.cash}")

        # pick two assets at random
        pair = np.random.choice(portfolio.assets, 2, replace=False)
        # compute the pair
        stocks = pd.Series(index=portfolio.assets, data=0.0)
        stocks[pair] = [snapshot.nav, -snapshot.nav] / snapshot.prices[pair].values

        portfolio[now] = 0.1*stocks

    fig = portfolio.nav.plot()
    fig.show()

