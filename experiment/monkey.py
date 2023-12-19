import numpy as np
import pandas as pd

from cvx.simulator import FuturesBuilder, interpolate
from cvx.simulator.utils.grid import resample_index

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/prices_hashed.csv", index_col=0, parse_dates=True, header=0
    ).dropna(axis=0, how="all")

    # the prices used by the simulator can have no internal NaNs
    prices = prices.apply(interpolate)

    # move to a coarser grid
    grid = resample_index(prices.index, "W")

    # construct the portfolio using a Builder
    builder = FuturesBuilder(prices=prices, aum=1e6)

    # iterate
    for t, state in builder:
        # t[-1] is in the grid ==> rebalance
        if t[-1] in grid:
            # the builder is using the latest greatest and correct prices
            # Those weights are correctly drifted
            state.weights
            state.nav

            # let's use a monkey to compute some new weights
            weights = np.random.rand(len(state.assets))
            weights = weights / np.sum(weights)
            assert np.isclose(np.sum(weights), 1.0), f"{np.sum(weights), state.assets}"

            builder.weights = weights / np.sum(weights)
        else:
            # on days that are not in the grid we just forward the position
            builder.position = state.position

    portfolio = builder.build()
    portfolio.snapshot()
