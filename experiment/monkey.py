import numpy as np
import pandas as pd

from cvx.simulator import FuturesBuilder, interpolate
from cvx.simulator.utils.grid import resample_index

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/prices_hashed.csv", index_col=0, parse_dates=True, header=0
    )

    prices = prices.apply(interpolate)

    grid = resample_index(prices.index, "W")

    builder = FuturesBuilder(prices=prices, aum=1e6)

    for t, state in builder:
        if t[-1] in grid:
            # rebalance
            weights = np.random.rand(len(state.assets))
            builder.cashposition = state.aum * weights / np.sum(weights)
        else:
            builder.position = state.position
