from cvx.simulator import Builder, Portfolio


def shift(portfolio: Portfolio, periods: int = 0):
    """Shift the portfolio by a number of periods"""
    # shift the weights
    weights = portfolio.weights.shift(periods)
    builder = Builder.from_weights(
        weights=weights, prices=portfolio.prices, initial_aum=portfolio.aum.iloc[0]
    )
    return builder.build()
