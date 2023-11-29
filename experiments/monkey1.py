from io import StringIO

import pandas as pd
from loguru import logger

from cvx.simulator.builder import builder

if __name__ == "__main__":
    csvStringIO = StringIO(
        "date,A,B\n2022-01-01,,200.0\n2022-01-02,100.0,200.0\n2022-01-03,100.0,200.0\n2022-01-04,100.0,\n"
    )
    prices = pd.read_csv(csvStringIO, sep=",", parse_dates=True, index_col=0)
    print(prices)
    print(prices.index)

    b = builder(prices=prices, initial_cash=2000)

    for t, state in b:
        logger.info(t[-1])
        logger.info(f"Nav: {state.nav}")
        logger.info(f"Cash: {state.cash}")
        logger.info(f"Value: {state.value}")
        logger.info(f"Assets: {state.assets}")
        logger.info(f"Position: {state.position}")
        logger.info(100 * "*")
        # print(state.value)
        # print(state.position)
        # print("******************")
        b.set_weights(t[-1], pd.Series(index=state.assets, data=1 / len(state.assets)))
        logger.info(f"Nav: {state.nav}")
        logger.info(f"Cash: {state.cash}")
        logger.info(f"Value: {state.value}")
        logger.info(f"Assets: {state.assets}")
        logger.info(f"Position: {state.position}")
        logger.info(200 * "-")
