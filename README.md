# simulator
Tool to support backtests


Given a universe of $m$ assets we are given prices for each of them at time $t_1, t_2, \ldots t_n$, 
e.g. we operate using $n \times m$ matrix where each column corresponds to a particular asset.

In a backtest we iterate in time (e.g. row by row) through the matrix and allocate positions to all or some of the assets.
This tool shall help to simplify the accounting. It keeps track of the available cash, the profits achieved, etc.

Our approach follows an object-oriented pattern. The users defines a portfolio object and by looping through this object (we have overloaded '__iter__') 
we can update the target position at each step.

We create the portfolio object by loading a frame of prices and initialize the initial amount of cash used in our experiment:

```python
    import pandas as pd
    from cvx.simulator.portfolio import build_portfolio
    
    prices = pd.read_csv(Path("resources") / "price.csv", index_col=0, parse_dates=True, header=0).ffill(
    portfolio = build_portfolio(prices=prices, initial_cash=1e6)
```
