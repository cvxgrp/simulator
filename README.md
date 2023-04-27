# simulator
Tool to support backtests


Given a universe of $m$ assets we are given prices for each of them at time $t_1, t_2, \ldots t_n$, 
e.g. we operate using $n \times m$ matrix where each column corresponds to a particular asset.

In a backtest we iterate in time (e.g. row by row) through the matrix and allocate positions to all or some of the assets.
This tool shall help to simplify the accounting. It keeps track of the available cash, the profits achieved, etc.

## Modus operandi

The simulator shall be completely agnostic as to the trading policy/strategy.
Our approach follows a rather common pattern:

* Create the portfolio object
* Loop through time
* Analyse results

We demonstrate those steps with somewhat silly policies. They are never good strategies, but are always valid ones.

### Create the portfolio object

The users defines a portfolio object by loading a frame of prices and initialize the initial amount of cash used in our experiment:
u
```python
import pandas as pd
from cvx.simulator.portfolio import build_portfolio

prices = pd.read_csv(Path("resources") / "price.csv", index_col=0, parse_dates=True, header=0).ffill(
portfolio = build_portfolio(prices=prices, initial_cash=1e6)
```

It is also possible to specify a model for trading costs.

### Loop through time

We have overloaded the '__iter__' method to create a custom loop. 
Let's start with a first strategy. Each day we choose two names from the universe at random.
Buy one (say 0.1 of your portfolio wealth) and short one the same amount.

```python
for before, now, snapshot in portfolio:
    # pick two assets at random
    pair = np.random.choice(portfolio.assets, 2, replace=False)
    # compute the pair
    stocks = pd.Series(index=portfolio.assets, data=0.0)
    stocks[pair] = [snapshot.nav, -snapshot.nav] / snapshot.prices[pair].values
    # update the position 
    portfolio[now] = 0.1 * stocks
```

A lot of magic is hidden in the snapshot variable. 
The snapshot gives access to the currently available cash, the current prices and the current valuation of all holdings.

Here's a slightly more realistic loop. Given a set of $4$ assets we want to implmenent the popular $1/n$ strategy.

```python
for _, now, snapshot in portfolio:
    # each day we invest a quarter of the capital in the assets
    portfolio[now] = 0.25 * snapshot.nav / snapshot.prices
```

Note that we update the position at time 'now' using a series of actual stocks rather than weights or cashpositions.
Future versions of this package may support such conventions, too.

### Analyse results

The loop above is filling up the desired positions. The portfolio object is now ready for further analysis.
It is possible dive into the data, e.g.

```python
portfolio.nav
portfolio.cash
portfolio.equity
...
``` 



Future versions of this package may support such conventions, too.


