# [cvxsimulator](https://www.cvxgrp.org/simulator/book)

[![PyPI version](https://badge.fury.io/py/cvxsimulator.svg)](https://badge.fury.io/py/cvxsimulator)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/simulator/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxsimulator?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxsimulator)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/simulator/badge.png?branch=main)](https://coveralls.io/github/cvxgrp/simulator?branch=main)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/cvxgrp/simulator)

Given a universe of $m$ assets we are given prices for each of them at
time $t_1, t_2, \ldots t_n$, e.g. we operate using an $n \times m$ matrix where
each column corresponds to a particular asset.

In a backtest we iterate in time (e.g. row by row) through the matrix and
allocate positions to all or some of the assets. This tool shall help to
simplify the accounting. It keeps track of the available cash,
the profits achieved, etc.

## Creating portfolios

The simulator shall be completely agnostic as to the trading policy/strategy.
Our approach follows a rather common pattern:

* [Create the builder object](#create-the-builder-object)
* [Loop through time](#loop-through-time)
* [Build the portfolio](#build-the-portfolio)

We demonstrate those steps with somewhat silly policies.
They are never good strategies, but are always valid ones.

### Create the builder object

The user defines a builder object by loading prices
and initialize the amount of cash used in an experiment:

```python
import pandas as pd
from cvx.simulator import Builder

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True, header=0)
b = Builder(prices=prices, initial_cash=1e6)
```

Prices have to be valid, there may be NaNs only at the beginning and the end of
each column in frame.
There can be no NaNs hiding in the middle of any time series.

It is also possible to specify a model for trading costs.
The builder helps to fill up the frame of positions. Only once done
we construct the actual portfolio.

### Loop through time

We have overloaded the `__iter__` and `__setitem__` methods to create a custom loop.
Let's start with a first strategy. Each day we choose two names from the
universe at random.
Buy one (say 0.1 of your portfolio wealth) and short one the same amount.

```python
for t, state in b:
    # pick two assets at random
    pair = np.random.choice(state.assets, 2, replace=False)
    # compute the pair
    units = pd.Series(index=state.assets, data=0.0)
    units[pair] = [state.nav, -state.nav] / state.prices[pair].values
    # update the position
    b.position = 0.1 * units
    # Do not apply trading costs
    b.aum = state.aum
```

Here t is the growing list of timestamps, e.g. in the first iteration
t is $t1$, in the second iteration it will be $t1, t2$ etc.

A lot of magic is hidden in the state variable.
The state gives access to the currently available cash, the current prices
and the current valuation of all holdings.

Here's a slightly more realistic loop. Given a set of $4$ assets we want to
implemenent the popular $1/n$ strategy.

```python
for t, state in b:
    # each day we invest a quarter of the capital in the assets
    b.position = 0.25 * state.nav / state.prices
    b.aum = state.aum
```

Note that we update the position at the last element in the t list
using a series of actual units rather than weights or cashpositions.
The builder class also exposes setters for such alternative conventions.

```python
for t, state in b:
    # each day we invest a quarter of the capital in the assets
    b.weights = np.ones(4)*0.25
    b.aum = state.aum
```

### Build the portfolio

Once finished it is possible to build the portfolio object

```python
portfolio = b.build()
```

## Analytics

The portfolio object supports further analysis and exposes
a number of properties, e.g.

```python
portfolio.nav
portfolio.cash
portfolio.equity
```

We have also integrated the [quantstats](https://github.com/ranaroussi/quantstats)
package for further analysis. Hence it is possible to perform

```python
portfolio.snapshot()
portfolio.metrics()
portfolio.plots()
portfolio.html()
```

We also added an enum

```python
portfolio.plot(kind=Plot.DRAWDOWN)
```

supporting all plots defined in quantstats.

![quantstats](https://raw.githubusercontent.com/cvxgrp/simulator/main/portfolio.png)

## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in pyproject.toml.

## Jupyter

We install [JupyterLab](https://jupyter.org) on fly within the aforementioned
virtual environment. Executing

```bash
make jupyter
```

will install and start the jupyter lab.
