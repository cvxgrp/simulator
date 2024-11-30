---
title: Monkey
marimo-version: 0.9.27
---

# Monkey portfolios

```{.python.marimo}
import numpy as np
import pandas as pd
```

```{.python.marimo}
from cvx.simulator import Builder

pd.options.plotting.backend = "plotly"
```

```{.python.marimo}
prices = pd.read_csv(
    "data/stock-prices.csv", header=0, index_col=0, parse_dates=True
)
```

```{.python.marimo}
b = Builder(prices=prices, initial_aum=1000000.0)
np.random.seed(42)
for _time, _state in b:
    _n = len(_state.assets)
    _w = np.random.rand(n)
    _w = w / np.sum(w)
    assert np.all(_w >= 0)
    assert np.allclose(np.sum(_w), 1)
    b.weights = w
    b.aum = _state.aum
```

```{.python.marimo}
_portfolio = b.build()
_portfolio.nav.plot()
```

```{.python.marimo}
b_1 = Builder(prices=prices, initial_aum=1000000.0)
np.random.seed(42)
for _time, _state in b_1:
    _n = len(state.assets)
    _w = np.random.rand(n)
    _w = w / np.sum(w)
    assert np.all(_w >= 0)
    assert np.allclose(np.sum(_w), 1)
    b_1.weights = w
    b_1.aum = state.aum
```

```{.python.marimo}
_portfolio = b_1.build()
_portfolio.snapshot(aggregate=True)
```

```{.python.marimo}
import marimo as mo
```