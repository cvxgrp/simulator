---
title: Monkey
marimo-version: 0.9.27
---

# Monkey portfolios

```{.python.marimo}
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(__file__).parent
```

```{.python.marimo}
from cvx.simulator import Builder

pd.options.plotting.backend = "plotly"
```

```{.python.marimo}
prices = pd.read_csv(
    folder / "data" / "stock-prices.csv", header=0, index_col=0, parse_dates=True
)
```

```{.python.marimo}
_builder = Builder(prices=prices, initial_aum=1000000.0)
np.random.seed(42)
for _time, _state in _builder:
    _n = len(_state.assets)
    _w = np.random.rand(_n)
    _w = _w / np.sum(_w)
    assert np.all(_w >= 0)
    assert np.allclose(np.sum(_w), 1)
    _builder.weights = _w
    _builder.aum = _state.aum

_portfolio = _builder.build()
_portfolio.snapshot(aggregate=True)
```

```{.python.marimo}
_builder = Builder(prices=prices, initial_aum=1000000.0)
np.random.seed(42)
for _time, _state in _builder:
    _n = len(_state.assets)
    _w = np.random.rand(_n)
    _w = _w / np.sum(_w)
    assert np.all(_w >= 0)
    assert np.allclose(np.sum(_w), 1)
    _builder.weights = _w
    _builder.aum = _state.aum

_portfolio = _builder.build()
_portfolio.snapshot(aggregate=True)
```

```{.python.marimo}
import marimo as mo
```