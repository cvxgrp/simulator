---
title: Oneassetfadingout
marimo-version: 0.9.27
---

# One asset fading out

```{.python.marimo}
import numpy as np
import pandas as pd

from cvx.simulator import Builder
```

```{.python.marimo}
# two assets, A and B, constant price for A=100 and B=200
prices = pd.read_csv("data/prices.csv", header=0, index_col=0, parse_dates=True)
prices.loc["2022-01-03", "B"] = np.NaN
prices.loc["2022-01-04", "B"] = np.NaN
prices
```

## Iterate

```{.python.marimo}
b = Builder(prices=prices, initial_aum=2000)

for t, state in b:
    b.weights = np.ones(len(state.assets)) / len(state.assets)
    b.aum = state.aum
```

```{.python.marimo}
b.units
```

```{.python.marimo}
b.prices
```

```{.python.marimo}
portfolio = b.build()
```

```{.python.marimo}
portfolio.prices
```

```{.python.marimo}
portfolio.nav
```

```{.python.marimo}
portfolio.weights
```

```{.python.marimo}
import marimo as mo
```