---
title: Oneassetfadingout
marimo-version: 0.9.27
---

# One asset fading out

```{.python.marimo}
from pathlib import Path

import numpy as np
import pandas as pd

from cvx.simulator import Builder

folder = Path(__file__).parent
```

```{.python.marimo}
# two assets, A and B, constant price for A=100 and B=200
prices = pd.read_csv(
    folder / "data" / "prices.csv", header=0, index_col=0, parse_dates=True
)
prices.loc["2022-01-03", "B"] = np.nan
prices.loc["2022-01-04", "B"] = np.nan
prices
```

## Iterate

```{.python.marimo}
builder = Builder(prices=prices, initial_aum=2000)

for t, state in builder:
    builder.weights = np.ones(len(state.assets)) / len(state.assets)
    builder.aum = state.aum

portfolio = builder.build()
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