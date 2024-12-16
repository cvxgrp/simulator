---
title: Oneassetfadingout
marimo-version: 0.10.2
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
_builder = Builder(prices=prices, initial_aum=2000)

for t, _state in _builder:
    _builder.weights = np.ones(len(_state.assets)) / len(_state.assets)
    _builder.aum = _state.aum

portfolio = _builder.build()
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