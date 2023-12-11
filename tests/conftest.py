"""global fixtures"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cvx.simulator.builder import Builder


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    """prices fixture"""
    return pd.read_csv(
        resource_dir / "price.csv", index_col=0, parse_dates=True, header=0
    ).ffill()


@pytest.fixture()
def portfolio(prices):
    """portfolio fixture"""
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    b = Builder(prices, initial_cash=1e6)

    for t, state in b:
        b.position = positions.loc[t[-1]]

    return b.build()
