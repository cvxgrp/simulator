"""global fixtures"""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import pandas as pd
import pytest


def sharpe(x, n=252):
    return x.mean() * sqrt(n) / x.std()


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
def prices_hashed(resource_dir):
    """prices fixture"""
    return pd.read_csv(
        resource_dir / "prices_hashed.csv", index_col=0, parse_dates=True, header=0
    )
