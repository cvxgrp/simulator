"""global fixtures"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cvx.simulator import interpolate


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    """
    Fixture for the prices
    :param resource_dir: the resource directory (fixture)
    """
    prices = pd.read_csv(
        resource_dir / "prices_hashed.csv", parse_dates=True, index_col=0
    )
    return prices.apply(interpolate)
