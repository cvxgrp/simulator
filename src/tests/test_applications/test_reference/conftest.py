"""global fixtures"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


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
    return pd.read_csv(resource_dir / "prices.csv", parse_dates=True, index_col=0).tail(1000)


@pytest.fixture()
def spreads(resource_dir, prices):
    """
    Fixture for the spread
    :param resource_dir: the resource directory (fixture)
    """
    return pd.read_csv(resource_dir / "spreads.csv", parse_dates=True, index_col=0).loc[prices.index]
