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
    """prices fixture"""
    return pd.read_csv(
        resource_dir / "price.csv", index_col=0, parse_dates=True, header=0
    ).ffill()


@pytest.fixture()
def nav():
    return pd.read_csv(
        Path(__file__).parent / "resources" / "nav.csv",
        index_col=0,
        parse_dates=True,
        header=0,
    ).squeeze()
