"""global fixtures"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from jquantstats.api import build_data


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    """prices fixture"""
    return pd.read_csv(resource_dir / "price.csv", index_col=0, parse_dates=True, header=0).ffill()


@pytest.fixture()
def nav():
    return pd.read_csv(
        Path(__file__).parent / "resources" / "nav.csv",
        index_col=0,
        parse_dates=True,
        header=0,
    ).squeeze()


@pytest.fixture()
def nav_pl(resource_dir):
    return pl.read_csv(resource_dir / "nav_returns.csv", try_parse_dates=True)


@pytest.fixture()
def data(nav_pl):
    """data fixture"""
    return build_data(returns=nav_pl)


# if __name__ == '__main__':
#    x = pd.read_csv(Path(__file__).parent / "resources/nav.csv", index_col=0, parse_dates=True).pct_change()
#    x.to_csv(Path(__file__).parent / "resources" / "nav_returns.csv")
#    print(x)
