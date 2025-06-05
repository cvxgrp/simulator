"""global fixtures"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    """prices fixture"""
    frame = pd.read_csv(resource_dir / "price.csv", index_col=0, parse_dates=True, header=0).ffill()
    frame.index.name = "date"
    return frame


@pytest.fixture()
def prices_pl(resource_dir):
    """prices_pl fixture"""
    frame = pl.read_csv(resource_dir / "price.csv", try_parse_dates=True)
    frame = frame.with_columns(pl.col("date").cast(pl.Datetime("ns")))
    return frame.fill_null(strategy="forward")

    # return frame


@pytest.fixture()
def nav(resource_dir):
    return pd.read_csv(
        resource_dir / "nav.csv",
        index_col=0,
        parse_dates=True,
        header=0,
    ).squeeze()
