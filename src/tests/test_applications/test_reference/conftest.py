"""Global fixtures for the test_reference module.

This module contains fixtures that are used by multiple test files in the
test_reference module. It provides access to test resources such as price
data and spread data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """Create a fixture for the resource directory.

    This fixture provides the path to the resources directory, which contains
    test data files such as prices.csv and spreads.csv.

    Returns:
    -------
    Path
        Path to the resources directory

    """
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    """Create a fixture for price data.

    This fixture loads price data from a CSV file in the resources directory.
    It returns the last 1000 rows of the price data.

    Parameters
    ----------
    resource_dir : Path
        Path to the resources directory

    Returns:
    -------
    pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    """
    return pd.read_csv(resource_dir / "prices.csv", parse_dates=True, index_col=0).tail(1000)


@pytest.fixture()
def spreads(resource_dir, prices):
    """Fixture for the spread.

    :param resource_dir: the resource directory (fixture).
    """
    return pd.read_csv(resource_dir / "spreads.csv", parse_dates=True, index_col=0).loc[prices.index]
