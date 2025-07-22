"""Global fixtures for the test_talk module.

This module contains fixtures that are used by multiple test files in the
test_talk module. It provides access to test resources such as price data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cvxsimulator import interpolate


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """Create a fixture for the resource directory.

    This fixture provides the path to the resources directory, which contains
    test data files such as prices_hashed.csv.

    Returns:
    -------
    Path
        Path to the resources directory

    """
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    """Create a fixture for price data.

    This fixture loads price data from a CSV file in the resources directory
    and applies interpolation to fill any missing values.

    Parameters
    ----------
    resource_dir : Path
        Path to the resources directory

    Returns:
    -------
    pd.DataFrame
        DataFrame of interpolated asset prices with dates as index and assets as columns

    """
    prices = pd.read_csv(resource_dir / "prices_hashed.csv", parse_dates=True, index_col=0)
    return prices.apply(interpolate)
