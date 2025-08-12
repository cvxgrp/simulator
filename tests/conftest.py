"""global fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """Provide the path to the test resources directory.

    This fixture returns the path to the resources directory containing
    test data files. It has session scope, so it's created once per test session.

    Returns:
    -------
    Path
        Path to the resources directory

    """
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    """Provide a pandas DataFrame of price data for testing.

    This fixture loads price data from a CSV file in the resources directory,
    sets the first column as the index with datetime parsing, and forward-fills
    any missing values. The index is named 'date'.

    Parameters
    ----------
    resource_dir : Path
        Path to the resources directory (from the resource_dir fixture)

    Returns:
    -------
    pd.DataFrame
        DataFrame of price data with dates as index and assets as columns

    """
    frame = pd.read_csv(resource_dir / "price.csv", index_col=0, parse_dates=True, header=0).ffill()
    frame.index.name = "date"
    return frame


@pytest.fixture()
def prices_pl(resource_dir):
    """Provide a polars DataFrame of price data for testing.

    This fixture loads price data from a CSV file in the resources directory,
    attempts to parse dates, explicitly casts the 'date' column to a datetime type
    with nanosecond precision, and forward-fills any missing values.

    Parameters
    ----------
    resource_dir : Path
        Path to the resources directory (from the resource_dir fixture)

    Returns:
    -------
    pl.DataFrame
        Polars DataFrame of price data with a 'date' column and asset columns

    """
    frame = pl.read_csv(resource_dir / "price.csv", try_parse_dates=True)
    frame = frame.with_columns(pl.col("date").cast(pl.Datetime("ns")))
    return frame.fill_null(strategy="forward")


@pytest.fixture()
def nav(resource_dir):
    """Provide a pandas Series of NAV (Net Asset Value) data for testing.

    This fixture loads NAV data from a CSV file in the resources directory,
    sets the first column as the index with datetime parsing, and squeezes
    the result into a Series.

    Parameters
    ----------
    resource_dir : Path
        Path to the resources directory (from the resource_dir fixture)

    Returns:
    -------
    pd.Series
        Series of NAV values with dates as index

    """
    return pd.read_csv(
        resource_dir / "nav.csv",
        index_col=0,
        parse_dates=True,
        header=0,
    ).squeeze()


@pytest.fixture()
def readme_path() -> Path:
    """Provide the path to the project's README.md file.

    This fixture searches for the README.md file by starting in the current
    directory and moving up through parent directories until it finds the file.

    Returns:
    -------
    Path
        Path to the README.md file

    Raises:
    ------
    FileNotFoundError
        If the README.md file cannot be found in any parent directory

    """
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        candidate = current_dir / "README.md"
        if candidate.is_file():
            return candidate
        current_dir = current_dir.parent
    raise FileNotFoundError("README.md not found in any parent directory")
