#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""
Utilities for converting between returns and prices.

This module provides functions for converting return series to price series,
which is useful for backtesting strategies when only return data is available.
"""

import pandas as pd
import polars as pl


def _rescale(r: pd.Series) -> pd.Series:
    """
    Rescale a return series to create a price series starting at 1.

    This function converts a series of returns to a series of prices by:
    1. Dropping any NaN values
    2. Computing the cumulative product of (1 + return)
    3. Normalizing so the first value is 1

    Parameters
    ----------
    r : pd.Series
        Series of returns (as decimals, not percentages)

    Returns
    -------
    pd.Series
        Series of prices, starting at 1

    Examples
    --------
    >>> import pandas as pd
    >>> returns = pd.Series([0.05, 0.03, -0.02, 0.01])
    >>> _rescale(returns)
    0    1.000000
    1    1.050000
    2    1.081500
    3    1.059870
    4    1.070469
    dtype: float64
    """
    r = r.dropna()
    a = (r + 1).cumprod()
    return a / a.iloc[0]


def returns2prices(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame of returns to a DataFrame of prices.

    This function applies the _rescale function to each column of a DataFrame
    of returns, converting them to price series that all start at 1.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame where each column is a series of returns for an asset

    Returns
    -------
    pd.DataFrame
        DataFrame where each column is a series of prices for an asset,
        all starting at 1

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> returns = pd.DataFrame({
    ...     'A': [0.05, 0.03, -0.02, 0.01],
    ...     'B': [0.02, 0.01, 0.03, -0.01]
    ... })
    >>> returns2prices(returns)
           A         B
    0  1.000000  1.000000
    1  1.050000  1.020000
    2  1.081500  1.030200
    3  1.059870  1.061106
    4  1.070469  1.050495
    """
    prices = returns.apply(_rescale, axis=0)
    return prices


def _rescale_pl(r: pl.Series) -> pl.Series:
    """
    Rescale a polars Series of returns to create a price series starting at 1.

    This function converts a series of returns to a series of prices by:
    1. Dropping any null values
    2. Computing the cumulative product of (1 + return)
    3. Normalizing so the first value is 1

    Parameters
    ----------
    r : pl.Series
        Series of returns (as decimals, not percentages)

    Returns
    -------
    pl.Series
        Series of prices, starting at 1
    """
    r = r.drop_nulls()
    a = (r + 1).cum_prod()
    return a / a.first()


def returns2prices_pl(returns: pl.DataFrame) -> pl.DataFrame:
    """
    Convert a polars DataFrame of returns to a DataFrame of prices.

    This function applies the _rescale_pl function to each column of a DataFrame
    of returns, converting them to price series that all start at 1.

    Parameters
    ----------
    returns : pl.DataFrame
        DataFrame where each column is a series of returns for an asset

    Returns
    -------
    pl.DataFrame
        DataFrame where each column is a series of prices for an asset,
        all starting at 1

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.DataFrame({
    ...     'A': [0.05, 0.03, -0.02, 0.01],
    ...     'B': [0.02, 0.01, 0.03, -0.01]
    ... })
    >>> returns2prices_pl(returns)
           A         B
    0  1.000000  1.000000
    1  1.050000  1.020000
    2  1.081500  1.030200
    3  1.059870  1.061106
    4  1.070469  1.050495
    """
    return returns.select(
        [
            ((1 + pl.col(col_name)).cum_prod() / (1 + pl.col(col_name)).first()).alias(col_name)
            for col_name in returns.columns
        ]
    )
