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
"""Interpolation utilities for time series data.

This module provides functions for interpolating missing values in time series
and validating that time series don't have missing values in the middle.
"""

import pandas as pd
import polars as pl


def interpolate(ts):
    """Interpolate missing values in a time series between the first and last valid indices.

    This function fills forward (ffill) missing values in a time series, but only
    between the first and last valid indices. Values outside this range remain NaN/null.

    Parameters
    ----------
    ts : pd.Series or pl.Series
        The time series to interpolate

    Returns:
    -------
    pd.Series or pl.Series
        The interpolated time series

    Examples:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> ts = pd.Series([1, np.nan, np.nan, 4, 5])
    >>> interpolate(ts)
    0    1.0
    1    1.0
    2    1.0
    3    4.0
    4    5.0
    dtype: float64

    """
    # Check if the input is a valid type
    if not isinstance(ts, pd.Series | pl.Series):
        raise TypeError(f"Expected pd.Series or pl.Series, got {type(ts)}")

    # If the input is a polars Series, use the polars-specific function
    if isinstance(ts, pl.Series):
        return interpolate_pl(ts)
    first = ts.first_valid_index()
    last = ts.last_valid_index()

    if first is not None and last is not None:
        ts_slice = ts.loc[first:last]
        ts_slice = ts_slice.ffill()
        result = ts.copy()
        result.loc[first:last] = ts_slice
        return result
    return ts


def valid(ts) -> bool:
    """Check if a time series has no missing values between the first and last valid indices.

    This function verifies that a time series doesn't have any NaN/null values in the middle.
    It's acceptable to have NaNs/nulls at the beginning or end of the series.

    Parameters
    ----------
    ts : pd.Series or pl.Series
        The time series to check

    Returns:
    -------
    bool
        True if the time series has no missing values between the first and last valid indices,
        False otherwise

    Examples:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> ts1 = pd.Series([np.nan, 1, 2, 3, np.nan])  # NaNs only at beginning and end
    >>> valid(ts1)
    True
    >>> ts2 = pd.Series([1, 2, np.nan, 4, 5])  # NaN in the middle
    >>> valid(ts2)
    False

    """
    # Check if the input is a valid type
    if not isinstance(ts, pd.Series | pl.Series):
        raise TypeError(f"Expected pd.Series or pl.Series, got {type(ts)}")

    # If the input is a polars Series, use the polars-specific function
    if isinstance(ts, pl.Series):
        return valid_pl(ts)
    # Check if the series with NaNs dropped has the same indices as the interpolated series with NaNs dropped
    # If they're the same, there are no NaNs in the middle of the series
    return ts.dropna().index.equals(interpolate(ts).dropna().index)


def interpolate_pl(ts: pl.Series) -> pl.Series:
    """Interpolate missing values in a polars time series between the first and last valid indices.

    This function fills forward (ffill) missing values in a time series, but only
    between the first and last valid indices. Values outside this range remain null.

    Parameters
    ----------
    ts : pl.Series
        The time series to interpolate

    Returns:
    -------
    pl.Series
        The interpolated time series

    Examples:
    --------
    >>> import polars as pl
    >>> ts = pl.Series([1, None, None, 4, 5])
    >>> interpolate_pl(ts)
    shape: (5,)
    Series: '' [i64]
    [
        1
        1
        1
        4
        5
    ]

    """
    # Find first and last valid indices
    non_null_indices = ts.is_not_null().arg_true()

    if len(non_null_indices) == 0:
        return ts

    first = non_null_indices[0]
    last = non_null_indices[-1]

    # Create a new series with the same length as the original
    values = ts.to_list()

    # Fill forward within the slice between first and last valid indices
    current_value = None
    for i in range(first, last + 1):
        if values[i] is not None:
            current_value = values[i]
        elif current_value is not None:
            values[i] = current_value

    # Create a new series with the filled values
    return pl.Series(values, dtype=ts.dtype)


def valid_pl(ts: pl.Series) -> bool:
    """Check if a polars time series has no missing values between the first and last valid indices.

    This function verifies that a time series doesn't have any null values in the middle.
    It's acceptable to have nulls at the beginning or end of the series.

    Parameters
    ----------
    ts : pl.Series
        The time series to check

    Returns:
    -------
    bool
        True if the time series has no missing values between the first and last valid indices,
        False otherwise

    Examples:
    --------
    >>> import polars as pl
    >>> ts1 = pl.Series([None, 1, 2, 3, None])  # Nulls only at beginning and end
    >>> valid_pl(ts1)
    True
    >>> ts2 = pl.Series([1, 2, None, 4, 5])  # Null in the middle
    >>> valid_pl(ts2)
    False

    """
    # Get indices of non-null values
    non_null_indices = ts.is_not_null().arg_true()

    if len(non_null_indices) <= 1:
        return True

    # Check if the range of indices is continuous
    first = non_null_indices[0]
    last = non_null_indices[-1]
    expected_count = last - first + 1

    # If all values between first and last valid indices are non-null,
    # then the count of non-null values should equal the range size
    return len([i for i in non_null_indices if first <= i <= last]) == expected_count


def interpolate_df_pl(df: pl.DataFrame) -> pl.DataFrame:
    """Interpolate missing values in a polars DataFrame between the first and last valid indices for each column.

    This function applies interpolate_pl to each column of a DataFrame,
    filling forward (ffill) missing values in each column, but only
    between the first and last valid indices. Values outside this range remain null.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to interpolate

    Returns:
    -------
    pl.DataFrame
        The interpolated DataFrame

    Examples:
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     'A': [1.0, None, None, 4.0, 5.0],
    ...     'B': [None, 2.0, None, 4.0, None]
    ... })
    >>> interpolate_df_pl(df)
    shape: (5, 2)
    ┌─────┬──────┐
    │ A   ┆ B    │
    │ --- ┆ ---  │
    │ f64 ┆ f64  │
    ╞═════╪══════╡
    │ 1.0 ┆ null │
    │ 1.0 ┆ 2.0  │
    │ 1.0 ┆ 2.0  │
    │ 4.0 ┆ 4.0  │
    │ 5.0 ┆ null │
    └─────┴──────┘

    """
    # Apply interpolate_pl to each column
    result = {}
    for col in df.columns:
        result[col] = interpolate_pl(df[col])

    return pl.DataFrame(result)


def valid_df_pl(df: pl.DataFrame) -> bool:
    """Check if a polars DataFrame has no missing values between the first and last valid indices for each column.

    This function verifies that each column in the DataFrame doesn't have any null values in the middle.
    It's acceptable to have nulls at the beginning or end of each column.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to check

    Returns:
    -------
    bool
        True if all columns in the DataFrame have no missing values between their first and last valid indices,
        False otherwise

    Examples:
    --------
    >>> import polars as pl
    >>> df1 = pl.DataFrame({
    ...     'A': [None, 1, 2, 3, None],  # Nulls only at beginning and end
    ...     'B': [None, 2, 3, 4, None]   # Nulls only at beginning and end
    ... })
    >>> valid_df_pl(df1)
    True
    >>> df2 = pl.DataFrame({
    ...     'A': [1, 2, None, 4, 5],     # Null in the middle
    ...     'B': [1, 2, 3, 4, 5]         # No nulls
    ... })
    >>> valid_df_pl(df2)
    False

    """
    # Check each column
    for col in df.columns:
        if not valid_pl(df[col]):
            return False

    return True
