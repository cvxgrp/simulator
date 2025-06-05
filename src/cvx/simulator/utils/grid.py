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
Grid resampling utilities for time series data.

This module provides functions for resampling time series data to coarser time grids,
which is useful for simulating periodic rebalancing of portfolios (e.g., monthly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def iron_frame(frame: pd.DataFrame | pl.DataFrame, rule: str) -> pd.DataFrame | pl.DataFrame:
    """
    Resample a DataFrame to keep values constant on a coarser time grid.

    This function takes a DataFrame with a datetime index and creates a new
    DataFrame where values change only at specified intervals (e.g., monthly) and
    remain constant between those intervals.

    Parameters
    ----------
    frame : Union[pd.DataFrame, pl.DataFrame]
        The DataFrame to be resampled, must have a datetime index
    rule : str
        The frequency string for resampling (e.g., 'M' for month-end,
        'MS' for month-start, 'W' for week)

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        A new DataFrame with the same index as the input, but with values
        changing only at the specified frequency

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime
    >>> dates = pd.date_range('2023-01-01', '2023-01-10')
    >>> df = pd.DataFrame({'value': range(10)}, index=dates)
    >>> iron_frame(df, 'W')  # Weekly resampling
                value
    2023-01-01    0.0
    2023-01-02    0.0
    2023-01-03    0.0
    2023-01-04    0.0
    2023-01-05    0.0
    2023-01-06    0.0
    2023-01-07    0.0
    2023-01-08    7.0
    2023-01-09    7.0
    2023-01-10    7.0
    """
    # Handle pandas DataFrame
    if isinstance(frame, pd.DataFrame):
        s_index = resample_index(pd.DatetimeIndex(frame.index), rule)
        return _project_frame_to_grid(frame, s_index)

    # Handle polars DataFrame
    elif isinstance(frame, pl.DataFrame):
        # Check if the DataFrame has a datetime column that can be used as an index
        datetime_cols = [col for col in frame.columns if frame[col].dtype in (pl.Date, pl.Datetime)]

        if not datetime_cols:
            raise ValueError("Polars DataFrame must have at least one datetime column to use as index")

        # Use the first datetime column as the index
        index_col = datetime_cols[0]

        # Convert to pandas with the datetime column as index
        pandas_frame = frame.to_pandas()
        pandas_frame = pandas_frame.set_index(index_col)

        # Resample and project
        s_index = resample_index(pd.DatetimeIndex(pandas_frame.index), rule)
        result_pandas = _project_frame_to_grid(pandas_frame, s_index)

        # Convert back to polars, preserving the index as a column
        result_polars = pl.from_pandas(result_pandas.reset_index())

        return result_polars

    else:
        raise TypeError(f"Expected pd.DataFrame or pl.DataFrame, got {type(frame)}")


def resample_index(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """
    Resample a DatetimeIndex to a lower frequency using a specified rule.

    This function creates a new DatetimeIndex with dates at the specified
    frequency (e.g., month-end, week-end).

    Parameters
    ----------
    index : pd.DatetimeIndex
        The original datetime index to resample
    rule : str
        The pandas frequency string for resampling (e.g., 'M' for month-end,
        'MS' for month-start, 'W' for week)

    Returns
    -------
    pd.DatetimeIndex
        A new DatetimeIndex with dates at the specified frequency

    Notes
    -----
    This function does not modify the input index object but returns a new one.
    For polars DataFrames, convert the index to a pandas DatetimeIndex first.

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2023-01-01', '2023-03-31')
    >>> resample_index(dates, 'M')  # Month-end resampling
    DatetimeIndex(['2023-01-31', '2023-02-28', '2023-03-31'], dtype='datetime64[ns]', freq=None)
    """
    series = pd.Series(index=index, data=index)
    a = series.resample(rule=rule).first()
    return pd.DatetimeIndex(a.values)


def _project_frame_to_grid(frame: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Project a DataFrame to a coarser grid while maintaining the original index.

    This function  a new DataFrame that takes values from the original
    frame only at the dcreatesates specified in the grid, and forward-fills these values
    until the next grid date.

    Parameters
    ----------
    frame : pd.DataFrame
        The DataFrame to project (must have a datetime index)
    grid : pd.DatetimeIndex
        The coarser grid of dates to use for resampling

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the same index as the input, but with values
        changing only at the grid dates

    Notes
    -----
    This is useful for simulating periodic rebalancing of portfolios. For example,
    with monthly rebalancing, positions would change only on specific dates (e.g.,
    month-end) and remain constant for the rest of the month.

    This is a private helper function that works with pandas DataFrames only.
    For polars DataFrames, use iron_frame which handles the conversion.
    """
    sample = np.nan * frame
    for t in grid:
        sample.loc[t] = frame.loc[t]
    # sample.loc[grid] = frame.loc[grid]
    return sample.ffill()
