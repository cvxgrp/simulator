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
Interpolation utilities for time series data.

This module provides functions for interpolating missing values in time series
and validating that time series don't have missing values in the middle.
"""

import narwhals as nw


def interpolate(ts: nw.Series) -> nw.Series:
    """
    Interpolate missing values in a time series between the first and last valid indices.

    This function fills forward (ffill) missing values in a time series, but only
    between the first and last valid indices. Values outside this range remain NaN/null.

    Parameters
    ----------
    ts : nw.Series
        The time series to interpolate

    Returns
    -------
    nw.Series
        The interpolated time series

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import narwhals as nw
    >>> ts = nw.Series(pd.Series([1, np.nan, np.nan, 4, 5]))
    >>> interpolate(ts)
    0    1.0
    1    1.0
    2    1.0
    3    4.0
    4    5.0
    dtype: float64
    """
    first = ts.first_valid_index()
    last = ts.last_valid_index()

    if first is not None and last is not None:
        ts_slice = ts.loc[first:last]
        ts_slice = ts_slice.ffill()
        result = ts.copy()
        result.loc[first:last] = ts_slice
        return result
    return ts


def valid(ts: nw.Series) -> bool:
    """
    Check if a time series has no missing values between the first and last valid indices.

    This function verifies that a time series doesn't have any NaN/null values in the middle.
    It's acceptable to have NaNs/nulls at the beginning or end of the series.

    Parameters
    ----------
    ts : nw.Series
        The time series to check

    Returns
    -------
    bool
        True if the time series has no missing values between the first and last valid indices,
        False otherwise

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import narwhals as nw
    >>> ts1 = nw.Series(pd.Series([np.nan, 1, 2, 3, np.nan]))  # NaNs only at beginning and end
    >>> valid(ts1)
    True
    >>> ts2 = nw.Series(pd.Series([1, 2, np.nan, 4, 5]))  # NaN in the middle
    >>> valid(ts2)
    False
    """
    # Check if the series with NaNs dropped has the same indices as the interpolated series with NaNs dropped
    # If they're the same, there are no NaNs in the middle of the series
    return ts.dropna().index.equals(interpolate(ts).dropna().index)
