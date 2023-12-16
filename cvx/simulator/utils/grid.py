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
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def iron_frame(frame: pd.DataFrame, rule: Any) -> pd.DataFrame:
    """
    The iron_frame function takes a pandas DataFrame
    and keeps it constant on a coarser grid.

    :param frame: The frame to be ironed
    :param rule: The rule to be used for the construction of the grid
    :return: the ironed frame
    """
    s_index = resample_index(pd.DatetimeIndex(frame.index), rule)
    return _project_frame_to_grid(frame, s_index)


def resample_index(index: pd.DatetimeIndex, rule: Any) -> pd.DatetimeIndex:
    """
    The resample_index function resamples a pandas DatetimeIndex object
    to a lower frequency using a specified rule.


    Note that the function does not modify the input index object,
    but rather returns a pandas DatetimeIndex
    """
    series = pd.Series(index=index, data=index)
    a = series.resample(rule=rule).first()
    return pd.DatetimeIndex(a.values)


def _project_frame_to_grid(frame: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """
    The project_frame_to_grid function projects a pandas DataFrame
    to a coarser grid while still sharing the same index.
    It does that by taking over values of the frame from the coarser
    grid that are then forward filled.
    An application would be monthly rebalancing of a portfolio.
    E.g. on days in a particular grid we adjust the position and keep
    it constant for the rest of the month.

    :param frame: the frame (existing on a finer grid)
    :param grid: the coarse grid
    :return: a frame changing only values on days in the grid
    """
    sample = np.NaN * frame
    for t in grid:
        sample.loc[t] = frame.loc[t]
    # sample.loc[grid] = frame.loc[grid]
    return sample.ffill()
