# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd


def iron_frame(frame, rule):
    """
    The iron_frame function takes a pandas DataFrame
    and keeps it constant on a coarser grid.

    :param frame: The frame to be ironed
    :param rule: The rule to be used for the construction of the grid
    :return: the ironed frame
    """
    s_index = resample_index(frame.index, rule)
    return _project_frame_to_grid(frame, s_index)


def resample_index(index, rule):
    """
    The resample_index function resamples a pandas DatetimeIndex object
    to a lower frequency using a specified rule.


    Note that the function does not modify the input index object,
    but rather returns a pandas DatetimeIndex
    """
    series = pd.Series(index=index, data=index)
    a = series.resample(rule=rule).first()
    return pd.DatetimeIndex(a.values)


def _project_frame_to_grid(frame, grid):
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
    sample.loc[grid] = frame.loc[grid]
    return sample.ffill()
