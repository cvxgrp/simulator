import pandas as pd
import numpy as np


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


def project_frame_to_grid(frame, grid):
    """The project_frame_to_grid function projects a pandas
    DataFrame object onto a given index grid.

    The function returns a new DataFrame that is only updated for times in the grid,
    otherwise the previous values carry over.

    Note that the function does not modify the input frame object, but rather returns a new object.
    """
    sample = pd.DataFrame(index=frame.index, columns=frame.columns, data=np.NaN)
    sample.loc[grid] = frame.loc[grid]
    return sample.ffill()
