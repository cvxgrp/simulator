import numpy as np
import pandas as pd

from cvx.simulator.utils.interpolation import interpolate, valid


def test_interpolate():
    ts = pd.Series(
        data=[
            np.nan,
            np.nan,
            2,
            3,
            np.nan,
            np.nan,
            4,
            5,
            np.nan,
            np.nan,
            6,
            np.nan,
            np.nan,
        ]
    )
    a = interpolate(ts)
    assert valid(a)
