import numpy as np
import pandas as pd

from cvx.simulator.utils.interpolation import interpolate, valid


def test_interpolate():
    ts = pd.Series(
        data=[
            np.NaN,
            np.NaN,
            2,
            3,
            np.NaN,
            np.NaN,
            4,
            5,
            np.NaN,
            np.NaN,
            6,
            np.NaN,
            np.NaN,
        ]
    )
    a = interpolate(ts)
    assert valid(a)
