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
import numpy as np
import pandas as pd


def interpolate(ts):
    """interpolate"""
    first = ts.first_valid_index()
    last = ts.last_valid_index()

    ts.loc[first:last] = ts.loc[first:last].ffill()
    return ts


def valid(ts):
    # check the two series are identical
    return (ts.dropna().index).equals(interpolate(ts).dropna().index)


if __name__ == "__main__":
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
    print(valid(a))
