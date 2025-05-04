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


def _periods(ts):
    """
    compute the number of periods in a time series
    """
    series = pd.Series(data=ts.index)
    return 365 * 24 * 60 * 60 / (series.diff().dropna().mean().total_seconds())


def sharpe(ts, n=None):
    """
    compute the sharpe ratio of a time series
    """
    std = ts.std()
    if std > 0:
        n = n or _periods(ts)
        return (ts.mean() / std) * np.sqrt(n)
    else:
        return np.inf
