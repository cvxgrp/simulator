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
import pandas as pd


def _rescale(r: pd.Series):
    """
    Rescale a time series to start at 1
    """
    r = r.dropna()
    a = (r + 1).cumprod()
    return a / a.iloc[0]


def returns2prices(returns: pd.DataFrame):
    """
    Convert returns to prices
    """
    prices = returns.apply(_rescale, axis=0)
    return prices
