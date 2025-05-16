# #    Copyright 2023 Stanford University Convex Optimization Group
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.
# """
# Performance metrics for financial time series.
#
# This module provides functions for calculating various performance metrics
# for financial time series, such as the Sharpe ratio.
# """
#
# from typing import Optional, Union
#
# import numpy as np
# import pandas as pd
#
#
# def _periods(ts: pd.Series) -> float:
#     """
#     Compute the number of periods in a year for a time series.
#
#     This function estimates the number of periods per year based on the average
#     time difference between consecutive observations in the time series.
#
#     Parameters
#     ----------
#     ts : pd.Series
#         Time series with a datetime index
#
#     Returns
#     -------
#     float
#         Estimated number of periods per year
#
#     Notes
#     -----
#     The calculation assumes a 365-day year and converts the average time
#     difference between observations from seconds to the number of such
#     periods in a year.
#     """
#     series = pd.Series(data=ts.index)
#     return 365 * 24 * 60 * 60 / (series.diff().dropna().mean().total_seconds())
#
#
# def sharpe(ts: pd.Series, n: Optional[float] = None) -> float:
#     """
#     Compute the Sharpe ratio of a time series.
#
#     The Sharpe ratio is a measure of risk-adjusted return, calculated as
#     the mean return divided by the standard deviation of returns, and then
#     annualized by multiplying by the square root of the number of periods
#     per year.
#
#     Parameters
#     ----------
#     ts : pd.Series
#         Time series of returns
#     n : float, optional
#         Number of periods per year for annualization. If None, it will be
#         estimated from the time series using the _periods function.
#
#     Returns
#     -------
#     float
#         Sharpe ratio of the time series
#
#     Notes
#     -----
#     If the standard deviation is zero, the function returns infinity,
#     as this represents a theoretical "perfect" risk-adjusted return.
#
#     Examples
#     --------
#     >>> import pandas as pd
#     >>> import numpy as np
#     >>> from datetime import datetime, timedelta
#     >>> dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
#     >>> returns = pd.Series([0.001, 0.002, -0.001, 0.003, 0.001,
#     ...                      0.002, 0.001, -0.002, 0.003, 0.001], index=dates)
#     >>> sharpe(returns, n=252)  # Using 252 trading days per year
#     2.6833...
#     """
#     std = ts.std()
#     if std > 0:
#         n = n or _periods(ts)
#         return (ts.mean() / std) * np.sqrt(n)
#     else:
#         return np.inf
