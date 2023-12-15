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

from enum import Enum
from typing import Any

import pandas as pd
import quantstats as qs


class Plot(Enum):
    DAILY_RETURNS = 1
    DISTRIBUTION = 2
    DRAWDOWN = 3
    DRAWDOWNS_PERIODS = 4
    EARNINGS = 5
    HISTOGRAM = 6
    LOG_RETURNS = 7
    MONTHLY_HEATMAP = 8
    # see issue: https://github.com/ranaroussi/quantstats/issues/276
    MONTHLY_RETURNS = 9
    RETURNS = 10
    ROLLING_BETA = 11
    ROLLING_SHARPE = 12
    ROLLING_SORTINO = 13
    ROLLING_VOLATILITY = 14
    YEARLY_RETURNS = 15

    def plot(self, returns: pd.DataFrame, **kwargs: Any) -> Any:
        func = getattr(qs.plots, self.name.lower())
        return func(returns=returns, **kwargs)
