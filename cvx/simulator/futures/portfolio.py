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

from dataclasses import dataclass

from cvx.simulator.portfolio import Portfolio


@dataclass(frozen=True)
class FuturesPortfolioCumulated(Portfolio):
    aum: float

    @property
    def nav(self):
        profit = (self.cashposition.shift(1) * self.returns.fillna(0.0)).sum(axis=1)

        return profit.cumsum() + self.aum


# @dataclass(frozen=True)
# class FuturesPortfolioCompounded(Portfolio):
#     aum: float
#
#     @property
#     def returns(self):
#         return self.profit / self.aum
#     @property
#     def nav(self):
#         return (self.returns + 1).cumprod() * self.aum
