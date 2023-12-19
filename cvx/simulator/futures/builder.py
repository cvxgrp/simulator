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
from dataclasses import dataclass

from cvx.simulator.futures.portfolio import FuturesPortfolio
from cvx.simulator.utils.rescale import returns2prices

from .._abc.builder import Builder


@dataclass
class FuturesBuilder(Builder):
    aum: float = 1e6

    def __post_init__(self):
        super().__post_init__()
        self._state.cash = self.aum

    def build(self):
        """Build Futures Portfolio"""
        return FuturesPortfolio(prices=self.prices, units=self.units, aum=self.aum)

    @classmethod
    def from_returns(cls, returns):
        """Build Futures Portfolio from returns"""
        # compute artificial prices (but scaled such their returns are correct)

        prices = returns2prices(returns)
        return cls(prices=prices)
