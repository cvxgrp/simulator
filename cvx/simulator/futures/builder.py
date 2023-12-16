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

import pandas as pd

from cvx.simulator.futures.portfolio import FuturesPortfolio
from cvx.simulator.futures.state import FuturesState
from cvx.simulator.futures.utils import returns2prices

from .._abc.builder import Builder


@dataclass
class FuturesBuilder(Builder):
    def build(self):
        """Build Futures Portfolio"""
        return FuturesPortfolio(prices=self.prices, units=self.units, aum=self.aum)

    aum: float = 1e6
    _state: FuturesState = None

    def __post_init__(self) -> None:
        """
        The __post_init__ method is a special method of initialized instances
        of the _Builder class and is called after initialization.
        It sets the initial amount of cash in the portfolio to be equal to the input initial_cash parameter.

        The method takes no input parameter. It initializes the cash attribute in the internal
        _State object with the initial amount of cash in the portfolio, self.initial_cash.

        Note that this method is often used in Python classes for additional initialization routines
        that can only be performed after the object is fully initialized. __post_init__
        is called automatically after the object initialization.
        """

        super().__post_init__()
        self._state = FuturesState()

    @classmethod
    def from_returns(cls, returns):
        """Build Futures Portfolio from returns"""
        # compute artificial prices (but scaled such their returns are correct)

        prices = returns2prices(returns)
        return cls(prices=prices)

    @Builder.position.setter
    def position(self, position: pd.Series) -> None:
        """
        The position property returns the current position of the portfolio.
        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        self._units.loc[self._state.time, self._state.assets] = position
        self._state.position = position
