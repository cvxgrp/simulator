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

import numpy as np
import pandas as pd

from .._abc.builder import Builder
from .portfolio import EquityPortfolio
from .state import EquityState


@dataclass
class EquityBuilder(Builder):
    initial_cash: float = 1e6
    _state: EquityState = None
    _cash: pd.Series = None

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

        self._cash = pd.Series(index=self.index, data=np.NaN)
        self._state = EquityState()
        self._state.cash = self.initial_cash

    @property
    def weights(self) -> np.array:
        """
        Get the current weights from the state
        """
        return self._state.weights[self._state.assets].values

    @weights.setter
    def weights(self, weights: np.array) -> None:
        """
        The weights property sets the current weights of the portfolio.
        We convert the weights to positions using the current prices and the NAV
        """
        self.position = self._state.nav * weights / self.current_prices

    @Builder.position.setter
    def position(self, position: pd.Series) -> None:
        """
        The position property returns the current position of the portfolio.
        It returns a pandas Series object containing the current position of the portfolio.

        Returns: pd.Series: a pandas Series object containing the current position of the portfolio.
        """
        self._units.loc[self._state.time, self._state.assets] = position
        self._state.position = position

        self._cash[self._state.time] = self._state.cash

    @property
    def cash(self):
        """
        The cash property returns the current cash available in the portfolio.
        """
        return self._cash

    def build(self) -> EquityPortfolio:
        """A function that creates a new instance of the EquityPortfolio
        class based on the internal state of the Portfolio builder object.

        Returns: EquityPortfolio: A new instance of the EquityPortfolio class
        with the attributes (prices, units, initial_cash, trading_cost_model) as specified in the Portfolio builder.

        Notes: The function simply creates a new instance of the EquityPortfolio
        class with the attributes (prices, units, initial_cash, trading_cost_model) equal
        to the corresponding attributes in the Portfolio builder object.
        The resulting EquityPortfolio object will have the same state as the Portfolio builder from which it was built.
        """

        return EquityPortfolio(prices=self.prices, units=self.units, cash=self.cash)
