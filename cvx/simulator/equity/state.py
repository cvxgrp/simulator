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

import numpy as np
import pandas as pd

from .._abc.state import State


@dataclass
class EquityState(State):

    """The _State class defines a state object used to keep track of the current
    state of the portfolio.

    Attributes:

    prices: a pandas Series object containing the stock prices of the current
    portfolio state

    position: a pandas Series object containing the current holdings of the portfolio

    cash: the amount of cash available in the portfolio.

    time: the current time of the portfolio

    By default, prices and position are set to None, while cash is set to 1 million.
    These attributes can be updated and accessed through setter and getter methods
    """

    @property
    def nav(self) -> float:
        """
        The nav property computes the net asset value (NAV) of the portfolio,
        which is the sum of the current value of the
        portfolio as determined by the value property,
        and the current amount of cash available in the portfolio.
        """
        return self.value + self.cash

    @State.position.setter
    def position(self, position: np.array):
        """
        Update the position of the state. Computes the required trades
        and updates the cash balance accordingly.
        """
        # update the position
        position = pd.Series(index=self.assets, data=position)

        # compute the trades (can be fractional)
        self._trades = position.subtract(self.position, fill_value=0.0)

        # update only now as otherwise the trades would be wrong
        self._position = position

        # cash is spent for shares or received for selling them
        self.cash -= self.gross.sum()
