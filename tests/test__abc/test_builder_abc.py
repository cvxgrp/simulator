from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from cvx.simulator._abc.builder import Builder
from cvx.simulator._abc.state import State


@dataclass
class TestState(State):
    @State.position.setter
    def position(self):
        pass


@dataclass
class TestBuilder(Builder):
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
        self._state = TestState()

    @Builder.position.setter
    def position(self, position):
        print(position)
        self._units.loc[self._state.time] = position
        print(self._units)

    def build(self):
        pass


@pytest.fixture()
def builder(prices):
    builder = TestBuilder(prices=prices)
    builder._state = TestState(prices=prices.iloc[0])
    return builder


def test_index(builder, prices):
    """
    Test that the index of the builder is the same as the index of the prices
    :param builder: the builder object (fixture)
    """
    assert len(builder.index) == 602
    pd.testing.assert_index_equal(builder.index, prices.index)


def test_prices(builder, prices):
    """
    Test that the prices of the builder are the same as the prices
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    pd.testing.assert_frame_equal(builder.prices, prices)


def test_stocks(builder):
    """
    Test that the units are all zero
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    pd.testing.assert_frame_equal(builder.units, np.NaN * builder.prices)


def test_iteration(builder):
    """
    iterate through builder and verify that the yielded times match the index
    :param builder: the builder object (fixture)
    """
    assert {t[-1] for t, _ in builder} == set(builder.index)


def test_valid(builder):
    """Verify all assets are valid"""
    assert np.all(builder.valid)


def test_intervals(builder):
    x = builder.intervals
    assert x["last"].loc["G"] == pd.Timestamp("2015-04-22")


def test_current_prices(builder):
    prices = builder.prices.iloc[0].values
    np.allclose(builder.current_prices, prices)


def test_set_position(builder, prices):
    builder._state.time = prices.index[0]
    x = pd.Series(index=prices.columns, data=np.ones(7))

    builder.position = x
    assert np.allclose(builder.position.values, np.ones(7))
    assert builder.cashposition.sum() == pytest.approx(prices.iloc[0].sum())


def test_set_cashposition(builder, prices):
    builder._state.time = prices.index[0]
    builder.cashposition = prices.iloc[0]

    assert np.allclose(builder.position.values, np.ones(7))