# -*- coding: utf-8 -*-
from __future__ import annotations

from cvx.simulator.grid import iron_frame


def test_iron_frame(prices):
    """
    Test the project frame to grid function
    :param prices:
    :return:
    """
    # compute the coarse grid
    frame = iron_frame(prices, rule="M")

    # fish for days the prices in the frame are changing
    price_change = frame.diff().abs().sum(axis=1)
    # only take into account significant changes
    ts = price_change[price_change > 1.0]
    # verify that there are 27 days with significant changes
    assert len(ts) == 27
