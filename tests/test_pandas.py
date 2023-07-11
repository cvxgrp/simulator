# -*- coding: utf-8 -*-
import pandas as pd


def test_ratio():
    a = pd.Series(index=["a", "b"], data=[1.0, 2.0])
    b = pd.Series(index=["a", "b", "c"], data=[2.0, 4.0, 6.0])
    assert set(a.index).issubset(set(b.index))
    ratio = a / b
    pd.testing.assert_index_equal(ratio.index, b.index)
