import pandas as pd
import numpy as np
import pytest

from cvx.simulator.metrics import Metrics

def test_length_one():
    profit = pd.Series(data=[1.0])
    m = Metrics(daily_profit=profit)

    assert np.isnan(m.std_profit)
    assert np.isnan(m.sr_profit)

def test_index_wrong_order():
    #t = pd.Timestamp.today().date()
    #print(t)
    profit = pd.Series(index=[5,4], data=[2.0, 3.0])
    with pytest.raises(AssertionError):
        m = Metrics(profit)

def test_nan_value():
    profit = pd.Series(index=[1, 2], data=[np.NaN, 2.0])
    with pytest.raises(AssertionError):
        m = Metrics(profit)
