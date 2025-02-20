import pytest

from cvx.simulator.utils.metric import sharpe


def test_sharpe(nav):
    s = sharpe(nav.pct_change())
    print(s)
    assert s == pytest.approx(-0.10388478316042028)
