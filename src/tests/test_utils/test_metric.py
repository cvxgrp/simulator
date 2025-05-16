import pytest


def test_sharpe(data) -> None:
    s = data.stats.sharpe()
    assert s["NAV"] == pytest.approx(-0.10388478316042028)
