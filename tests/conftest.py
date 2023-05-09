"""global fixtures"""

from pathlib import Path

import pytest
import pandas as pd

from cvx.simulator.portfolio import EquityPortfolio


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def prices(resource_dir):
    return pd.read_csv(resource_dir / "price.csv", index_col=0, parse_dates=True, header=0)


@pytest.fixture()
def portfolio(prices):
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    return EquityPortfolio(prices, stocks=positions, initial_cash=1e6)


@pytest.fixture()
def returns(resource_dir):
    return pd.read_csv(resource_dir / "ts.csv", index_col=0, header=None, parse_dates=True).squeeze().pct_change()
