import pandas as pd
import pytest

from cvx.simulator.utils.month import Aggregate, monthlytable


@pytest.fixture()
def nav(resource_dir):
    """nav fixture"""
    return pd.read_csv(
        resource_dir / "nav.csv", index_col=0, parse_dates=True, header=0
    ).squeeze()


def test_month_compounded(nav):
    table = monthlytable(nav.pct_change().fillna(0.0), Aggregate.COMPOUND)
    print(table)


def test_month_cumulative(nav):
    table = monthlytable(nav.pct_change().fillna(0.0), Aggregate.CUMULATIVE)
    print(table)
