from cvx.simulator.utils.month import Aggregate, monthlytable


def test_month_compounded(nav):
    table = monthlytable(nav.pct_change().fillna(0.0), Aggregate.COMPOUND)
    print(table)


def test_month_cumulative(nav):
    table = monthlytable(nav.pct_change().fillna(0.0), Aggregate.CUMULATIVE)
    print(table)
