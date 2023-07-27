# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Dict

TIMESERIES = Dict[datetime, float]
TIMEFRAME = Dict[str, TIMESERIES]


def a(xx: TIMESERIES) -> None:
    print(xx)
    print(type(xx))


if __name__ == "__main__":
    import pandas as pd

    x = pd.Series(
        index=pd.date_range("2020-01-01", "2020-01-10", freq="D"), data=range(10)
    )

    a(xx=x.to_dict())

    x = pd.Series(index=range(10), data=range(10))

    a(xx=x.to_dict())
