# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Dict

TIMESERIES = Dict[datetime, float]
TIMEFRAME = Dict[str, TIMESERIES]
