from dataclasses import dataclass
import abc

import pandas as pd


@dataclass(frozen=True)
class TradingCostModel(abc.ABC):
    name: str

    @abc.abstractmethod
    def eval(self, trades_volume, trades_stocks) -> pd.DataFrame:
        pass


@dataclass(frozen=True)
class LinearCostModel(TradingCostModel):
    factor: float = 0.0
    bias: float = 0.0

    def eval(self, trades_volume, trades_stocks):
        return self.factor * trades_volume.abs() + self.bias * trades_stocks.abs()