from dataclasses import dataclass
import abc


@dataclass(frozen=True)
class TradingCostModel(abc.ABC):
    name: str

    @abc.abstractmethod
    def eval(self, prices, trades, **kwargs):
        pass


@dataclass(frozen=True)
class LinearCostModel(TradingCostModel):
    factor: float = 0.0
    bias: float = 0.0

    def eval(self, prices, trades, **kwargs):
        volume = prices * trades
        return self.factor * volume.abs() + self.bias * trades.abs()
