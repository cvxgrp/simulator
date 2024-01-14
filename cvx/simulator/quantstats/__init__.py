# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2023 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Ran Aroussi, Thomas Schmelzer"

from . import stats, utils, reports

__all__ = ["stats", "reports", "utils", "extend_pandas"]


def extend_pandas():
    """
    Extends pandas by exposing methods to be used like:
    df.sharpe(), df.best('day'), ...
    """
    from pandas.core.base import PandasObject as _po

    _po.compsum = stats.compsum
    _po.comp = stats.comp
    _po.expected_return = stats.expected_return
    _po.geometric_mean = stats.geometric_mean
    _po.ghpr = stats.ghpr
    _po.outliers = stats.outliers
    _po.remove_outliers = stats.remove_outliers
    _po.best = stats.best
    _po.worst = stats.worst
    _po.consecutive_wins = stats.consecutive_wins
    _po.consecutive_losses = stats.consecutive_losses
    _po.exposure = stats.exposure
    _po.win_rate = stats.win_rate
    _po.avg_return = stats.avg_return
    _po.avg_win = stats.avg_win
    _po.avg_loss = stats.avg_loss
    _po.volatility = stats.volatility
    _po.rolling_volatility = stats.rolling_volatility
    _po.implied_volatility = stats.implied_volatility
    _po.sharpe = stats.sharpe
    _po.smart_sharpe = stats.smart_sharpe
    _po.sortino = stats.sortino
    _po.smart_sortino = stats.smart_sortino
    _po.adjusted_sortino = stats.adjusted_sortino
    _po.rolling_sortino = stats.rolling_sortino
    _po.omega = stats.omega
    _po.cagr = stats.cagr
    _po.rar = stats.rar
    _po.skew = stats.skew
    _po.kurtosis = stats.kurtosis
    _po.calmar = stats.calmar
    _po.ulcer_index = stats.ulcer_index
    _po.ulcer_performance_index = stats.ulcer_performance_index
    _po.serenity_index = stats.serenity_index
    _po.risk_of_ruin = stats.risk_of_ruin
    _po.ror = stats.ror
    _po.value_at_risk = stats.value_at_risk
    _po.var = stats.var
    _po.conditional_value_at_risk = stats.conditional_value_at_risk
    _po.cvar = stats.cvar
    _po.expected_shortfall = stats.expected_shortfall
    _po.tail_ratio = stats.tail_ratio
    _po.payoff_ratio = stats.payoff_ratio
    _po.win_loss_ratio = stats.win_loss_ratio
    _po.profit_factor = stats.profit_factor
    _po.gain_to_pain_ratio = stats.gain_to_pain_ratio
    _po.cpc_index = stats.cpc_index
    _po.common_sense_ratio = stats.common_sense_ratio
    _po.outlier_win_ratio = stats.outlier_win_ratio
    _po.outlier_loss_ratio = stats.outlier_loss_ratio
    _po.recovery_factor = stats.recovery_factor
    _po.risk_return_ratio = stats.risk_return_ratio
    _po.max_drawdown = stats.max_drawdown
    _po.to_drawdown_series = stats.to_drawdown_series
    _po.kelly_criterion = stats.kelly_criterion
    _po.probabilistic_sharpe_ratio = stats.probabilistic_sharpe_ratio

    # methods from utils
    _po.to_returns = utils.to_returns
    _po.to_prices = utils.to_prices
    _po.aggregate_returns = utils.aggregate_returns
    _po.curr_month = utils._pandas_current_month
    _po.date = utils._pandas_date
    _po.mtd = utils._mtd
    _po.qtd = utils._qtd
    _po.ytd = utils._ytd

    _po.metrics = reports.metrics


# extend_pandas()
