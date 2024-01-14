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

from datetime import datetime as _dt
from math import ceil as _ceil
from math import sqrt as _sqrt

import numpy as _np
import pandas as _pd
from dateutil.relativedelta import relativedelta

from . import stats as _stats


def _get_trading_periods(periods_per_year=252):
    half_year = _ceil(periods_per_year / 2)
    return periods_per_year, half_year


def metrics(
    returns,
    rf=0.0,
    mode="basic",
    compounded=True,
    periods_per_year=252,
    match_dates=True,
    **kwargs,
):
    if match_dates:
        returns = returns.dropna()
    returns.index = returns.index.tz_localize(None)
    win_year, _ = _get_trading_periods(periods_per_year)

    # benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    # strategy_colname = kwargs.get("strategy_title", "Strategy")

    s_start = {"returns": returns.index.strftime("%Y-%m-%d")[0]}
    s_end = {"returns": returns.index.strftime("%Y-%m-%d")[-1]}
    s_rf = {"returns": rf}

    df = returns.fillna(0)

    pct = 1
    # pct multiplier
    if kwargs.get("as_pct", False):
        pct = 100

    # return df
    dd = _calc_dd(
        df,
        as_pct=kwargs.get("as_pct", False),
    )

    metrics = _pd.DataFrame()
    metrics["Start Period"] = _pd.Series(s_start)
    metrics["End Period"] = _pd.Series(s_end)
    metrics["Risk-Free Rate %"] = _pd.Series(s_rf) * 100
    metrics["Time in Market %"] = _stats.exposure(df) * pct

    if compounded:
        # display cumulative return with two digits after the decimal point
        metrics["Cumulative Return %"] = f"{_stats.comp(df) * pct,'.2f'}"
        # metrics["Cumulative Return %"] = f"{_stats.comp(df) * pct,.2f}"

    else:
        metrics["Total Return %"] = (df.sum() * pct).map("{:,.2f}".format)

    metrics["CAGR﹪%"] = _stats.cagr(df, rf, compounded) * pct

    metrics["Sharpe"] = _stats.sharpe(df, rf, win_year, True)
    metrics["Prob. Sharpe Ratio %"] = (
        _stats.probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )
    if mode.lower() == "full":
        metrics["Smart Sharpe"] = _stats.smart_sharpe(df, rf, win_year, True)
        # metrics['Prob. Smart Sharpe Ratio %'] = _stats.probabilistic_sharpe_ratio(df, rf, win_year, False, True) * pct

    metrics["Sortino"] = _stats.sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        # metrics['Prob. Sortino Ratio %'] = _stats.probabilistic_sortino_ratio(df, rf, win_year, False) * pct
        metrics["Smart Sortino"] = _stats.smart_sortino(df, rf, win_year, True)
        # metrics['Prob. Smart Sortino Ratio %'] = _stats.probabilistic_sortino_ratio(df, rf, win_year, False, True) * pct

    metrics["Sortino/√2"] = metrics["Sortino"] / _sqrt(2)
    if mode.lower() == "full":
        # metrics['Prob. Sortino/√2 Ratio %'] = _stats.probabilistic_adjusted_sortino_ratio(df, rf, win_year, False) * pct
        metrics["Smart Sortino/√2"] = metrics["Smart Sortino"] / _sqrt(2)
        # metrics['Prob. Smart Sortino/√2 Ratio %'] = _stats.probabilistic_adjusted_sortino_ratio(df, rf, win_year, False, True) * pct
    metrics["Omega"] = _stats.omega(df, 0.0, win_year)

    if mode.lower() == "full":
        ret_vol = _stats.volatility(df, win_year, True) * pct

        metrics["Volatility (ann.) %"] = [ret_vol]

        metrics["Calmar"] = _stats.calmar(df)
        metrics["Skew"] = _stats.skew(df)
        metrics["Kurtosis"] = _stats.kurtosis(df)

        metrics["Expected Daily %%"] = (
            _stats.expected_return(df, compounded=compounded) * pct
        )
        metrics["Expected Monthly %%"] = (
            _stats.expected_return(df, compounded=compounded, aggregate="M") * pct
        )
        metrics["Expected Yearly %%"] = (
            _stats.expected_return(df, compounded=compounded, aggregate="A") * pct
        )
        metrics["Kelly Criterion %"] = _stats.kelly_criterion(df) * pct
        metrics["Risk of Ruin %"] = _stats.risk_of_ruin(df)

        metrics["Daily Value-at-Risk %"] = -abs(_stats.var(df) * pct)
        metrics["Expected Shortfall (cVaR) %"] = -abs(_stats.cvar(df) * pct)

    if mode.lower() == "full":
        metrics["Max Consecutive Wins *int"] = _stats.consecutive_wins(df)
        metrics["Max Consecutive Losses *int"] = _stats.consecutive_losses(df)

    metrics["Gain/Pain Ratio"] = _stats.gain_to_pain_ratio(df, rf)
    metrics["Gain/Pain (1M)"] = _stats.gain_to_pain_ratio(df, rf, "M")
    # if mode.lower() == 'full':
    #     metrics['GPR (3M)'] = _stats.gain_to_pain_ratio(df, rf, "Q")
    #     metrics['GPR (6M)'] = _stats.gain_to_pain_ratio(df, rf, "2Q")
    #     metrics['GPR (1Y)'] = _stats.gain_to_pain_ratio(df, rf, "A")

    metrics["Payoff Ratio"] = _stats.payoff_ratio(df)
    metrics["Profit Factor"] = _stats.profit_factor(df)
    metrics["Common Sense Ratio"] = _stats.common_sense_ratio(df)
    metrics["CPC Index"] = _stats.cpc_index(df)
    metrics["Tail Ratio"] = _stats.tail_ratio(df)
    metrics["Outlier Win Ratio"] = _stats.outlier_win_ratio(df)
    metrics["Outlier Loss Ratio"] = _stats.outlier_loss_ratio(df)

    # returns
    comp_func = _stats.comp if compounded else _np.sum

    today = df.index[-1]  # _dt.today()
    metrics["MTD %"] = comp_func(df[df.index >= _dt(today.year, today.month, 1)]) * pct

    d = today - relativedelta(months=3)
    metrics["3M %"] = comp_func(df[df.index >= d]) * pct

    d = today - relativedelta(months=6)
    metrics["6M %"] = comp_func(df[df.index >= d]) * pct

    metrics["YTD %"] = comp_func(df[df.index >= _dt(today.year, 1, 1)]) * pct

    d = today - relativedelta(years=1)
    metrics["1Y %"] = comp_func(df[df.index >= d]) * pct

    d = today - relativedelta(months=35)
    metrics["3Y (ann.) %"] = _stats.cagr(df[df.index >= d], 0.0, compounded) * pct

    d = today - relativedelta(months=59)
    metrics["5Y (ann.) %"] = _stats.cagr(df[df.index >= d], 0.0, compounded) * pct

    d = today - relativedelta(years=10)
    metrics["10Y (ann.) %"] = _stats.cagr(df[df.index >= d], 0.0, compounded) * pct

    metrics["All-time (ann.) %"] = _stats.cagr(df, 0.0, compounded) * pct

    # best/worst
    if mode.lower() == "full":
        metrics["Best Day %"] = _stats.best(df, compounded=compounded) * pct
        metrics["Worst Day %"] = _stats.worst(df) * pct
        metrics["Best Month %"] = (
            _stats.best(df, compounded=compounded, aggregate="M") * pct
        )
        metrics["Worst Month %"] = _stats.worst(df, aggregate="M") * pct
        metrics["Best Year %"] = (
            _stats.best(df, compounded=compounded, aggregate="A") * pct
        )
        metrics["Worst Year %"] = (
            _stats.worst(df, compounded=compounded, aggregate="A") * pct
        )

    # dd
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics["Recovery Factor"] = _stats.recovery_factor(df)
    metrics["Ulcer Index"] = _stats.ulcer_index(df)
    metrics["Serenity Index"] = _stats.serenity_index(df, rf)

    # win rate
    if mode.lower() == "full":
        metrics["Avg. Up Month %"] = (
            _stats.avg_win(df, compounded=compounded, aggregate="M") * pct
        )
        metrics["Avg. Down Month %"] = (
            _stats.avg_loss(df, compounded=compounded, aggregate="M") * pct
        )
        metrics["Win Days %%"] = _stats.win_rate(df) * pct
        metrics["Win Month %%"] = (
            _stats.win_rate(df, compounded=compounded, aggregate="M") * pct
        )
        metrics["Win Quarter %%"] = (
            _stats.win_rate(df, compounded=compounded, aggregate="Q") * pct
        )
        metrics["Win Year %%"] = (
            _stats.win_rate(df, compounded=compounded, aggregate="A") * pct
        )

    metrics["Longest DD Days"] = _pd.to_numeric(metrics["Longest DD Days"]).astype(
        "int"
    )
    metrics["Avg. Drawdown Days"] = _pd.to_numeric(
        metrics["Avg. Drawdown Days"]
    ).astype("int")

    metrics.columns = [col if "~" not in col else "" for col in metrics.columns]
    metrics.columns = [col[:-1] if "%" in col else col for col in metrics.columns]
    metrics = metrics.T

    # cleanups
    metrics.replace([-0, "-0"], 0, inplace=True)
    metrics.replace(
        [
            _np.nan,
            -_np.nan,
            _np.inf,
            -_np.inf,
            "-nan%",
            "nan%",
            "-nan",
            "nan",
            "-inf%",
            "inf%",
            "-inf",
            "inf",
        ],
        "-",
        inplace=True,
    )

    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [
        c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns
    ]
    metrics = metrics.T

    return metrics


def _calc_dd(df, as_pct=False):
    dd = _stats.to_drawdown_series(df)
    dd_info = _stats.drawdown_details(dd)

    if dd_info.empty:
        return _pd.DataFrame()

    if "returns" in dd_info:
        ret_dd = dd_info["returns"]
    # to match multiple columns like returns_1, returns_2, ...
    elif (
        any(dd_info.columns.get_level_values(0).str.contains("returns"))
        and dd_info.columns.get_level_values(0).nunique() > 1
    ):
        ret_dd = dd_info.loc[
            :, dd_info.columns.get_level_values(0).str.contains("returns")
        ]
    else:
        ret_dd = dd_info

    dd_stats = {
        "returns": {
            "Max Drawdown %": ret_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "Longest DD Days": str(
                _np.round(
                    ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "Avg. Drawdown %": ret_dd["max drawdown"].mean() / 100,
            "Avg. Drawdown Days": str(_np.round(ret_dd["days"].mean())),
        }
    }

    # pct multiplier
    pct = 100 if as_pct else 1

    dd_stats = _pd.DataFrame(dd_stats).T
    dd_stats["Max Drawdown %"] = dd_stats["Max Drawdown %"].astype(float) * pct
    dd_stats["Avg. Drawdown %"] = dd_stats["Avg. Drawdown %"].astype(float) * pct

    return dd_stats.T
