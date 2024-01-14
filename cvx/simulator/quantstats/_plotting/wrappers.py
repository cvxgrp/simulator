# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
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

import warnings

import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import seaborn as _sns
from matplotlib.ticker import (
    FuncFormatter as _FuncFormatter,
)
from matplotlib.ticker import (
    StrMethodFormatter as _StrMethodFormatter,
)
from pandas import DataFrame as _df

from .. import (
    stats as _stats,
)
from .. import (
    utils as _utils,
)
from . import core as _core

_FLATUI_COLORS = ["#fedd78", "#348dc1", "#af4b64", "#4fa487", "#9b59b6", "#808080"]
_GRAYSCALE_COLORS = (len(_FLATUI_COLORS) * ["black"]) + ["white"]

_HAS_PLOTLY = False
try:
    import plotly

    _HAS_PLOTLY = True
except ImportError:
    pass


def to_plotly(fig):
    if not _HAS_PLOTLY:
        return fig
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plotly.tools.mpl_to_plotly(fig)
        return plotly.plotly.iplot(fig, filename="quantstats-plot", overwrite=True)


def earnings(
    returns,
    start_balance=1e5,
    mode="comp",
    grayscale=False,
    figsize=(10, 6),
    title="Portfolio Earnings",
    fontname="Arial",
    lw=1.5,
    subtitle=True,
    savefig=None,
    show=True,
):
    colors = _GRAYSCALE_COLORS if grayscale else _FLATUI_COLORS
    alpha = 0.5 if grayscale else 0.8

    returns = _utils.make_portfolio(returns, start_balance, mode)

    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[0] * 0.55)

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.suptitle(
        title, fontsize=14, y=0.995, fontname=fontname, fontweight="bold", color="black"
    )

    if subtitle:
        ax.set_title(
            "\n%s - %s ;  P&L: %s (%s)                "
            % (
                returns.index.date[1:2][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
                _utils._score_str(
                    "${:,}".format(round(returns.values[-1] - returns.values[0], 2))
                ),
                _utils._score_str(
                    "{:,}%".format(
                        round((returns.values[-1] / returns.values[0] - 1) * 100, 2)
                    )
                ),
            ),
            fontsize=12,
            color="gray",
        )

    mx = returns.max()
    returns_max = returns[returns == mx]
    ix = returns_max[~_np.isnan(returns_max)].index[0]
    returns_max = _np.where(returns.index == ix, mx, _np.nan)

    ax.plot(
        returns.index,
        returns_max,
        marker="o",
        lw=0,
        alpha=alpha,
        markersize=12,
        color=colors[0],
    )
    ax.plot(returns.index, returns, color=colors[1], lw=1 if grayscale else lw)

    ax.set_ylabel(
        "Value of  ${:,.0f}".format(start_balance),
        fontname=fontname,
        fontweight="bold",
        fontsize=12,
    )

    ax.yaxis.set_major_formatter(_FuncFormatter(_core.format_cur_axis))
    ax.yaxis.set_label_coords(-0.1, 0.5)

    fig.set_facecolor("white")
    ax.set_facecolor("white")
    fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def returns(
    returns,
    benchmark=None,
    grayscale=False,
    figsize=(10, 6),
    fontname="Arial",
    lw=1.5,
    match_volatility=False,
    compound=True,
    cumulative=True,
    resample=None,
    ylabel="Cumulative Returns",
    subtitle=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    title = "Cumulative Returns" if compound else "Returns"
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += " vs %s" % benchmark.upper()
        else:
            title += " vs Benchmark"
        if match_volatility:
            title += " (Volatility Matched)"

        benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    fig = _core.plot_timeseries(
        returns,
        benchmark,
        title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=False,
        resample=resample,
        compound=compound,
        cumulative=cumulative,
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def yearly_returns(
    returns,
    benchmark=None,
    fontname="Arial",
    grayscale=False,
    hlw=1.5,
    hlcolor="red",
    hllabel="",
    match_volatility=False,
    log_scale=False,
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    title = "EOY Returns"
    if benchmark is not None:
        title += "  vs Benchmark"
        benchmark = (
            _utils._prepare_benchmark(benchmark, returns.index)
            .resample("A")
            .apply(_stats.comp)
            .resample("A")
            .last()
        )

    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    if compounded:
        returns = returns.resample("A").apply(_stats.comp)
    else:
        returns = returns.resample("A").apply(_df.sum)
    returns = returns.resample("A").last()

    fig = _core.plot_returns_bars(
        returns,
        benchmark,
        fontname=fontname,
        hline=returns.mean(),
        hlw=hlw,
        hllabel=hllabel,
        hlcolor=hlcolor,
        match_volatility=match_volatility,
        log_scale=log_scale,
        resample=None,
        title=title,
        figsize=figsize,
        grayscale=grayscale,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def distribution(
    returns,
    fontname="Arial",
    grayscale=False,
    ylabel=True,
    figsize=(10, 6),
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    title=None,
    prepare_returns=True,
):
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    fig = _core.plot_distribution(
        returns,
        fontname=fontname,
        grayscale=grayscale,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        title=title,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def histogram(
    returns,
    benchmark=None,
    resample="M",
    fontname="Arial",
    grayscale=False,
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
        if benchmark is not None:
            benchmark = _utils._prepare_returns(benchmark)

    if resample == "W":
        title = "Weekly "
    elif resample == "M":
        title = "Monthly "
    elif resample == "Q":
        title = "Quarterly "
    elif resample == "A":
        title = "Annual "
    else:
        title = ""

    return _core.plot_histogram(
        returns,
        benchmark,
        resample=resample,
        grayscale=grayscale,
        fontname=fontname,
        title="Distribution of %sReturns" % title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )


def drawdowns_periods(
    returns,
    periods=5,
    lw=1.5,
    log_scale=False,
    fontname="Arial",
    grayscale=False,
    title=None,
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    fig = _core.plot_longest_drawdowns(
        returns,
        periods=periods,
        lw=lw,
        log_scale=log_scale,
        fontname=fontname,
        grayscale=grayscale,
        title=title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_beta(
    returns,
    benchmark,
    window1=126,
    window1_label="6-Months",
    window2=252,
    window2_label="12-Months",
    lw=1.5,
    fontname="Arial",
    grayscale=False,
    figsize=(10, 3),
    ylabel=True,
    subtitle=True,
    savefig=None,
    show=True,
    prepare_returns=True,
):
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    fig = _core.plot_rolling_beta(
        returns,
        benchmark,
        window1=window1,
        window1_label=window1_label,
        window2=window2,
        window2_label=window2_label,
        title="Rolling Beta to Benchmark",
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_volatility(
    returns,
    benchmark=None,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.5,
    fontname="Arial",
    grayscale=False,
    figsize=(10, 3),
    ylabel="Volatility",
    subtitle=True,
    savefig=None,
    show=True,
):
    returns = _stats.rolling_volatility(returns, period, periods_per_year)

    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index)
        benchmark = _stats.rolling_volatility(
            benchmark, period, periods_per_year, prepare_returns=False
        )

    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Volatility (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_sharpe(
    returns,
    benchmark=None,
    rf=0.0,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.25,
    fontname="Arial",
    grayscale=False,
    figsize=(10, 3),
    ylabel="Sharpe",
    subtitle=True,
    savefig=None,
    show=True,
):
    returns = _stats.rolling_sharpe(
        returns,
        rf,
        period,
        True,
        periods_per_year,
    )

    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        benchmark = _stats.rolling_sharpe(
            benchmark, rf, period, True, periods_per_year, prepare_returns=False
        )

    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Sharpe (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_sortino(
    returns,
    benchmark=None,
    rf=0.0,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.25,
    fontname="Arial",
    grayscale=False,
    figsize=(10, 3),
    ylabel="Sortino",
    subtitle=True,
    savefig=None,
    show=True,
):
    returns = _stats.rolling_sortino(returns, rf, period, True, periods_per_year)

    if benchmark is not None:
        benchmark = _utils._prepare_benchmark(benchmark, returns.index, rf)
        benchmark = _stats.rolling_sortino(
            benchmark, rf, period, True, periods_per_year, prepare_returns=False
        )

    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Sortino (%s)" % period_label,
        fontname=fontname,
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def monthly_heatmap(
    returns,
    benchmark=None,
    annot_size=10,
    figsize=(10, 5),
    cbar=True,
    square=False,
    returns_label="Strategy",
    compounded=True,
    eoy=False,
    grayscale=False,
    fontname="Arial",
    ylabel=True,
    savefig=None,
    show=True,
    active=False,
):
    # colors, ls, alpha = _core._get_colors(grayscale)
    cmap = "gray" if grayscale else "RdYlGn"

    returns = _stats.monthly_returns(returns, eoy=eoy, compounded=compounded) * 100

    fig_height = len(returns) / 2.5

    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0] * 1.051, max([fig_height, figsize[1]]))

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # _sns.set(font_scale=.9)
    if active and benchmark is not None:
        ax.set_title(
            f"{returns_label} - Monthly Active Returns (%)\n",
            fontsize=14,
            y=0.995,
            fontname=fontname,
            fontweight="bold",
            color="black",
        )
        benchmark = (
            _stats.monthly_returns(benchmark, eoy=eoy, compounded=compounded) * 100
        )
        active_returns = returns - benchmark

        ax = _sns.heatmap(
            active_returns,
            ax=ax,
            annot=True,
            center=0,
            annot_kws={"size": annot_size},
            fmt="0.2f",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )
    else:
        ax.set_title(
            f"{returns_label} - Monthly Returns (%)\n",
            fontsize=14,
            y=0.995,
            fontname=fontname,
            fontweight="bold",
            color="black",
        )
        ax = _sns.heatmap(
            returns,
            ax=ax,
            annot=True,
            center=0,
            annot_kws={"size": annot_size},
            fmt="0.2f",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )
    # _sns.set(font_scale=1)

    # align plot to match other
    if ylabel:
        ax.set_ylabel("Years", fontname=fontname, fontweight="bold", fontsize=12)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.tick_params(colors="#808080")
    _plt.xticks(rotation=0, fontsize=annot_size * 1.2)
    _plt.yticks(rotation=0, fontsize=annot_size * 1.2)

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def monthly_returns(
    returns,
    annot_size=10,
    figsize=(10, 5),
    cbar=True,
    square=False,
    compounded=True,
    eoy=False,
    grayscale=False,
    fontname="Arial",
    ylabel=True,
    savefig=None,
    show=True,
):
    return monthly_heatmap(
        returns=returns,
        annot_size=annot_size,
        figsize=figsize,
        cbar=cbar,
        square=square,
        compounded=compounded,
        eoy=eoy,
        grayscale=grayscale,
        fontname=fontname,
        ylabel=ylabel,
        savefig=savefig,
        show=show,
    )
