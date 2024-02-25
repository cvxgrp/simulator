# from __future__ import annotations
#
# import pandas as pd
# import pytest
#
# from cvx.simulator.builder import Builder
# from cvx.simulator.utils.interpolation import interpolate
# from cvx.simulator.utils.metric import sharpe
#
#
# @pytest.fixture()
# def prices_interpolated(prices_hashed):
#     return prices_hashed.apply(interpolate)
#
#
# @pytest.fixture()
# def position(resource_dir):
#     return pd.read_csv(
#         resource_dir / "cashposition.csv", index_col=0, header=0, parse_dates=True
#     )
#
#
# @pytest.fixture()
# def portfolio(prices_interpolated, position):
#     builder = Builder(prices=prices_interpolated, initial_aum=1e7)
#
#     for t, state in builder:
#         pos = position[state.assets].loc[t[-1]].fillna(0.0)
#         builder.cashposition = pos
#         builder.aum = state.aum
#
#     portfolio = builder.build()
#     return portfolio
#
#
# def test_prices(prices_interpolated, portfolio):
#     pd.testing.assert_frame_equal(portfolio.prices, prices_interpolated)
#
#
# def test_nav(portfolio):
#     assert portfolio.nav[-1] == pytest.approx(96213100.91697493)
#
#
# def test_metrics(portfolio):
#     assert portfolio.profit.kurtosis() == pytest.approx(30.54402394742987)
#     assert sharpe(portfolio.profit) == pytest.approx(0.5511187319241556)
