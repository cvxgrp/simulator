"""Filling gaps between the first and last valid observation.

Covers `interpolate` and the polars-specific `interpolate_pl` and
`interpolate_df_pl`. The contract under test throughout: missing values
*between* the first and last valid observation are carried forward; missing
values outside that span are left alone, because there is nothing to carry.

Deciding whether a series has such gaps is a separate behaviour, tested in
`test_validity.py`.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cvx.simulator.utils import interpolate, interpolate_df_pl, interpolate_pl, valid, valid_pl

# --------------------------------------------------------------------------
# pandas
# --------------------------------------------------------------------------


def test_leading_and_trailing_gaps_survive_pandas() -> None:
    """Interior NaNs are filled; NaNs before the first and after the last value are not.

    The series runs [NaN, NaN, 2, 3, NaN, NaN, 4, 5, NaN, NaN, 6, NaN, NaN],
    so it exercises all three regions at once.
    """
    ts = pd.Series(data=[np.nan, np.nan, 2, 3, np.nan, np.nan, 4, 5, np.nan, np.nan, 6, np.nan, np.nan])

    assert valid(interpolate(ts))


def test_gap_is_filled_with_the_previous_value_pandas() -> None:
    """A hole takes the last observation before it, not an average of its neighbours."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    ts = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], index=dates)

    result = interpolate(ts)

    assert valid(result)
    assert result[dates[2]] == 2.0


def test_all_nan_series_is_returned_unchanged_pandas() -> None:
    """With no valid observation there is no span to fill, so nothing changes."""
    ts = pd.Series([np.nan, np.nan, np.nan])

    pd.testing.assert_series_equal(interpolate(ts), ts)


def test_empty_series_is_returned_empty_pandas() -> None:
    """An empty series interpolates to an empty series rather than raising."""
    assert len(interpolate(pd.Series([]))) == 0


# --------------------------------------------------------------------------
# polars Series
# --------------------------------------------------------------------------


def test_leading_and_trailing_gaps_survive_polars() -> None:
    """The pandas contract holds for polars: interior nulls filled, edges preserved."""
    ts = pl.Series([None, None, 2, 3, None, None, 4, 5, None, None, 6, None, None])

    assert valid(interpolate(ts))


def test_interior_nulls_carry_the_previous_value_polars() -> None:
    """[1.0, null, null, 4.0] fills forward to [1.0, 1.0, 1.0, 4.0]."""
    result = interpolate(pl.Series([1.0, None, None, 4.0]))

    assert result.to_list() == [1.0, 1.0, 1.0, 4.0]


def test_all_null_series_is_returned_unchanged_polars() -> None:
    """With no valid observation there is nothing to carry forward."""
    result = interpolate(pl.Series([None, None, None]))

    assert result.to_list() == [None, None, None]


def test_single_observation_leaves_both_edges_null_polars() -> None:
    """One value means the valid span has zero width, so neither edge is filled."""
    result = interpolate_pl(pl.Series([None, 1, None]))

    assert result.to_list() == [None, 1, None]


def test_empty_series_is_returned_empty_polars() -> None:
    """An empty series interpolates to an empty series rather than raising."""
    assert len(interpolate_pl(pl.Series([], dtype=pl.Float64))) == 0


# --------------------------------------------------------------------------
# polars DataFrame
# --------------------------------------------------------------------------


def test_every_column_is_filled_independently() -> None:
    """Each column gets its own valid span; a date column passes through untouched."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    dframe = pl.DataFrame({"date": dates, "A": [1.0, 2.0, None, 4.0, 5.0], "B": [None, 2.0, 3.0, None, 5.0]})

    result = interpolate_df_pl(dframe)

    assert result["date"].to_list() == dates
    assert valid_pl(result["A"])
    assert valid_pl(result["B"])
    # B's leading null survives — it precedes B's first observation.
    assert result["A"].to_list() == [1.0, 2.0, 2.0, 4.0, 5.0]
    assert result["B"].to_list() == [None, 2.0, 3.0, 3.0, 5.0]


def test_filling_is_not_limited_to_numeric_columns() -> None:
    """Strings and booleans carry forward the same way integers and floats do."""
    dframe = pl.DataFrame(
        {
            "int": [1, None, None, 4, 5],
            "float": [1.0, None, None, 4.0, 5.0],
            "str": ["a", None, None, "d", "e"],
            "bool": [True, None, None, False, True],
        }
    )

    result = interpolate_df_pl(dframe)

    for col in result.columns:
        assert valid_pl(result[col])
    assert result["int"].to_list() == [1, 1, 1, 4, 5]
    assert result["float"].to_list() == [1.0, 1.0, 1.0, 4.0, 5.0]
    assert result["str"].to_list() == ["a", "a", "a", "d", "e"]
    assert result["bool"].to_list() == [True, True, True, False, True]


def test_empty_dataframe_is_returned_empty() -> None:
    """A frame with no columns has nothing to fill and must not raise."""
    assert interpolate_df_pl(pl.DataFrame()).shape == (0, 0)


# --------------------------------------------------------------------------
# type dispatch
# --------------------------------------------------------------------------


def test_non_series_input_is_rejected() -> None:
    """Anything that is neither a pandas nor a polars Series raises TypeError.

    The message names the offending type, so the caller can see what they
    passed rather than only that it was wrong.
    """
    with pytest.raises(TypeError, match=r"Expected pd\.Series or pl\.Series, got <class 'list'>"):
        interpolate([1, 2, 3])
