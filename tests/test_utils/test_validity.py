"""Deciding whether a series has a gap in the middle.

Covers `valid` and the polars-specific `valid_pl` and `valid_df_pl`. The
contract under test throughout: a series is valid when every value between its
first and last observation is present. Missing values *outside* that span —
before an asset starts trading, or after it stops — are expected and do not
make the series invalid.

Filling those interior gaps is a separate behaviour, tested in
`test_interpolation.py`.
"""

import pandas as pd
import polars as pl
import pytest

from cvx.simulator.utils import valid, valid_df_pl, valid_pl

# --------------------------------------------------------------------------
# pandas
# --------------------------------------------------------------------------


def test_edge_gaps_are_valid_but_interior_gaps_are_not_pandas() -> None:
    """NaNs at the ends are fine; a NaN between two observations is not."""
    assert valid(pd.Series([float("nan"), 1, 2, 3, float("nan")]))
    assert not valid(pd.Series([1, 2, float("nan"), 4, 5]))


def test_empty_series_is_valid_pandas() -> None:
    """An empty series has no interior to inspect, so it cannot have a gap."""
    assert valid(pd.Series([]))


# --------------------------------------------------------------------------
# polars Series
# --------------------------------------------------------------------------


def test_edge_gaps_are_valid_but_interior_gaps_are_not_polars() -> None:
    """The pandas contract holds for polars nulls."""
    assert valid(pl.Series([None, 1, 2, 3, None]))
    assert not valid(pl.Series([1, 2, None, 4, 5]))


def test_valid_dispatches_polars_input_to_valid_pl() -> None:
    """`valid` on a polars Series must agree with calling `valid_pl` directly.

    Guards the dispatch in `valid`: were it to fall through to the pandas
    branch, a polars Series would be judged by the wrong code path.
    """
    unbroken = pl.Series([None, 1, 2, 3, None])
    broken = pl.Series([1, 2, None, 4, 5])

    assert valid(unbroken) == valid_pl(unbroken)
    assert valid(broken) == valid_pl(broken)


def test_fewer_than_two_observations_is_valid_polars() -> None:
    """Zero or one observation cannot straddle a gap, so both are valid."""
    assert valid_pl(pl.Series([None, None, None]))
    assert valid_pl(pl.Series([None, 1, None]))


def test_empty_series_is_valid_polars() -> None:
    """An empty series has no interior to inspect, so it cannot have a gap."""
    assert valid_pl(pl.Series([], dtype=pl.Float64))


# --------------------------------------------------------------------------
# polars DataFrame
# --------------------------------------------------------------------------


def test_frame_is_valid_when_every_column_is() -> None:
    """Columns may start and stop at different times and still be valid together."""
    dframe = pl.DataFrame(
        {
            "A": [None, 1, 2, 3, None],  # starts late, ends early
            "B": [None, 2, 3, 4, None],
            "C": [1, 2, 3, 4, 5],  # complete
        }
    )

    assert valid_df_pl(dframe)


def test_one_broken_column_invalidates_the_frame() -> None:
    """A single interior gap anywhere makes the whole frame invalid."""
    dframe = pl.DataFrame(
        {
            "A": [None, 1, 2, 3, None],
            "B": [1, 2, None, 4, 5],  # gap in the middle
            "C": [1, 2, 3, 4, 5],
        }
    )

    assert not valid_df_pl(dframe)


def test_empty_dataframe_is_valid() -> None:
    """A frame with no columns has no column that could be broken."""
    assert valid_df_pl(pl.DataFrame())


# --------------------------------------------------------------------------
# type dispatch
# --------------------------------------------------------------------------


def test_non_series_input_is_rejected() -> None:
    """Anything that is neither a pandas nor a polars Series raises TypeError.

    The message names the offending type, so the caller can see what they
    passed rather than only that it was wrong.
    """
    with pytest.raises(TypeError, match=r"Expected pd\.Series or pl\.Series, got <class 'list'>"):
        valid([1, 2, 3])
