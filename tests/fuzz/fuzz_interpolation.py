"""Fuzz the cvx.simulator polars interpolation helpers against arbitrary series.

``interpolate_pl``/``interpolate_df_pl`` interior-forward-fill polars series and
frames, and ``valid_pl``/``valid_df_pl`` check that the non-null span is
contiguous. None should crash with an unexpected exception on adversarial input
(empty, all-null, NaN/inf) — they should return a result or raise a documented
error. This harness exercises that contract with coverage-guided input.

Run locally:
    pip install atheris numpy pandas polars
    python tests/fuzz/fuzz_interpolation.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the heavy native dependencies OUTSIDE the instrumentation block.
# Atheris's import hook miscompiles parts of polars' Python machinery, so we let
# these load uninstrumented and instrument only the first-party package. The
# cvx.simulator package __init__ pulls in pandas/polars (and jquantstats), so
# they must be cached uninstrumented beforehand.
import numpy as np  # noqa: F401  # pre-imported uninstrumented
import pandas as pd  # noqa: F401  # pre-imported uninstrumented
import polars as pl

with atheris.instrument_imports():
    from cvx.simulator.utils.interpolation import (
        interpolate_df_pl,
        interpolate_pl,
        valid_df_pl,
        valid_pl,
    )

_ALLOWED = (ValueError, TypeError, pl.exceptions.PolarsError)


def _series(fdp: atheris.FuzzedDataProvider, name: str, n: int) -> pl.Series:
    """Build a nullable float polars Series from fuzzed bytes."""
    return pl.Series(name, [None if fdp.ConsumeBool() else fdp.ConsumeFloat() for _ in range(n)])


def test_one_input(data: bytes) -> None:
    """Exercise the polars interpolation helpers with a fuzzed series and frame."""
    fdp = atheris.FuzzedDataProvider(data)
    n = fdp.ConsumeIntInRange(0, 24)

    series = _series(fdp, "v", n)
    for fn in (valid_pl, interpolate_pl):
        with contextlib.suppress(_ALLOWED):
            fn(series)

    frame = pl.DataFrame({"a": _series(fdp, "a", n), "b": _series(fdp, "b", n)})
    for fn in (valid_df_pl, interpolate_df_pl):
        with contextlib.suppress(_ALLOWED):
            fn(frame)


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
