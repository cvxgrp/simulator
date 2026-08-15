"""Execute every doctest in the package as part of the ordinary test run.

Without this, the ``>>>`` examples in the public API are documentation that
nothing verifies: ``interrogate`` only asks whether a docstring exists, and
``make test`` collects ``tests/`` alone. That gap is not hypothetical — the
polars example in ``interpolate_pl`` had drifted from what the function
actually renders, and stayed that way because no gate ever ran it.

Modules are discovered rather than listed, so a new module's examples are
covered the day it is added.
"""

import contextlib
import doctest
import importlib
import io
import pkgutil

import pytest

import cvx.simulator


def _modules() -> list[str]:
    """Return the import paths of every module in the cvx.simulator package.

    Returns:
    -------
    list[str]
        Dotted module paths, including the package root itself.

    """
    names = [cvx.simulator.__name__]
    names.extend(info.name for info in pkgutil.walk_packages(cvx.simulator.__path__, prefix="cvx.simulator."))
    return sorted(names)


@pytest.mark.parametrize("module_name", _modules())
def test_doctests(module_name: str) -> None:
    """Run the doctests of a single module and fail with the offending diff.

    Parameters
    ----------
    module_name : str
        Dotted path of the module whose docstrings should be executed.

    """
    module = importlib.import_module(module_name)

    # doctest reports failures to stdout; capture it so the assertion message
    # carries the expected/got diff rather than just a count.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = doctest.testmod(module, verbose=False)

    assert result.failed == 0, f"{module_name}: {result.failed} doctest failure(s)\n\n{buffer.getvalue()}"


def test_public_api_carries_examples() -> None:
    """Every name exported from cvx.simulator must document a runnable example.

    Guards the gap this suite was added to close: a class can hold a perfect
    docstring, score 100% on interrogate, and still show the reader nothing
    they can run.
    """
    parser = doctest.DocTestParser()
    missing = []

    for name in cvx.simulator.__all__:
        obj = getattr(cvx.simulator, name)
        if not (isinstance(obj, type) or callable(obj)):
            continue  # __version__ and friends carry no docstring of their own
        docstring = obj.__doc__ or ""
        if not parser.get_examples(docstring):
            missing.append(name)

    assert not missing, f"exported names without a doctest example: {missing}"
