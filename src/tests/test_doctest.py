"""Test that all docstrings in the project can be run with doctest."""

import doctest
import importlib
import pkgutil
import sys
from pathlib import Path


def get_all_modules(package_name):
    """Recursively find all modules in a package.

    Parameters
    ----------
    package_name : str
        The name of the package to search.

    Returns
    -------
    list
        A list of module names.

    """
    package = importlib.import_module(package_name)
    modules = []

    if hasattr(package, "__path__"):
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            if is_pkg:
                modules.extend(get_all_modules(name))
            else:
                modules.append(name)

    return modules


def test_doctest():
    """Test that all docstrings in the project can be run with doctest.

    This test recursively finds all modules in the cvx.simulator package,
    and runs doctest on each module to verify that the examples in the
    docstrings work as expected.
    """
    # Add the src directory to the Python path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))

    # Get all modules in the cvx.simulator package
    modules = get_all_modules("cvx.simulator")

    # Check that we found some modules
    assert modules, "No modules found in cvx.simulator package"

    # Modules to skip due to complex output formatting that's hard to test with doctest
    skip_modules = [
        #'cvx.simulator.portfolio'
    ]

    # Keep track of modules with failing doctests
    failing_modules = []

    # Run doctest on each module
    for module_name in modules:
        # Skip modules that are known to have doctest issues
        if any(module_name.startswith(skip) for skip in skip_modules):
            print(f"Skipping doctests for {module_name} (known formatting issues)")
            continue

        module = importlib.import_module(module_name)
        print(f"Running doctests for {module_name}")

        # Run doctest on the module with ELLIPSIS and NORMALIZE_WHITESPACE flags
        result = doctest.testmod(module, verbose=True, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)

        # If any tests failed, add the module to the list of failing modules
        if result.failed > 0:
            failing_modules.append(f"{module_name} ({result.failed} failed)")
            print(f"FAILED: {module_name} - {result.failed} tests failed")

    # Write failing modules to a file for examination
    if failing_modules:
        with open("failing_doctests.txt", "w") as f:
            f.write("\n".join(failing_modules))

    # Assert that there are no failing modules
    assert not failing_modules, f"The following modules have failing doctests: {', '.join(failing_modules)}"
