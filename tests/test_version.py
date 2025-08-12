"""Tests for the version attribute of the cvxsimulator package.

This module contains a simple test to verify that the cvxsimulator package
has a non-None __version__ attribute, which is important for package versioning.
"""

import cvxsimulator  # Import the main package only


def test_version():
    """Test that the cvxsimulator package has a version.

    This test verifies that the __version__ attribute of the cvxsimulator
    package is not None, ensuring that the package has proper version information.
    """
    assert cvxsimulator.__version__ is not None
