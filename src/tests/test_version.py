"""Tests for the version attribute of the cvx.simulator package.

This module contains a simple test to verify that the cvx.simulator package
has a non-None __version__ attribute, which is important for package versioning.
"""

import cvx.simulator


def test_version():
    """Test that the cvx.simulator package has a version.

    This test verifies that the __version__ attribute of the cvx.simulator
    package is not None, ensuring that the package has proper version information.
    """
    assert cvx.simulator.__version__ is not None
