"""Tests for doctest examples in the project documentation.

This module contains tests that verify that the code examples in the project's
documentation (specifically in the README file) can be executed successfully
using Python's doctest module.
"""

import doctest


def test_doc(readme_path):
    """Test that the README file's code examples work correctly.

    This test runs doctest on the project's README file to verify that all
    code examples in the file execute correctly and produce the expected output.

    Parameters
    ----------
    readme_path : Path
        Path to the README file to test

    """
    doctest.testfile(str(readme_path), module_relative=False, verbose=True, optionflags=doctest.ELLIPSIS)
