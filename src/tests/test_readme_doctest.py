"""
Test that the code examples in the README.md file work as expected.

This module extracts the Python code blocks from the README.md file and runs them as doctests.
"""

import doctest
import os

import pytest


@pytest.fixture()
def readme_path():
    """Find the README.md file in the project root directory."""
    # Start from the current directory and go up until we find README.md
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # Stop at root directory
        readme_path = os.path.join(current_dir, "README.md")
        if os.path.exists(readme_path):
            return readme_path
        current_dir = os.path.dirname(current_dir)

    raise FileNotFoundError("README.md not found in any parent directory")


def test_readme(readme_path):
    result = doctest.testfile(str(readme_path), module_relative=False, verbose=True, optionflags=doctest.ELLIPSIS)
    print(result)

    assert result.failed == 0, f"{result.failed} doctests failed"
