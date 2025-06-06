"""Test that all docstrings in the project can be run with doctest."""

import doctest


def test_doc(readme_path):
    doctest.testfile(str(readme_path), module_relative=False, verbose=True, optionflags=doctest.ELLIPSIS)
