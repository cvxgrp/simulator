"""
Test that the code examples in the README.md file work as expected.
"""

import doctest


def test_readme(readme_path):
    result = doctest.testfile(str(readme_path), module_relative=False, verbose=True, optionflags=doctest.ELLIPSIS)
    print(result)

    assert result.failed == 0, f"{result.failed} doctests failed"
