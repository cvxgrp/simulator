"""Test that the code examples in the README.md file work as expected."""

import doctest


def test_readme(readme_path):
    """Test that the code examples in the README.md file work as expected.

    This test runs doctest on the README.md file to verify that all code examples
    in the file execute correctly and produce the expected output.

    Parameters
    ----------
    readme_path : Path
        Path to the README.md file

    """
    result = doctest.testfile(str(readme_path), module_relative=False, verbose=True, optionflags=doctest.ELLIPSIS)
    print(result)

    assert result.failed == 0, f"{result.failed} doctests failed"
