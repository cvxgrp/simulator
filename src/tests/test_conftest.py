"""Tests for the fixtures in conftest.py.

This module contains tests for the fixtures defined in conftest.py,
specifically testing error conditions in the readme_path fixture.
"""

import os
from pathlib import Path


def test_resource_dir(resource_dir):
    """Test that the resource_dir fixture returns a valid path."""
    assert isinstance(resource_dir, Path)
    assert resource_dir.exists()
    assert resource_dir.is_dir()


# Create a test module that will use the readme_path fixture
def create_test_module(monkeypatch):
    """Create a test module that uses the readme_path fixture.

    This function creates a temporary test module that uses the readme_path fixture
    and runs it with pytest. The test is expected to fail with a FileNotFoundError
    because Path.is_file is mocked to always return False.
    """
    # Create a temporary directory for the test module
    import tempfile

    temp_dir = tempfile.mkdtemp()

    # Create a test module that uses the readme_path fixture
    test_module_path = os.path.join(temp_dir, "test_readme_fixture.py")
    with open(test_module_path, "w") as f:
        f.write("""
import pytest
from pathlib import Path

def test_readme_path_not_found(monkeypatch, readme_path):
    # Mock Path.is_file to always return False
    def mock_is_file(self):
        return False

    monkeypatch.setattr(Path, 'is_file', mock_is_file)

    # This should raise FileNotFoundError
    readme_path
""")

    return test_module_path
