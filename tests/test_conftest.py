"""Tests for the fixtures in conftest.py.

This module contains tests for the fixtures defined in conftest.py,
specifically testing error conditions in the readme_path fixture.
"""

from pathlib import Path


def test_resource_dir(resource_dir):
    """Test that the resource_dir fixture returns a valid path."""
    assert isinstance(resource_dir, Path)
    assert resource_dir.exists()
    assert resource_dir.is_dir()
