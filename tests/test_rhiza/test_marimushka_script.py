"""Tests for the marimushka Makefile target using a sandboxed environment.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Provides test fixtures for testing git-based workflows and version management.
"""

import os
import shutil
import subprocess

# Get shell path and make command once at module level
SHELL = shutil.which("sh") or "/bin/sh"
MAKE = shutil.which("make") or "/usr/bin/make"


def test_marimushka_target_success(git_repo):
    """Test successful execution of the marimushka Makefile target."""
    # Setup directories in the git repo
    marimo_folder = git_repo / "book" / "marimo"
    marimo_folder.mkdir(parents=True)
    (marimo_folder / "notebook.py").touch()

    output_folder = git_repo / "_marimushka"

    # Run the make target
    env = os.environ.copy()
    env["MARIMO_FOLDER"] = "book/marimo"
    env["MARIMUSHKA_OUTPUT"] = "_marimushka"

    result = subprocess.run([MAKE, "marimushka"], env=env, cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "Exporting notebooks" in result.stdout
    assert (output_folder / "index.html").exists()
    assert (output_folder / "notebooks" / "notebook.html").exists()


def test_marimushka_missing_folder(git_repo):
    """Test marimushka target behavior when MARIMO_FOLDER is missing."""
    env = os.environ.copy()
    env["MARIMO_FOLDER"] = "missing"

    result = subprocess.run([MAKE, "marimushka"], env=env, cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "does not exist" in result.stdout


def test_marimushka_no_python_files(git_repo):
    """Test marimushka target behavior when MARIMO_FOLDER has no python files."""
    marimo_folder = git_repo / "book" / "marimo"
    marimo_folder.mkdir(parents=True)
    # No .py files created

    output_folder = git_repo / "_marimushka"

    env = os.environ.copy()
    env["MARIMO_FOLDER"] = "book/marimo"
    env["MARIMUSHKA_OUTPUT"] = "_marimushka"

    result = subprocess.run([MAKE, "marimushka"], env=env, cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "No Python files found" in result.stdout
    assert (output_folder / "index.html").exists()
    assert "No notebooks found" in (output_folder / "index.html").read_text()
