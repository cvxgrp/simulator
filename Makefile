# Makefile for cvxsimulator project
# This file provides common development tasks and utilities for the project
# Run 'make help' to see a list of available commands

# Colors for pretty output - used for formatting console messages
BLUE  := \033[36m  # Used for command names and highlights
BOLD  := \033[1m   # Used for emphasizing important text
RESET := \033[0m   # Resets text formatting to default

# Set the default goal to the help target when 'make' is run without arguments
.DEFAULT_GOAL := help

# Declare phony targets - these don't represent actual files
# This prevents conflicts with any files that might have the same names
.PHONY: help install fmt test coverage marimo clean

# Paths used throughout the Makefile
VENV_MARKER  := .venv/.installed  # Marker file to track virtual environment installation
TEST_DIR     := src/tests         # Directory containing test files
SOURCE_DIR   := src               # Directory containing source code
MARIMO_DIR   := book/marimo       # Directory containing marimo notebooks

##@ Development Setup
# This section contains targets for setting up the development environment

# Target to create a Python virtual environment if it doesn't exist
# This is an internal target not meant to be called directly by users
$(VENV_MARKER):
	@printf "$(BLUE)Setting up virtual environment...$(RESET)\n"
	@if ! command -v uv >/dev/null 2>&1; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv venv --python 3.12  # Create a Python 3.12 virtual environment using uv
	@touch $(VENV_MARKER)   # Create a marker file to indicate successful installation

install: $(VENV_MARKER) ## Install all dependencies using uv
	@printf "$(BLUE)Installing dependencies...$(RESET)\n"
	@uv sync --dev --frozen --all-extras  # Install dependencies from pyproject.toml with all extras
	@uv pip install pre-commit pytest pytest-cov marimo  # Install additional development tools

##@ Code Quality
# This section contains targets for code formatting and linting

fmt: install ## Run code formatting and linting
	@printf "$(BLUE)Running formatters and linters...$(RESET)\n"
	@uvx pre-commit install          # Install pre-commit hooks into the git repository
	@uvx pre-commit run --all-files  # Run all pre-commit hooks on all files

ty: install
	@uvx ty check


##@ Testing
# This section contains targets for running tests and generating coverage reports

test: install ## Run all tests
	@printf "$(BLUE)Running tests...$(RESET)\n"
	@uv run pytest $(TEST_DIR)  # Run all tests in the test directory

coverage: install ## Run tests with coverage
	@printf "$(BLUE)Running tests with coverage...$(RESET)\n"
	# Generate coverage reports in both terminal and HTML formats
	@uv run pytest --cov=src/cvx --cov-report=term --cov-report=html $(TEST_DIR) -k "not test_readme_path_not_found"
	@printf "$(BLUE)HTML report generated at $(BOLD)htmlcov/index.html$(RESET)\n"
	# Ensure code coverage is at least 100% for the main code (excluding tests)
	@uv run pytest --cov=src/cvx --cov-fail-under=100 $(TEST_DIR) -k "not test_readme_path_not_found"

##@ Marimo & Jupyter
# This section contains targets for working with Marimo notebooks

marimo: install ## Start a Marimo server
	@printf "$(BLUE)Starting Marimo server...$(RESET)\n"
	@uv run marimo edit $(MARIMO_DIR)  # Start a Marimo server for editing notebooks

##@ Cleanup
# This section contains targets for cleaning up generated files

clean: ## Clean generated files and directories
	@printf "$(BLUE)Cleaning project...$(RESET)\n"
	@git clean -d -X -f  # Remove all untracked files and directories that are ignored by git

##@ Help
# This section contains the help target that displays available commands

help: ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	# Parse the Makefile to extract targets and their descriptions
	# - Looks for lines with pattern: target: ## description
	# - Also processes section headers marked with ##@
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
