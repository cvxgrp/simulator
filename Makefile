# Colors for pretty output
BLUE  := \033[36m
BOLD  := \033[1m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: help install fmt test coverage marimo clean

# Paths
VENV_MARKER  := .venv/.installed
TEST_DIR     := src/tests
SOURCE_DIR   := src
MARIMO_DIR   := book/marimo

##@ Development Setup

$(VENV_MARKER):
	@printf "$(BLUE)Setting up virtual environment...$(RESET)\n"
	@if ! command -v uv >/dev/null 2>&1; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv venv --python 3.12
	@touch $(VENV_MARKER)

install: $(VENV_MARKER) ## Install all dependencies using uv
	@printf "$(BLUE)Installing dependencies...$(RESET)\n"
	@uv sync --dev --frozen --all-extras
	@uv pip install pre-commit pytest pytest-cov marimo

##@ Code Quality

fmt: install ## Run code formatting and linting
	@printf "$(BLUE)Running formatters and linters...$(RESET)\n"
	@uv run pre-commit install
	@uv run pre-commit run --all-files

##@ Testing

test: install ## Run all tests
	@printf "$(BLUE)Running tests...$(RESET)\n"
	@uv run pytest $(TEST_DIR)

coverage: install ## Run tests with coverage
	@printf "$(BLUE)Running tests with coverage...$(RESET)\n"
	@uv run pytest --cov=$(SOURCE_DIR) --cov-report=term --cov-report=html $(TEST_DIR)
	@printf "$(BLUE)HTML report generated at $(BOLD)htmlcov/index.html$(RESET)\n"
	@uv run pytest --cov=$(SOURCE_DIR) --cov-fail-under=90 $(TEST_DIR)

##@ Marimo & Jupyter

marimo: install ## Start a Marimo server
	@printf "$(BLUE)Starting Marimo server...$(RESET)\n"
	@uv run marimo edit $(MARIMO_DIR)

##@ Cleanup

clean: ## Clean generated files and directories
	@printf "$(BLUE)Cleaning project...$(RESET)\n"
	@git clean -d -X -f

##@ Help

help: ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
