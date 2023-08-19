#!/bin/bash
pipx install poetry
poetry config virtualenvs.in-project true
poetry install
poetry run pip install ipykernel pre-commit
poetry run pre-commit install
