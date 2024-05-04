#!/bin/bash
pipx install poetry
poetry config virtualenvs.in-project true
poetry install
