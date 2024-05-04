#!/bin/bash
pipx install poetry
poetry config virtualenvs.in-project false
poetry install
