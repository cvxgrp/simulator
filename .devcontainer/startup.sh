#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync -vv --frozen
# install marimo
uv pip install --no-cache-dir marimo
