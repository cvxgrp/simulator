# CLAUDE.md

Guidance for working in this repository.

## What this is

`cvxsimulator` — a small backtest simulator for investors, part of the
[cvxgrp](https://github.com/cvxgrp) ecosystem. It takes a price frame and a
sequence of positions and reports the resulting portfolio, rather than trying to
be a trading system. Runtime dependencies are `pandas`, `polars`, `pyarrow`,
`numpy` and `jquantstats`.

The package lives under `src/cvx/simulator/` (there is deliberately no
`src/cvx/__init__.py` — `cvx` is a namespace shared with sibling projects):

- `builder.py` — the incremental construction path: iterate the price index,
  set positions or cash per timestamp, then `build()` a portfolio.
- `portfolio.py` — the finished, immutable result and its reporting surface.
- `state.py` — the per-timestamp view the builder exposes while iterating
  (prices, holdings, cash, NAV).
- `_analytics.py` — the private return/NAV maths behind the two above.
- `utils/interpolation.py` — the interpolation and validity helpers used to
  handle gaps in a price frame.

`book/` holds the marimo notebooks, `web/` the static assets they publish.

## Ownership: locally owned vs Rhiza-managed

This repo syncs its dev infrastructure from the
[`jebel-quant/rhiza`](https://github.com/jebel-quant/rhiza) template. The pinned
version lives in `.rhiza/template.yml` (`ref:`), and `/rhiza:update` re-applies
the template. **The authoritative, machine-generated list of synced files is the
`files:` block of `.rhiza/template.lock`** — when in doubt, consult it. The split
below summarises it.

### Locally owned — edit these freely

- `src/` — the library source
- `tests/` — the test suite
- `pyproject.toml` — project metadata, dependency groups, tool config, and the
  `[tool.rhiza-task]` table that configures the gates
- `README.md`, `CHANGELOG.md`, `mkdocs.yml`, `CLAUDE.md`
- `book/`, `web/` — notebooks and the assets they publish
- `.rhiza/template.yml` — the template pin and the `profiles:`/`templates:`
  selection. The one file under `.rhiza/` this repo owns.
- `local.mk` — repo-specific make targets. The `Makefile` `-include`s it, and the
  template deliberately does not ignore it.

### Rhiza-managed — do NOT edit in place; fix upstream

These are overwritten by the next sync. To change one, open a PR against
`jebel-quant/rhiza` (or exclude the path in `.rhiza/template.yml`), then re-sync:

- `.github/workflows/rhiza_*.yml` — all CI/CD workflows
- `.github/` scaffolding — `dependabot.yml`, `release.yml`, rulesets,
  `secret_scanning.yml`. `CONFIG.md` is excluded in `template.yml` rather than
  synced.
- `Makefile` — a 71-line shim that pins `RHIZA_TASK` and forwards every unmatched
  target to that CLI. Nothing goes below it; the next sync overwrites whatever was
  appended. Repo targets belong in `local.mk`.
- `.pre-commit-config.yaml`, `ruff.toml`, `pytest.ini`, `.bandit`,
  `.editorconfig`, `.python-version`, `cliff.toml` — tooling config
- `.devcontainer/` — the container definition
- `LICENSE`, `SECURITY.md`, and the synced `docs/` pages

`SECURITY.md` in particular is synced here: an edit to it is drift the next sync
reverts, and the `check-managed-files` pre-commit hook refuses the commit.

## Quality gates

Since rhiza v1.4 the gates are tasks in the pinned `rhiza-task` CLI rather than
synced make fragments. Run them as bare `make <target>` (the shim forwards to
`uvx rhiza-task <task>`) — never call `.venv/bin/...` directly. `make help` lists
every task the pinned CLI knows, plus anything `local.mk` adds.

- `make install` — create the venv and sync dependencies
- `make fmt` — the pre-commit hooks over all files
- `make typecheck` — `ty` **and** `mypy`, because `[tool.rhiza-task]` sets
  `typechecker = "both"`
- `make test` — the full pytest suite with the coverage gate
- `make coverage` — coverage measurement into `_tests/coverage.xml`
- `make docs-coverage` — interrogate docstring coverage
- `make deps` — deptry unused/missing dependency analysis
- `make security` — the bandit scan
- `make license` — fail on GPL/LGPL/AGPL
- `make rhiza-test` — the rhiza repository checks
- `make all` — everything above, in CI's order

Do not reach for `make mutation`. The task still exists in the CLI, but rhiza
v1.5.0 stopped offering mutation testing (Jebel-Quant/rhiza#1492) and the recipe
drives a mutmut 2.x CLI that mutmut 3 removed.

## Conventions

- **Coverage must stay at 100%.** `[tool.rhiza-task]` sets
  `coverage-fail-under = 100`, above rhiza-task's default of 90. It moved there
  from `.rhiza/.env` when rhiza v1.4.2 retired the make layer — set it in
  `[tool.rhiza-task]`, not in `[tool.coverage.report]`, which the CLI outranks.
- CI runs the matrix on **ubuntu, macOS and Windows** (`ci-os-matrix` in
  `[tool.rhiza-task]`). Anything path- or newline-sensitive fails on Windows
  first, so prefer `pathlib` over string joins.
- `[tool.deptry.package_module_name_map]` maps distribution names to import
  names. A new dependency whose import name differs from its package name needs
  an entry, or `make deps` reports it as missing.
- The per-test timeout is 60s (`pytest-timeout`).
- Three markers are declared: `stress`, `property`, `kaleido`. Use them rather
  than inventing new ones, and deselect with `-m "not stress"`.

## Test layout

Tests are grouped **by behaviour, not by source module**, and that is deliberate:
`[tool.check_test_layout]` sets `enforce = false` with a recorded reason, so the
checker reports the layout as intentional rather than as violations. Do not
"fix" it by reshuffling the suite — per-module reach is guaranteed by the 100%
coverage gate instead of by file mirroring.

- `tests/test_builder.py`, `test_portfolio.py`, `test_state.py` and their
  `*_validation.py` siblings — the happy path and the input-validation path for
  each of the three core objects, split so a validation change does not churn the
  behavioural file.
- `tests/test_applications/test_reference/` — the Markowitz reference
  application, kept as an end-to-end check that the API composes.
- `tests/test_applications/test_talk/` — the five numbered experiments from the
  talk, each pinned against hashed price fixtures.
- `tests/test_doctests.py` — collects the docstring examples. `pytest.ini` does
  **not** pass `--doctest-modules`, so this file is what makes them run.
- `tests/test_polars.py` — the polars path alongside the pandas one.
- `tests/test_utils/` — interpolation and validity helpers.
- `tests/test_conftest.py`, `test_rhiza_packaging.py`, `test_version.py` —
  fixture and repo-level invariants.

CSV fixtures live in `tests/resources/` and per-application `resources/`
directories; shared fixtures in `tests/conftest.py`.
