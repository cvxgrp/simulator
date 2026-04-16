# Marimo Notebooks

This repository contains interactive [Marimo](https://marimo.io/) notebooks.

## Available Notebooks

Notebooks live in `book/marimo/notebooks/` (configured via `MARIMO_FOLDER` in `.rhiza/.env`):

| Notebook | Description |
|---|---|
| `Balanced.py` | Balanced portfolio simulation |
| `monkey.py` | Random monkey portfolio |
| `OneAssetFadingOut.py` | Single asset fade-out analysis |
| `pairs.py` | Pairs trading simulation |

## Running the Notebooks

### Using the Makefile

From the repository root:

```bash
make marimo
```

This will start the Marimo server and open all notebooks in the notebooks folder as specified in `.rhiza/.env`.

### Validating Notebooks

To validate that all notebooks run without errors:

```bash
make marimo-validate
```

### Running a Specific Notebook

To run a single notebook:

```bash
marimo edit book/marimo/notebooks/monkey.py
```

### Using uv (Recommended)

The notebooks include inline dependency metadata, making them self-contained:

```bash
uv run book/marimo/notebooks/monkey.py
```

This will automatically install the required dependencies and run the notebook.

## Notebook Structure

Marimo notebooks are **pure Python files** (`.py`), not JSON. This means:

- Easy version control with Git
- Standard code review workflows
- No hidden metadata
- Compatible with all Python tools

Each notebook includes inline metadata that specifies its dependencies:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.1",
#     "cvxsimulator>=1.4.6",
# ]
# ///
```

## Configuration

Marimo is configured in `pyproject.toml` to properly import the local package:

```toml
[tool.marimo.runtime]
pythonpath = ["src"]
```

The notebook folder is configured in `.rhiza/.env`:

```env
MARIMO_FOLDER=book/marimo/notebooks
```

## CI/CD Integration

The `.github/workflows/rhiza_marimo.yml` workflow automatically:

1. Discovers all `.py` files in `book/marimo/notebooks/`
2. Runs each notebook in a fresh environment
3. Verifies that notebooks can bootstrap themselves
4. Ensures reproducibility

This guarantees that all notebooks remain functional and up-to-date.

## Creating New Notebooks

To create a new Marimo notebook:

1. Create a new `.py` file in the notebooks directory:
   ```bash
   marimo edit book/marimo/notebooks/my_notebook.py
   ```

2. Add inline metadata at the top:
   ```python
   # /// script
   # requires-python = ">=3.11"
   # dependencies = [
   #     "marimo==0.23.1",
   #     # ... other dependencies
   # ]
   # ///
   ```

3. Start building your notebook with cells

4. Test it runs in a clean environment:
   ```bash
   uv run book/marimo/notebooks/my_notebook.py
   ```

5. Commit and push — the CI will validate it automatically

## Learn More

- **Marimo Documentation**: [https://docs.marimo.io/](https://docs.marimo.io/)
- **Example Gallery**: [https://marimo.io/examples](https://marimo.io/examples)
- **Community Discord**: [https://discord.gg/JE7nhX6mD8](https://discord.gg/JE7nhX6mD8)

## Tips

- **Reactivity**: Remember that cells automatically re-run when their dependencies change
- **Pure Python**: Edit notebooks in any text editor, not just Marimo's UI
- **Git-Friendly**: Notebooks diff and merge like regular Python files
- **Self-Contained**: Use inline metadata to make notebooks reproducible
- **Interactive**: Take advantage of Marimo's rich UI components for better user experience
