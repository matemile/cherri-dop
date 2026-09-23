# CHERRI project and DOP framework environment using Anemoi

A reproducible Python environment of a DOP framework for CHERRI project purposes, managed with uv.

## Prerequisites

- uv
- Python 3.11, or permission for uv to download it
- For GPU execution: an appropriate NVIDIA driver and a PyTorch build matching the cluster policy

## Create or update the environment

```bash
uv sync --all-groups
```

The environment is created in `.venv/`.

## Run commands

```bash
uv run python scripts/check_install.py
uv run python -c "import anemoi.training; print('Anemoi import succeeded')"
```

## Activate interactively

```bash
source .venv/bin/activate
```

## Reproducible CI installation

```bash
uv sync --locked --all-groups
```

## Update dependencies

```bash
uv lock --upgrade
uv sync --all-groups
```

Review the `uv.lock` changes, test the environment, and commit both
`pyproject.toml` and `uv.lock`.
