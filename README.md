# CHERRI project and DOP framework environment using Anemoi

A reproducible Python environment of a DOP framework for CHERRI project purposes, managed with uv.

## Prerequisites

- uv
- Python 3.11, or permission for uv to download it
- For GPU execution: an appropriate NVIDIA driver and a PyTorch build matching the cluster policy

## Create or update the environment

On ATOS, set `SCRATCH` and create the project's single environment with:

```bash
export SCRATCH=/ec/res4/scratch/sbaa
bash scripts/setup_anemoi_env.sh
```

The environment is created in `anemoi-env/`. It uses uv-managed Python 3.11 with development headers, as required by Triton's runtime compiler.

## Run commands

```bash
source anemoi-env/bin/activate
uv run --active python scripts/check_install.py
uv run --active python -c "import anemoi.training; print('Anemoi import succeeded')"
```

## Activate interactively

```bash
source anemoi-env/bin/activate
```

## Reproducible CI installation

```bash
source anemoi-env/bin/activate
uv sync --active --locked --all-groups
```

## Update dependencies

```bash
source anemoi-env/bin/activate
uv lock --upgrade
uv sync --active --all-groups
```

Review the `uv.lock` changes, test the environment, and commit both
`pyproject.toml` and `uv.lock`.
