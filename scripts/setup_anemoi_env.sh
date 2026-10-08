#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$PROJECT_DIR/anemoi-env"

: "${SCRATCH:?Set SCRATCH to a writable ATOS directory first}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH%/}/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCRATCH%/}/uv-python}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    uv python install 3.11
    uv venv --python 3.11 --python-preference only-managed "$ENV_DIR"
fi

ENV_PYTHON="$ENV_DIR/bin/python"
PYTHON_INCLUDE_DIR="$("$ENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ ! -f "$PYTHON_INCLUDE_DIR/Python.h" ]]; then
    printf 'Python headers not found: %s/Python.h\n' "$PYTHON_INCLUDE_DIR" >&2
    exit 1
fi

UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync \
    --locked \
    --all-groups \
    --python "$ENV_PYTHON"

"$ENV_PYTHON" - <<'PY'
from importlib import import_module
from pathlib import Path

base_module = import_module("anemoi.models.models.base")
base_path = Path(base_module.__file__)
source = base_path.read_text()
stale = "self._graph_name_hidden = model_config.model.model.hidden_nodes_name"
fixed = "self._graph_name_hidden = model_config.model.hidden_nodes_name"

if stale in source:
    base_path.write_text(source.replace(stale, fixed, 1))
elif fixed not in source:
    raise SystemExit(f"Expected hidden-node config access not found in {base_path}")
PY

"$ENV_PYTHON" -c 'import torch; from anemoi.models.data import Batch; print("Python headers, torch", torch.__version__, "and Anemoi Batch import OK")'

if [[ "${VIRTUAL_ENV:-}" == "$PROJECT_DIR/.venv" || "${VIRTUAL_ENV:-}" == "$PROJECT_DIR/.venv-triton" ]]; then
    printf 'Deactivate the old project environment before cleanup: %s\n' "$VIRTUAL_ENV" >&2
    exit 1
fi

for OLD_ENV in "$PROJECT_DIR/.venv" "$PROJECT_DIR/.venv-triton"; do
    if [[ -d "$OLD_ENV" && ! -L "$OLD_ENV" ]]; then
        rm -rf -- "$OLD_ENV"
        printf 'Removed old project environment: %s\n' "$OLD_ENV"
    fi
done