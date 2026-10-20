#!/bin/bash
#SBATCH --job-name=build_lumi
#SBATCH --account=project_465003270
#SBATCH --partition=dev-g          # GPU only needed for the smoke test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out    # 'logs/' must exist before sbatch
#SBATCH --no-requeue

set -euo pipefail

export TORCH_SIF="${TORCH_SIF:-/scratch/project_465003270/milemate/containers/pytorch-2.9.1-PYG-rocm-6.4.4-py-3.12.3-v1.0.sif}"
export PROJECT_DIR=/scratch/project_465003270/milemate/base-dop
export UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

export UV_CACHE_DIR=/scratch/project_465003270/milemate/.uv-cache
export UV_LINK_MODE=copy
export UV_PYTHON_DOWNLOADS=never
export UV_NO_PROGRESS=1
export UV_CONCURRENT_BUILDS=$SLURM_CPUS_PER_TASK
mkdir -p "$UV_CACHE_DIR"

srun singularity exec --rocm \
    -B /pfs,/scratch,/project,/projappl,/flash \
    "$TORCH_SIF" bash -c '
set -euo pipefail

# The container sets PYTHONPATH=/opt/venv/...; it must not leak into uv or the venv
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

cd "$PROJECT_DIR"

# 1) fresh venv on the container interpreter
rm -rf .venv .container-site
"$UV_BIN" venv --python /opt/venv/bin/python .venv

# 2) resolve + install (torch/PyG are overridden away in pyproject.toml)
"$UV_BIN" lock
"$UV_BIN" sync --all-groups --locked

# 3) expose ONLY torch/PyG from the container (replaces the old _container_site.pth)
bash ./scripts/make_container_shim.sh

# 4) test
#.venv/bin/python - <<EOF
#import importlib.util, torch, torch_geometric, pytorch_lightning as pl, torchmetrics
#import anemoi.training, anemoi.models, anemoi.graphs
#import anemoi.training.diagnostics.callbacks.weight_averaging
#venv = "$PROJECT_DIR/.venv"
#print("torch    ", torch.__version__, torch.__file__)
#print("hip      ", torch.version.hip, "| gpu:", torch.cuda.is_available(), torch.cuda.device_count())
#print("pyg      ", torch_geometric.__version__, torch_geometric.__file__)
#print("lightning", pl.__version__, pl.__file__)
#assert "/.container-site/" in torch.__file__ or torch.__file__.startswith("/opt/venv"), "torch not from container"
#assert pl.__file__.startswith(venv), "pytorch_lightning not from .venv"
#spec = importlib.util.find_spec("transformers")
#assert spec is None or spec.origin.startswith(venv), f"container transformers leaking: {spec.origin}"
#print("OK")
#EOF
'
