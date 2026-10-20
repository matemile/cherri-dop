#!/bin/bash
#SBATCH --job-name=check_lumi
#SBATCH --account=project_465003270
#SBATCH --partition=dev-g          # GPU only needed for the smoke test; 'small'/'debug' works for install-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7          # LUMI-G: 7 cores per GCD
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out    # 'logs/' must exist BEFORE sbatch
#SBATCH --no-requeue

set -euo pipefail

export TORCH_SIF="${TORCH_SIF:-/scratch/project_465003270/milemate/containers/pytorch-2.9.1-PYG-rocm-6.4.4-py-3.12.3-v1.0.sif}"
export PROJECT_DIR=/scratch/project_465003270/milemate/base-dop
export UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"   # host uv binary; $HOME is auto-mounted

# uv settings (passed into the container through the environment)
export UV_CACHE_DIR=/scratch/project_465003270/milemate/.uv-cache   # keep the 20 GB home quota free
export UV_LINK_MODE=copy            # cache and venv may be on different filesystems
export UV_PYTHON_DOWNLOADS=never    # only ever use the container's Python
export UV_NO_PROGRESS=1
export UV_CONCURRENT_BUILDS=$SLURM_CPUS_PER_TASK
unset PYTHONPATH PYTHONHOME || true
mkdir -p "$UV_CACHE_DIR"

singularity exec -B /pfs,/scratch --rocm "$TORCH_SIF" env -u PYTHONPATH -u BASH_ENV \
  bash --noprofile --norc -c 'cd /scratch/project_465003270/milemate/base-dop && bash ./scripts/make_container_shim.sh'

#singularity exec --rocm "$TORCH_SIF" env -u PYTHONPATH -u BASH_ENV bash --noprofile --norc -c '
#  cd /scratch/project_465003270/milemate/base-dop
#  ls .container-site
#  cat .venv/lib/python3.12/site-packages/_container_torch.pth'

#singularity exec --rocm "$TORCH_SIF" /opt/venv/bin/python -c "import flash_attn; print(flash_attn.__version__)"

#singularity exec "$TORCH_SIF" bash -c 'echo "PYTHONPATH=$PYTHONPATH"'

#singularity exec -B /pfs,/scratch "$TORCH_SIF" bash -c '
#  cd /scratch/project_465003270/milemate/base-dop
#  ls .venv/lib/python3.12/site-packages | grep -i lightning
#  unset PYTHONPATH
#  .venv/bin/python -c "import pytorch_lightning as pl; print(pl.__version__, pl.__file__)"'
