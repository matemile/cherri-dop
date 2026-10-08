#!/bin/bash -l
#SBATCH --job-name=cherri-dop_atos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=dop_atos_%j.out
#SBATCH --no-requeue

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
ANEMOI_ENV="${ANEMOI_ENV:-$PROJECT_DIR/anemoi-env}"

: "${SCRATCH:?Set SCRATCH to a writable ATOS scratch directory before submitting}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export MLFLOW_ALLOW_FILE_STORE=true
mkdir -p "$SCRATCH"

source "$ANEMOI_ENV/bin/activate"
cd "$PROJECT_DIR"

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this job allocation")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

srun anemoi-training train \
    --config-name="config_test_atos.yaml" \
    --config-path="$PROJECT_DIR/training_configs"



