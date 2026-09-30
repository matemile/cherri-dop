#!/bin/bash -l
#SBATCH --job-name=dop_lumi_test
#SBATCH --account=project_465003270
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --no-requeue

set -euo pipefail

module load LUMI
module load CrayEnv
#source ~/.bashrc

export OMP_NUM_THREADS=6
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MAX_COUNT=0
export HYDRA_FULL_ERROR=1
#TORCH_SIF="${TORCH_SIF:-/scratch/project_465002688/pytorch-2.9.1-PYG-rocm-6.4.4-py-3.12.3-v1.0.sif}"
TORCH_SIF="${TORCH_SIF:-/scratch/project_465003270/milemate/containers/pytorch-2.9.1-PYG-rocm-6.4.4-py-3.12.3-v1.0.sif}"

CONFIG_DIR="/pfs/lustrep3/scratch/project_465003270/milemate/cherri-dop/training_configs"
SCRATCH="/pfs/lustrep3/scratch/project_465003270/milemate"

#srun singularity exec --rocm --bind /scratch/project_465002688/milemate:/scratch/project_465002688/milemate "$TORCH_SIF" bash -lc '
#
#python -I -m anemoi-training train --config-name=config_test.yaml --config-path="'"$CONFIG_DIR"'"
#
srun singularity exec --rocm -B /pfs:/pfs "$TORCH_SIF" bash -lc '
PROJECT_DIR=/pfs/lustrep3/scratch/project_465003270/milemate/cherri-dop
cd "$PROJECT_DIR"
set -x
pwd -P
ls -l .venv/bin/python .venv/bin/anemoi-training
head -n 1 .venv/bin/anemoi-training
readlink -f .venv/bin/python || true
.venv/bin/python -c "'"import sys; print(sys.executable); print(sys.prefix)"'"
.venv/bin/python .venv/bin/anemoi-training --help
source .venv/bin/activate

unset PYTHONPATH
unset PYTHONHOME
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export MLFLOW_ALLOW_FILE_STORE=true

echo "== cherri dop training start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
echo "== gpu env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} =="
echo "== cherri cli start $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
/pfs/lustrep3/scratch/project_465003270/milemate/cherri-dop/.venv/bin/anemoi-training train --config-name=config_test.yaml --config-path="'"$CONFIG_DIR"'" 
echo "== cherri dop training end   $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
'
