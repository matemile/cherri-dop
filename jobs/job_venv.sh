#!/bin/bash -l
#SBATCH --job-name=venv_lumi
#SBATCH --account=project_465003270
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
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
uv venv -p 3.11
uv sync --all-groups
'
