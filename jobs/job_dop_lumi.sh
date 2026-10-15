#!/bin/bash
#SBATCH --job-name=anemoi_train
#SBATCH --account=project_465003270
#SBATCH --partition=dev-g        # full-node GPU partition (use dev-g for short tests, max 3 h)
#SBATCH --nodes=2                     # 2 nodes x 8 GCDs = 16 GPUs; scale as needed
#SBATCH --gpus-per-node=8             # LUMI-G node = 4x MI250X = 8 GCDs ("GPUs")
#SBATCH --ntasks-per-node=8           # one task (rank) per GCD
#SBATCH --cpus-per-task=7             # 7 usable cores per GCD (1 core/CCD reserved)
#SBATCH --mem=480G                    # all usable node memory
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out       # 'logs/' must exist before sbatch
#SBATCH --no-requeue

set -euo pipefail

# ---------------------------------------------------------------- paths
export TORCH_SIF="${TORCH_SIF:-/scratch/project_465003270/milemate/containers/pytorch-2.9.1-PYG-rocm-6.4.4-py-3.12.3-v1.0.sif}"
export PROJECT_DIR=/scratch/project_465003270/milemate/base-dop
export CONFIG_DIR=/scratch/project_465003270/milemate/base-dop/training_configs
export CONFIG_NAME=config_test.yaml #${CONFIG_NAME:-config}-> $CONFIG_DIR/config.yaml
export SCRATCH_DIR=/scratch/project_465003270/milemate

# ------------------------------------------- container bindings (Slingshot/libfabric)
# Gives the container access to the host libfabric/CXI libs needed by the
# aws-ofi-rccl plugin for fast multi-node RCCL. Harmless on a single node.
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

# ---------------------------------------------------------- rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# ------------------------------------------------------- RCCL / network
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3     # Slingshot interfaces
export NCCL_NET_GDR_LEVEL=PHB                      # GPU-direct RDMA
export CXI_FORK_SAFE=1                             # dataloader workers fork()
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
# export NCCL_DEBUG=INFO                           # check for "NET/OFI" (plugin in use)

# ------------------------------------------------------------- PyTorch memory allocation
#export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
#export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF

# ------------------------------------------------------------- misc
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
unset PYTHONPATH PYTHONHOME || true

# CPU masks matching the LUMI-G GCD <-> CCD topology (local rank i -> GCD i)
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"

mkdir -p "$SCRATCH_DIR"
cd "$PROJECT_DIR"

echo "Job $SLURM_JOB_ID: $SLURM_NNODES nodes, $SLURM_NTASKS ranks, master $MASTER_ADDR"

# Strip PYTHONPATH (set by the container) and BASH_ENV (host Lmod init) with
# `env -u` BEFORE bash starts, so neither can leak into the venv.
# Bind mounts come from singularity-AI-bindings, so no extra -B is needed.
srun --cpu-bind="$CPU_BIND" \
    singularity exec --rocm "$TORCH_SIF" \
    env -u PYTHONPATH -u PYTHONHOME -u BASH_ENV \
    bash --noprofile --norc -c '
        set -euo pipefail
        export VIRTUAL_ENV=$PROJECT_DIR/.venv
        export PATH=$VIRTUAL_ENV/bin:$PATH

        # fail fast if the wrong Lightning is picked up (all ranks check, rank 0 prints)
        "$VIRTUAL_ENV/bin/python" - <<PYEOF
import os, sys, pytorch_lightning as pl, anemoi.utils
rank = os.environ.get("SLURM_PROCID")
venv = os.path.realpath("$VIRTUAL_ENV")
if rank == "0":
    print("PYTHONPATH   =", os.environ.get("PYTHONPATH"))
    print("lightning    =", pl.__version__, os.path.realpath(pl.__file__))
    print("anemoi.utils =", os.path.realpath(anemoi.utils.__file__))
if not os.path.realpath(pl.__file__).startswith(venv):
    sys.exit(f"rank {rank}: pytorch_lightning from {pl.__file__}, expected {venv}")
PYEOF

        # per-rank MIOpen cache on node-local /tmp (avoids Lustre lock contention)
        export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}-${SLURM_PROCID}
        export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
        mkdir -p "$MIOPEN_USER_DB_PATH"

        exec "$VIRTUAL_ENV/bin/anemoi-training" train \
            --config-dir="$CONFIG_DIR" \
            --config-name="$CONFIG_NAME" \
            system.hardware.accelerator=cuda \
            system.hardware.num_gpus_per_node=$SLURM_GPUS_PER_NODE \
            system.hardware.num_nodes=$SLURM_NNODES
    '
