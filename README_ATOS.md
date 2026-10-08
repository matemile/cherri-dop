# CHERRI-DOP on ATOS

This guide covers preparing the project environment and submitting the training job from an ATOS login node. Run training through Slurm; do not run GPU training directly on the login node.

## Prerequisites

## Set the cache and output paths

Run these commands in the shell where you will install and submit the job:

```bash
export SCRATCH=/ec/res4/scratch/sbaa
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export UV_PYTHON_INSTALL_DIR="$SCRATCH/uv-python"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "$UV_PYTHON_INSTALL_DIR"
```

To keep these variables across login sessions, add the exports to `~/.bashrc` once, then start a new shell or run `source ~/.bashrc`.

If uv reports that hardlinks are unavailable because the cache and project environment are on different filesystems, this is a performance warning, not a failed sync. To request copies explicitly, set:

```bash
export UV_LINK_MODE=copy
```

## Create the single project environment

From the repository root, create `anemoi-env/` with uv-managed Python 3.11, headers, and the locked dependencies:

```bash
cd /hpcperm/sbaa/anemoi/cherri-dop
bash scripts/setup_anemoi_env.sh
```

The setup checks for `Python.h`, synchronizes `uv.lock`, patches the pinned Anemoi model config accessor if needed, verifies PyTorch/Anemoi imports, and only then removes the old project-local `.venv/` and `.venv-triton/`. It does not touch `/ec/res4/hpcperm/sbaa/venvs/anemoi`.

The job and interactive commands should use `anemoi-env/`; activate it with `source anemoi-env/bin/activate`. The old `anemoi` environment may still be active in your shell, so deactivate it before activating `anemoi-env`.

## Dataset paths

The ATOS training config is `training_configs/config_test_atos.yaml`. It currently refers to:

- ERA5: `/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-n320-1979-2024-6h-v1-for-single-v2.zarr`
- Surface observations: `/ec/ai/project/ai-ml/datasets/dop-ea-ofb-0001-1979-2025-v4-combined-surface-observations.zarr`

Before submitting, check that both stores are readable from the compute nodes. The ERA5 store name ends in `2024`, while the config's test split is set to 2025; confirm that its metadata covers the requested dates or adjust the test period.

## Submit training

The training job requests one GPU, 8 CPUs, 128 GB of memory, and up to 12 hours. Set `SCRATCH` in the submitting shell, then submit from this repository:

```bash
export SCRATCH=/ec/res4/scratch/sbaa
sbatch --account=nonmifa1 --partition=gpu --qos=ng jobs/job_dop_atos.sh
```

Slurm writes the job log to `dop_atos_<job-id>.out` in the submission directory. The script uses `--no-requeue`, so an interrupted job will not be automatically resubmitted.
