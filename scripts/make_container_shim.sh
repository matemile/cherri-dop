#!/bin/bash
# Expose ONLY torch / PyG (and their compiled companions) from the container
# to the uv venv, instead of the whole /opt/venv site-packages.
# Run INSIDE the container, from the project directory, after `uv venv`:
#   singularity exec -B /pfs,/scratch "$TORCH_SIF" bash -c 'unset PYTHONPATH; cd "$PROJECT_DIR"; ./make_container_shim.sh'
set -euo pipefail
shopt -s nullglob

CSP=/opt/venv/lib/python3.12/site-packages
VENV=${VENV:-$PWD/.venv}
SHIM=${SHIM:-$PWD/.container-site}

PKGS=(
  torch torchgen functorch
  torchvision torchaudio
  triton pytorch_triton_rocm
  flash_attn flash_attn_2_cuda
  torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
)

rm -rf "$SHIM"
mkdir -p "$SHIM"
for pkg in "${PKGS[@]}"; do
  for p in "$CSP/$pkg" "$CSP/$pkg".libs "$CSP/$pkg"-*.dist-info "$CSP/$pkg"*.so; do
    [ -e "$p" ] || continue
    ln -s "$p" "$SHIM/$(basename "$p")"
    echo "linked $(basename "$p")"
  done
done

SP=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
rm -f "$SP/_container_site.pth"          # old: whole /opt/venv
echo "$SHIM" > "$SP/_container_torch.pth" # new: torch/PyG only
echo "wrote $SP/_container_torch.pth -> $SHIM"
