#!/bin/bash
# Expose ONLY torch / PyG / flash-attn (and their compiled companions) from the
# container to the uv venv, instead of the whole /opt/venv site-packages.
#
# Must run INSIDE the container, from the project directory, after `uv venv`/`uv sync`:
#   singularity exec --rocm "$TORCH_SIF" env -u PYTHONPATH -u BASH_ENV \
#       bash --noprofile --norc -c 'cd "$PROJECT_DIR" && bash ./make_container_shim.sh'
set -euo pipefail
shopt -s nullglob

CSP=/opt/venv/lib/python3.12/site-packages
VENV=${VENV:-$PWD/.venv}
SHIM=${SHIM:-$PWD/.container-site}

# --- safety checks: never touch the existing shim if we are in the wrong place
if [ ! -d "$CSP/torch" ]; then
  echo "ERROR: $CSP/torch not found - run this script INSIDE the container." >&2
  exit 1
fi
if ! "$VENV/bin/python" -c 'pass' 2>/dev/null; then
  echo "ERROR: $VENV/bin/python does not run here (wrong place or broken venv)." >&2
  exit 1
fi

PKGS=(
  torch torchgen functorch
  torchvision torchaudio
  triton pytorch_triton_rocm
  torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
  flash_attn flash_attn_2_cuda
)

# --- build into a temp dir, then swap (an interrupted run keeps the old shim)
TMP="$SHIM.tmp.$$"
rm -rf "$TMP"
mkdir -p "$TMP"
for pkg in "${PKGS[@]}"; do
  found=0
  for p in "$CSP/$pkg" "$CSP/$pkg".libs "$CSP/$pkg"-*.dist-info "$CSP/$pkg"*.so; do
    [ -e "$p" ] || continue
    ln -sfn "$p" "$TMP/$(basename "$p")"
    found=1
  done
  [ $found = 1 ] && echo "linked   $pkg" || echo "skipped  $pkg (not in container)"
done
rm -rf "$SHIM"
mv "$TMP" "$SHIM"

SP=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
rm -f "$SP/_container_site.pth"          # old: whole /opt/venv
echo "$SHIM" > "$SP/_container_torch.pth" # new: selected packages only
echo "wrote $SP/_container_torch.pth -> $SHIM"

# --- verify
"$VENV/bin/python" - <<'EOF'
import importlib.util, os
import torch
print("torch     ", torch.__version__, os.path.realpath(torch.__file__))
for m in ("torch_geometric", "flash_attn"):
    s = importlib.util.find_spec(m)
    print(f"{m:<10}", os.path.realpath(s.origin) if s else "NOT AVAILABLE")
EOF
