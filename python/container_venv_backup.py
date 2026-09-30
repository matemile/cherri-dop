from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).absolute().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
source_roots = [
    PROJECT_ROOT,
    WORKSPACE_ROOT / "anemoi-core" / "models" / "src",
    WORKSPACE_ROOT / "anemoi-core" / "training" / "src",
    WORKSPACE_ROOT / "anemoi-core" / "graphs" / "src",
    WORKSPACE_ROOT / "anemoi-inference",
]

source_path_strings = [str(path) for path in source_roots if path.exists()]
sys.path[:0] = source_path_strings


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python container_venv_backup.py <module-or-script> [args...]")

    target = sys.argv[1]
    target_args = sys.argv[2:]

    started = time.perf_counter()
    import torch

    print(
        f"torch import seconds={time.perf_counter() - started:.2f} version={torch.__version__} file={torch.__file__}",
        flush=True,
    )

    try:
        import transformers

        print(
            f"transformers preload version={transformers.__version__} file={transformers.__file__}",
            flush=True,
        )
    except ImportError as exc:
        print(f"transformers preload skipped: {exc}", flush=True)

    fallback_site_packages = PROJECT_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
    if fallback_site_packages.exists():
        sys.path.insert(len(source_path_strings), str(fallback_site_packages))
        print(f"venv fallback site-packages={fallback_site_packages}", flush=True)

    sys.argv = [target, *target_args]
    if target.endswith(".py") or "/" in target:
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path
        runpy.run_path(str(target_path), run_name="__main__")
    else:
        runpy.run_module(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
