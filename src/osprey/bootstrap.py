"""``osprey-bootstrap`` console entry point.

Thin launcher that runs the packaged ``_bootstrap/bootstrap.sh`` orchestrator,
which reproduces the DeepStream 8.0 image build on a bare-metal Ubuntu 24.04
host (system deps → CUDA/TRT/cuDNN → DeepStream SDK → pyds → native libs).

The heavy lifting lives in the shell scripts so users can also inspect and run
them directly. All environment toggles (OSPREY_ASSUME_CUDA, OSPREY_ONLY,
OSPREY_DS_VERSION, …) are read by those scripts and pass straight through.
"""

from __future__ import annotations

import os
import sys
from importlib import resources


def _bootstrap_script() -> str:
    """Return a filesystem path to the packaged bootstrap.sh."""
    # as_file materializes the resource on disk (works from wheels/zips too).
    with resources.as_file(
        resources.files("osprey._bootstrap") / "bootstrap.sh"
    ) as p:
        return str(p)


def main() -> None:
    script = _bootstrap_script()
    # Replace this process with bash so signals/exit codes propagate cleanly.
    argv = ["bash", script, *sys.argv[1:]]
    try:
        os.execvp("bash", argv)
    except FileNotFoundError:
        sys.exit("osprey-bootstrap: 'bash' not found on PATH.")


if __name__ == "__main__":
    main()
