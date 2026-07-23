#!/usr/bin/env bash
#
# Stage 30 — build & install pyds (DeepStream Python bindings).
#
# NVIDIA does not publish pyds to PyPI, and the wheel is specific to the host
# Python minor version + DeepStream version, so it must be built here. This
# mirrors the pyds layer of the DeepStream 8.0 image build exactly
# (docker_image/Dockerfile lines 19-37).
#
set -euo pipefail

DS_VERSION="${OSPREY_DS_VERSION:-8.0}"
DS_PATH="/opt/nvidia/deepstream/deepstream-${DS_VERSION}"
PIP="${OSPREY_PIP:-pip3}"

# Ubuntu 24.04's system Python is PEP-668 "externally managed" — pip refuses
# system-wide installs without this. Honors the same choice the user makes for
# `pip install osprey`. (`python3 -m build` uses its own isolated env, unaffected.)
export PIP_BREAK_SYSTEM_PACKAGES=1

# Skip only if pyds is a REAL compiled module. A bare `import pyds` can
# false-pass when a stray namespace-only 'pyds' directory is on sys.path (no
# __init__, PEP 420) — that leaves the host with NO working pyds while the
# check "succeeds". Require __file__ to be an actual .so.
if python3 -c "import pyds,sys; f=getattr(pyds,'__file__',None); sys.exit(0 if f and f.endswith('.so') else 1)" >/dev/null 2>&1; then
    echo "[30] pyds already installed (compiled module present) — skipping build."
    exit 0
fi
echo "[30] pyds not properly installed (missing, or namespace-only) — building."

echo "[30] Building pyds for DeepStream ${DS_VERSION}..."

"$PIP" install --no-cache-dir build

src_dir="${DS_PATH}/sources/deepstream_python_apps"
if [[ ! -d "$src_dir" ]]; then
    mkdir -p "${DS_PATH}/sources"
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git "$src_dir"
fi

cd "$src_dir"
git submodule update --init
python3 bindings/3rdparty/git-partial-submodule/git-partial-submodule.py restore-sparse

export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
PYMIN="$(python3 -c 'import sys; print(sys.version_info.minor)')"
export CMAKE_ARGS="-DDS_PATH=${DS_PATH} -DPYTHON_MINOR_VERSION=${PYMIN} -DDS_VERSION=${DS_VERSION}"

cd "${src_dir}/bindings"
python3 -m build

wheel="$(ls -1 "${src_dir}/bindings/dist/"pyds-*.whl 2>/dev/null | head -n1)"
if [[ -z "$wheel" ]]; then
    echo "[30] ERROR: pyds wheel was not produced."
    exit 1
fi
"$PIP" install "$wheel"

python3 -c "import pyds; print('[30] pyds installed OK')"
