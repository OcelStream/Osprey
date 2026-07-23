#!/usr/bin/env bash
#
# Osprey end-to-end bare-metal bootstrap.
#
# Turns a fresh Ubuntu 24.04 host into a working Osprey box WITHOUT Docker, by
# reproducing what the DeepStream 8.0 container image build did:
#     00  system apt build + GStreamer runtime deps
#     10  NVIDIA driver R570 + CUDA 12.8 + cuDNN 9.7 + TensorRT 10.9
#     20  DeepStream 8.0 SDK (.deb from NGC)  → the GStreamer plugins
#     30  build + install pyds (Python bindings)
#     40  native libs (shipped precompiled; optional TRT-plugin patch is manual)
#
# Usage (needs root for apt/dpkg/ldconfig):
#     sudo osprey-bootstrap
#     sudo OSPREY_ASSUME_CUDA=1 osprey-bootstrap      # skip stage 10
#     sudo OSPREY_ONLY=30 osprey-bootstrap            # run a single stage
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Preflight ---
if [[ "$(id -u)" -ne 0 ]]; then
    echo "ERROR: this bootstrap installs system packages — run as root (sudo)." >&2
    exit 1
fi

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${VERSION_ID:-}" != "24.04" ]]; then
        echo "WARNING: DeepStream 8.0 targets Ubuntu 24.04; detected ${PRETTY_NAME:-unknown}." >&2
    fi
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "WARNING: this bootstrap is wired for x86_64/dGPU; detected $(uname -m)." >&2
fi

STAGES=(00_system_deps 10_cuda_trt_cudnn 20_deepstream_sdk 30_pyds 40_native_libs)

run_stage() {
    local name="$1"
    local script="${HERE}/${name}.sh"
    [[ -f "$script" ]] || { echo "ERROR: missing stage script ${script}" >&2; exit 1; }
    echo ""
    echo "=================================================================="
    echo ">>> Osprey bootstrap stage: ${name}"
    echo "=================================================================="
    # Fail the whole bootstrap loudly if a stage fails — never report success
    # with a half-installed stack (e.g. DeepStream up but pyds never built).
    if ! bash "$script"; then
        echo "" >&2
        echo "XXX Osprey bootstrap FAILED at stage: ${name} (exit $?)" >&2
        exit 1
    fi
}

if [[ -n "${OSPREY_ONLY:-}" ]]; then
    for s in "${STAGES[@]}"; do
        if [[ "$s" == "${OSPREY_ONLY}"* ]]; then run_stage "$s"; fi
    done
else
    for s in "${STAGES[@]}"; do run_stage "$s"; done
fi

echo ""
echo "=================================================================="
echo "Osprey bootstrap complete."
echo "  Next:  pip install ospreyai   (if not already installed)"
echo "         osprey-server     # FastAPI control plane on :8000"
echo "         osprey-client     # discovers sockets, serves RTSP"
echo "If the NVIDIA driver was (re)installed in stage 10, REBOOT first."
echo "=================================================================="
