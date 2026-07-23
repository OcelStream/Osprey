#!/usr/bin/env bash
#
# Stage 10 — NVIDIA compute stack (driver + CUDA 12.8 + cuDNN 9.7 + TensorRT 10.9).
#
# This is the layer the NVIDIA container image provided for free; on bare metal
# it must be installed from NVIDIA's apt network repo. It is the heaviest and
# most fragile stage — version pins below match DeepStream 8.0's compatibility
# table (Ubuntu 24.04 / driver R570 / CUDA 12.8 / TRT 10.9.0 / cuDNN 9.7).
#
# Skip this whole stage if the host already has the stack:
#     OSPREY_ASSUME_CUDA=1 osprey-bootstrap
#
set -euo pipefail

if [[ "${OSPREY_ASSUME_CUDA:-0}" == "1" ]]; then
    echo "[10] OSPREY_ASSUME_CUDA=1 — skipping driver/CUDA/cuDNN/TensorRT install."
    exit 0
fi

echo "[10] Installing NVIDIA compute stack (driver/CUDA/cuDNN/TensorRT)..."

export DEBIAN_FRONTEND=noninteractive

# --- Add NVIDIA CUDA network repo for Ubuntu 24.04 / x86_64 (idempotent) ---
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    tmp_keyring="$(mktemp -d)/cuda-keyring.deb"
    curl -fSL --retry 3 -o "$tmp_keyring" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i "$tmp_keyring"
    rm -f "$tmp_keyring"
fi
apt-get update

# --- NVIDIA display driver R570 (skip if a working driver is already present) ---
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "[10] nvidia-smi already works — skipping driver install."
else
    echo "[10] Installing cuda-drivers-570 (a REBOOT is required afterward)."
    apt-get install -y cuda-drivers-570
fi

# --- CUDA 12.8 toolkit (skip if nvcc reports 12.8) ---
if command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -q "release 12.8"; then
    echo "[10] CUDA 12.8 toolkit already present — skipping."
else
    apt-get install -y cuda-toolkit-12-8
fi

# --- cuDNN 9.7.1 for CUDA 12 (PINNED to DeepStream 8.0's compat version) ---
# The -cuda-12 suffix keeps it off CUDA 13, but the version still floats to the
# newest 9.x (e.g. 9.24) unless pinned. Match the compat table exactly.
CUDNN_VER="${OSPREY_CUDNN_VERSION:-9.7.1.26-1}"
apt-get install -y "cudnn9-cuda-12=${CUDNN_VER}" \
  || apt-get install -y cudnn9-cuda-12 \
  || apt-get install -y libcudnn9-cuda-12

# --- TensorRT 10.9.0.34 for CUDA 12.8 (PINNED — do not let it float) ---
# DeepStream 8.0 requires TensorRT 10.9 on CUDA 12.8. Unlike the CUDA/cuDNN
# packages, the bare `tensorrt` metapackage has no CUDA-major suffix, so apt
# resolves it to the repo's NEWEST TRT — currently 11.x built for CUDA 13.3,
# which is INCOMPATIBLE with DeepStream 8.0 and pulls a conflicting CUDA 13
# runtime. Pin the exact version; override with OSPREY_TRT_VERSION if needed.
TRT_VER="${OSPREY_TRT_VERSION:-10.9.0.34-1+cuda12.8}"
# Pinning only the `tensorrt` metapackage fails — its dependency chain floats to
# the newest (11.x) and conflicts. Pin the whole TensorRT package family via an
# apt preference so the resolver installs a coherent 10.9 set.
cat > /etc/apt/preferences.d/osprey-tensorrt.pref <<EOF
Package: tensorrt* libnvinfer* python3-libnvinfer* libnvonnxparsers*
Pin: version ${TRT_VER}
Pin-Priority: 1001
EOF
if apt-get install -y tensorrt; then
    echo "[10] TensorRT ${TRT_VER} installed (pinned via apt preferences)."
else
    echo "[10] ERROR: could not install pinned TensorRT ${TRT_VER}."
    echo "     List available versions with:  apt-cache madison tensorrt"
    echo "     then re-run with OSPREY_TRT_VERSION=<exact-version>."
    exit 1
fi

echo "[10] NVIDIA compute stack installed."
echo "[10] NOTE: if the driver was (re)installed, REBOOT before continuing."
