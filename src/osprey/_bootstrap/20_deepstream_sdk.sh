#!/usr/bin/env bash
#
# Stage 20 — DeepStream 8.0 SDK.
#
# Installs the SDK from the NGC .deb. This is what provides the GStreamer
# plugins Osprey is built on (nvstreammux, nvinfer, nvtracker, nvunixfdsink/src,
# nvdsosd, nvv4l2h264enc) — none of which are on PyPI. The container image had
# the SDK pre-installed under /opt/nvidia/deepstream; here we install it.
#
set -euo pipefail

DS_VERSION="${OSPREY_DS_VERSION:-8.0}"
DS_PATH="/opt/nvidia/deepstream/deepstream-${DS_VERSION}"
DS_DEB_URL="${OSPREY_DS_DEB_URL:-https://api.ngc.nvidia.com/v2/resources/nvidia/deepstream/versions/${DS_VERSION}/files/deepstream-${DS_VERSION}_${DS_VERSION}.0-1_amd64.deb}"

if [[ -d "$DS_PATH" ]] && command -v gst-inspect-1.0 >/dev/null 2>&1 \
   && gst-inspect-1.0 nvstreammux >/dev/null 2>&1; then
    echo "[20] DeepStream ${DS_VERSION} already installed at ${DS_PATH} — skipping."
    exit 0
fi

echo "[20] Installing DeepStream ${DS_VERSION} SDK from: ${DS_DEB_URL}"

export DEBIAN_FRONTEND=noninteractive
work="$(mktemp -d)"
deb="${work}/deepstream-${DS_VERSION}.deb"
curl -fSL --retry 3 -o "$deb" "$DS_DEB_URL"

# apt-get (not dpkg -i) so the .deb's dependencies resolve from the repos
# configured in stage 10.
apt-get update
apt-get install -y "$deb"
rm -rf "$work"

# DeepStream ships a post-install script that pulls extra components and runs
# ldconfig. Run it when present.
if [[ -x "${DS_PATH}/install.sh" ]]; then
    echo "[20] Running DeepStream post-install (${DS_PATH}/install.sh)..."
    ( cd "$DS_PATH" && ./install.sh )
fi

# Put the DeepStream libs on the loader path so `import pyds` (and the plugins)
# resolve their .so deps without a manual LD_LIBRARY_PATH in every shell.
echo "${DS_PATH}/lib" > /etc/ld.so.conf.d/osprey-deepstream.conf
echo "/opt/nvidia/deepstream/deepstream/lib" >> /etc/ld.so.conf.d/osprey-deepstream.conf
ldconfig

# Verify the core plugin that everything depends on is now discoverable.
if gst-inspect-1.0 nvstreammux >/dev/null 2>&1; then
    echo "[20] DeepStream ${DS_VERSION} installed — nvstreammux is available."
else
    echo "[20] ERROR: nvstreammux not found after install. Check CUDA/TRT prerequisites (stage 10)."
    exit 1
fi
