#!/usr/bin/env bash
#
# Stage 00 — system packages.
#
# Installs the apt build toolchain + GStreamer/GObject-introspection runtime
# that both the pyds build and the DeepStream pipeline need. Mirrors the apt
# layer of the DeepStream 8.0 image build (docker_image/Dockerfile) and adds
# the GStreamer runtime plugins DeepStream requires on bare metal (the image
# had these pre-installed).
#
set -euo pipefail

echo "[00] Installing system build + runtime dependencies..."

export DEBIAN_FRONTEND=noninteractive
apt-get update

# --- Build toolchain + Python/GI/GStreamer dev (pyds build needs these) ---
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    python3-gi python3-gst-1.0 python3-gi-cairo \
    gobject-introspection gir1.2-gstreamer-1.0 \
    git cmake g++ build-essential meson ninja-build pkg-config \
    libglib2.0-dev libglib2.0-dev-bin libgstreamer1.0-dev \
    libtool m4 autoconf automake libgirepository-2.0-dev libcairo2-dev \
    curl ca-certificates

# --- GStreamer runtime plugins + libs the DeepStream SDK depends on ---
# (present in the NVIDIA container image; must be installed explicitly on host)
# NOTE: libmosquitto1 is a runtime dep of the nvtracker low-level lib
# (libnvds_nvmultiobjecttracker.so). Without it, nvtracker fails to dlopen,
# the pipeline never leaves READY, and every /add is (misleadingly) rejected
# with "Pipeline is not running". It ships in the DS container image but not
# on a bare host, so install it explicitly here.
apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 gstreamer1.0-tools \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 \
    gir1.2-gst-rtsp-server-1.0 gir1.2-gst-plugins-base-1.0 \
    libjansson4 libyaml-cpp-dev libjsoncpp-dev \
    libssl3 libjpeg-turbo8 \
    libmosquitto1

echo "[00] System dependencies installed."
