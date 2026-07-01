#!/bin/bash
set -e

echo "[entrypoint] Building TensorRT engines..."
python3 /deepstream_app/deepstream/app/build_engines.py

echo "[entrypoint] Starting application..."
exec "$@"
