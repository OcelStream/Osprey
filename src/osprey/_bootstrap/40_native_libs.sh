#!/usr/bin/env bash
#
# Stage 40 — native libraries.
#
# Osprey's custom nvinfer parsers (nvdsinfer_yolo_det/seg, nvdsinfer_rtdetr)
# and the IPC meta (de)serializer ship PRECOMPILED with the wheel, built for
# DeepStream 8.0 / CUDA 12.8 — the same versions this bootstrap installs — and
# the packaged pgie configs reference them by relative path. Nothing needs to
# be compiled or downloaded here for the standard pipeline.
#
set -euo pipefail

echo "[40] Precompiled parsers + meta serializer ship with the wheel — nothing to build."

# --- Sanity-check the shipped .so are present in the installed package ---
if python3 - <<'PY'
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("osprey")
if spec is None or not spec.submodule_search_locations:
    sys.exit("osprey package not importable")
root = pathlib.Path(list(spec.submodule_search_locations)[0])
need = [
    root / "server" / "deepstream" / "lib" / "nvdsinfer_yolo_det.so",
    root / "server" / "deepstream" / "lib" / "serialize_meta.so",
    root / "client" / "lib" / "deserialize_meta.so",
]
missing = [str(p) for p in need if not p.exists()]
if missing:
    sys.exit("missing shipped .so:\n  " + "\n  ".join(missing))
print("[40] shipped native libs present:", ", ".join(p.name for p in need))
PY
then
    :
else
    echo "[40] NOTE: run this stage after 'pip install osprey' so the shipped .so exist."
fi

# --- Optional, MANUAL: TensorRT end-to-end-NMS plugin patch ---
# Models whose ONNX embeds end-to-end NMS (EfficientNMS_TRT) need a TensorRT
# plugin library carrying those ops. Osprey does NOT automate this: it would
# mean overwriting your system TensorRT library with an externally-hosted
# binary. If you need it, do it deliberately, from a source YOU trust, and
# back up the original first. Reference recipe:
#   deepstream-8.0-nvfd/docker_image/patch_libnvinfer.sh
cat <<'NOTE'
[40] Standard YOLO/RT-DETR detection & segmentation work with the shipped
     parsers and need NO TensorRT plugin patch.
     Only ONNX models with embedded end-to-end NMS (EfficientNMS_TRT) require
     a patched libnvinfer_plugin. That patch overwrites a system TensorRT
     library, so Osprey leaves it as a deliberate manual step — see the
     reference recipe patch_libnvinfer.sh. Do it only with a binary you trust.
NOTE

# --- Final self-check: gi + pyds + osprey must import in THIS interpreter ---
# Soft (never fails the bootstrap) but surfaces the common "works in system
# python, missing in a venv" trap with an actionable message.
if python3 -m osprey.doctor; then
    :
else
    echo "[40] NOTE: some runtime imports are missing in this interpreter."
    echo "     Run 'osprey-doctor' with the SAME python you'll use to run Osprey."
fi

echo "[40] Done."
