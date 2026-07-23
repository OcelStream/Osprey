#!/usr/bin/env python3
"""
TensorRT engine pre-builder for DeepStream.

Reads GIE_N_CONFIG env vars → nvinfer .txt configs → ONNX models → builds
missing engines via trtexec.

The Re-ID model for NvDeepSORT (.etlt) is intentionally left to DeepStream:
the NvDeepSORT tracker converts and caches it automatically on first run.

Exits 0 when all needed engines exist or were built successfully.
Exits 1 when any engine build failed.
"""

import configparser
import logging
import os
import re
import shutil
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# nvinfer network-mode → trtexec precision flag
_ONNX_PRECISION = {"0": [], "1": ["--int8"], "2": ["--fp16"]}

# trtexec is part of the TensorRT install but frequently NOT on PATH (it lives
# under /usr/src/tensorrt/bin). Resolve it explicitly so the pre-builder works
# on a bare host, not just inside a container that put it on PATH.
_TRTEXEC_CANDIDATES = (
    "/usr/src/tensorrt/bin/trtexec",
    "/opt/nvidia/deepstream/deepstream/bin/trtexec",
    "/usr/local/tensorrt/bin/trtexec",
)


def find_trtexec() -> str | None:
    """Return a usable trtexec path, or None if it can't be found.

    Order: ``$TRTEXEC`` override → ``PATH`` → known TensorRT install dirs.
    """
    env = os.environ.get("TRTEXEC")
    if env and os.path.isfile(env):
        return env
    on_path = shutil.which("trtexec")
    if on_path:
        return on_path
    for cand in _TRTEXEC_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------

def _get_onnx_input_info(onnx_file: str) -> tuple[str, bool]:
    """Return (input_name, is_dynamic) for the first input of the ONNX model.

    is_dynamic is True when any dimension is symbolic/None (dynamic axes).
    Falls back to ('images', True) if the model cannot be inspected.
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-c",
                (
                    "import onnxruntime as rt, json; "
                    f"s = rt.InferenceSession('{onnx_file}', "
                    "providers=['CPUExecutionProvider']); "
                    "inp = s.get_inputs()[0]; "
                    "print(json.dumps({'name': inp.name, 'shape': inp.shape}))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout.strip())
            name = data.get("name", "images") or "images"
            shape = data.get("shape", [])
            # A dimension is dynamic when it is None or a non-integer string
            is_dynamic = any(
                d is None or not str(d).lstrip("-").isdigit()
                for d in shape
            )
            return name, is_dynamic
    except Exception:
        pass
    logger.debug(
        "Could not inspect ONNX inputs for %s — assuming dynamic, name='images'",
        onnx_file,
    )
    return "images", True


def _make_onnx_batch_dynamic(onnx_file: str) -> str | None:
    """Return path to a temp ONNX copy with a dynamic first (batch) dimension.

    Uses the `onnx` package to clear the static dim_value on input[0].dim[0]
    and replace it with the symbolic name 'batch'.  Returns None on failure.
    """
    tmp_path = onnx_file + ".dynbatch.onnx"
    result = subprocess.run(
        [
            sys.executable, "-c",
            (
                "import onnx; "
                f"m = onnx.load('{onnx_file}'); "
                "d = m.graph.input[0].type.tensor_type.shape.dim[0]; "
                "d.ClearField('dim_value'); "
                "d.dim_param = 'batch'; "
                f"onnx.save(m, '{tmp_path}')"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0 and os.path.isfile(tmp_path):
        return tmp_path
    logger.warning(
        "Could not patch ONNX batch dimension: %s",
        result.stderr.strip() or result.stdout.strip(),
    )
    return None


def build_onnx_engine(
    onnx_file: str,
    engine_file: str,
    batch_size: str,
    network_mode: str,
    gpu_id: str,
    infer_dims: str,
) -> bool:
    """Build a TensorRT engine from an ONNX model using trtexec."""
    if not os.path.isfile(onnx_file):
        logger.error("ONNX file not found: %s", onnx_file)
        return False

    os.makedirs(os.path.dirname(os.path.abspath(engine_file)), exist_ok=True)

    input_name, is_dynamic = _get_onnx_input_info(onnx_file)

    # infer-dims from nvinfer config uses semicolons: "3;640;640" → "3x640x640"
    spatial = infer_dims.replace(";", "x") if infer_dims else "3x640x640"

    # Static ONNX models have a fixed batch=1 baked in.  DeepStream nvinfer
    # requires the engine to support the configured batch-size (e.g. 10).
    # Patch the ONNX to make the batch axis dynamic, then build with shape
    # range flags so TensorRT produces a multi-profile engine.
    tmp_onnx = None
    onnx_to_build = onnx_file
    if not is_dynamic:
        logger.info(
            "  Static ONNX — patching batch dim to dynamic for batch-size=%s",
            batch_size,
        )
        tmp_onnx = _make_onnx_batch_dynamic(onnx_file)
        if tmp_onnx:
            onnx_to_build = tmp_onnx
            is_dynamic = True
        else:
            logger.warning(
                "  Patch failed — building as-is (engine will only support batch=1)"
            )

    trtexec = find_trtexec()
    if trtexec is None:
        logger.error(
            "trtexec not found (looked on PATH, $TRTEXEC, and %s) — cannot "
            "pre-build %s. Install TensorRT or set $TRTEXEC.",
            ", ".join(_TRTEXEC_CANDIDATES), onnx_file,
        )
        return False

    cmd = [
        trtexec,
        f"--onnx={onnx_to_build}",
        f"--saveEngine={engine_file}",
        f"--device={gpu_id}",
    ]

    if is_dynamic:
        cmd += [
            f"--minShapes={input_name}:1x{spatial}",
            f"--optShapes={input_name}:{batch_size}x{spatial}",
            f"--maxShapes={input_name}:{batch_size}x{spatial}",
        ]

    cmd += _ONNX_PRECISION.get(network_mode, [])

    logger.info("Building ONNX engine: %s → %s", onnx_file, engine_file)
    logger.info("  cmd: %s", " ".join(cmd))

    ret = subprocess.run(cmd).returncode

    if tmp_onnx and os.path.isfile(tmp_onnx):
        os.remove(tmp_onnx)

    if ret != 0:
        logger.error("trtexec failed (exit %d) for %s", ret, onnx_file)
        return False

    logger.info("Engine ready: %s", engine_file)
    return True


# ---------------------------------------------------------------------------
# Config readers
# ---------------------------------------------------------------------------

def parse_gie_config(path: str) -> dict:
    """Parse a nvinfer .txt config file and return relevant build params."""
    parser = configparser.ConfigParser()
    parser.read(path)
    p = dict(parser["property"]) if parser.has_section("property") else {}
    return {
        "onnx_file":    p.get("onnx-file", ""),
        "engine_file":  p.get("model-engine-file", ""),
        "batch_size":   p.get("batch-size", "1"),
        "network_mode": p.get("network-mode", "0"),
        "gpu_id":       p.get("gpu-id", "0"),
        "infer_dims":   p.get("infer-dims", ""),  # e.g. "3;640;640"
    }


def prebuild_engines(config_paths: list[str]) -> int:
    """Pre-build any missing engines for an explicit list of GIE config paths.

    This is the programmatic entry point used by the server at startup
    (``osprey.configure(gie_config=…)`` sets the configs directly rather than
    through ``GIE_N_CONFIG`` env vars). For each config it builds the ONNX to
    the config's ``model-engine-file`` path **only if that file is missing**,
    so nvinfer loads a ready engine instead of building one mid-startup (which
    would race the readiness timeout and, because nvinfer auto-names its own
    output, rebuild on every run).

    Returns the number of engines that failed to build (0 = all good/skipped).
    Best-effort: a failure is logged and left for nvinfer to retry, never raised.
    """
    errors = 0
    for config_path in config_paths:
        if not config_path or not os.path.isfile(config_path):
            logger.warning("GIE config not found, skipping pre-build: %s", config_path)
            continue
        params = parse_gie_config(config_path)
        onnx, engine = params["onnx_file"], params["engine_file"]
        if not onnx or not engine:
            logger.info(
                "GIE %s: no onnx-file/model-engine-file — leaving engine to nvinfer",
                config_path,
            )
            continue
        if os.path.isfile(engine):
            logger.info("Engine present, skip pre-build: %s", engine)
            continue
        ok = build_onnx_engine(
            onnx_file=onnx,
            engine_file=engine,
            batch_size=params["batch_size"],
            network_mode=params["network_mode"],
            gpu_id=params["gpu_id"],
            infer_dims=params["infer_dims"],
        )
        if not ok:
            errors += 1
    return errors


def collect_gie_models() -> list:
    """Collect build params for every GIE_N_CONFIG env var (sorted by N)."""
    models = []
    for key, value in os.environ.items():
        m = re.match(r"^GIE_(\d+)_CONFIG$", key)
        if not m:
            continue
        config_path = value.strip()
        if not os.path.isfile(config_path):
            logger.warning("GIE config not found: %s", config_path)
            continue
        params = parse_gie_config(config_path)
        models.append((int(m.group(1)), config_path, params))
    return sorted(models, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_custom_libs() -> None:
    """Verify the prebuilt nvinfer parser libraries that ship with Osprey are present."""
    lib_dir = os.path.join(os.path.dirname(__file__), "lib")
    required = (
        "nvdsinfer_yolo_det.so",
        "nvdsinfer_yolo_seg.so",
        "nvdsinfer_rtdetr.so",
    )
    for lib in required:
        path = os.path.join(lib_dir, lib)
        if os.path.isfile(path):
            logger.info("Parser library present: %s", path)
        else:
            logger.warning("Parser library missing: %s", path)


def main() -> None:
    check_custom_libs()
    built = errors = skipped = 0

    for idx, config_path, params in collect_gie_models():
        onnx   = params["onnx_file"]
        engine = params["engine_file"]

        if not onnx or not engine:
            logger.warning(
                "GIE_%d (%s): missing onnx-file or model-engine-file — skip",
                idx, config_path,
            )
            skipped += 1
            continue

        if os.path.isfile(engine):
            logger.info("GIE_%d engine exists, skip: %s", idx, engine)
            skipped += 1
            continue

        ok = build_onnx_engine(
            onnx_file=onnx,
            engine_file=engine,
            batch_size=params["batch_size"],
            network_mode=params["network_mode"],
            gpu_id=params["gpu_id"],
            infer_dims=params["infer_dims"],
        )
        if ok:
            built += 1
        else:
            errors += 1

    logger.info(
        "Engine build summary: %d built, %d already existed (skip), %d errors",
        built, skipped, errors,
    )
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
