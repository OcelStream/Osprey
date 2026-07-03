# TensorRT Engine Builder

`server/deepstream/app/build_engines.py` runs automatically at container
startup (via `entrypoint.sh`) before the DeepStream pipeline or the FastAPI
app start. It checks every configured model and builds any missing TensorRT
engines so the pipeline never encounters a cold-build delay at runtime.

---

## How it works

```
docker compose up
      │
      ▼
entrypoint.sh
      │
      ├─ python3 build_engines.py
      │         │
      │         ├─ read GIE_0_CONFIG, GIE_1_CONFIG, … from env
      │         │         │
      │         │         ├─ engine exists? → skip
      │         │         └─ engine missing? → trtexec → .engine
      │         │
      │         └─ exit 0 (all ok) / exit 1 (build failed)
      │
      └─ exec uvicorn ...   ← only reaches here if build_engines exits 0
```

The healthcheck `start_period` is set to 20 minutes so Docker waits patiently
while engines are building before declaring the container unhealthy.

---

## What gets built

### ONNX → TensorRT engine (via `trtexec`)

Every `GIE_N_CONFIG` env var points to an nvinfer config file. The builder
reads three fields from each:

| Config key | Used for |
|------------|----------|
| `onnx-file` | source model |
| `model-engine-file` | output engine path |
| `batch-size` | opt/max shape profile |
| `network-mode` | precision (`0`=FP32, `1`=INT8, `2`=FP16) |
| `gpu-id` | target GPU |
| `infer-dims` | spatial dims if set (e.g. `3;640;640`) |

The engine is built with a dynamic-batch profile:

```
min  = 1  × spatial_dims
opt  = batch_size × spatial_dims
max  = batch_size × spatial_dims
```

The input tensor name is read from the ONNX graph automatically
(`onnxruntime`), falling back to `images` if not available.

### Re-ID model (NvDeepSORT)

The Re-ID model (`resnet50_market1501.etlt`) is an encrypted TAO model.
Converting it requires the `tao-converter` tool which is not included in the
base image. DeepStream's `libnvds_nvmultiobjecttracker.so` handles this
conversion internally on first pipeline run and caches the result as
`resnet50_market1501.etlt_b100_gpu0_fp16.engine`.

`build_engines.py` intentionally does **not** try to build this engine.
DeepStream owns it.

---

## First run vs subsequent runs

| Situation | Behaviour |
|-----------|-----------|
| Engine file exists | Skipped immediately — no GPU work |
| ONNX exists, engine missing | `trtexec` builds the engine (5–20 min) |
| ONNX missing | Error — container exits 1, check your model path |
| Re-ID engine missing (NvDeepSORT) | DeepStream builds it on first pipeline run |

---

## Force a rebuild

Delete the engine file and restart:

```bash
rm server/deepstream/models/yolo11l_bbox_v8-trt.onnx.engine
docker compose restart deepstream
```

The builder detects the missing file and rebuilds from the ONNX.

---

## Add a new model

1. Add the ONNX to `server/deepstream/models/`
2. Create (or copy) an nvinfer config in `server/deepstream/config/`
3. Set the new env var in `.env`:

```env
GIE_1_CONFIG=/deepstream_app/deepstream/config/config_pgie_yolo_seg.txt
```

The builder picks up every `GIE_N_CONFIG` variable automatically — no code
changes needed.

---

## Run the builder manually

To build engines without starting the full stack:

```bash
docker compose run --rm deepstream python3 deepstream/app/build_engines.py
```

---

## Logs

During a build you will see:

```
[entrypoint] Building TensorRT engines...
2026-04-17 10:31:20 INFO: GIE_0 engine exists, skip: .../model.engine
2026-04-17 10:31:20 INFO: Building ONNX engine: .../model.onnx → .../model.engine
2026-04-17 10:31:20 INFO:   cmd: trtexec --onnx=... --saveEngine=... ...
...
2026-04-17 10:45:03 INFO: Engine ready: .../model.engine
2026-04-17 10:45:03 INFO: Engine build summary: 1 built, 0 already existed (skip), 0 errors
[entrypoint] Starting application...
```

If a build fails the container exits immediately with code 1 so the problem
is visible before the pipeline starts.
