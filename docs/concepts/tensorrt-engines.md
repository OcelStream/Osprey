# TensorRT Engines and ONNX — Deep Explanation

## What is ONNX?

**ONNX** (Open Neural Network Exchange) is an open format for representing
machine learning models. Think of it as a universal language that any training
framework can write and any inference runtime can read.

```
PyTorch  ──┐
TensorFlow ─┼──► ONNX (.onnx file) ──► TensorRT / ONNX Runtime / OpenVINO / ...
JAX      ──┘
```

An ONNX file contains:
- The **graph** — every layer (Conv, BN, ReLU, …) as nodes with edges
- The **weights** — the learned parameters, frozen at export time
- The **input/output shapes** — tensor names and dimensions

ONNX is **framework-independent and hardware-independent**. You can export
once from PyTorch and run anywhere. It is the standard hand-off format between
training and deployment.

### What an ONNX file actually stores

```
graph:
  input: images [batch, 3, 640, 640]   ← tensor name + shape
  nodes:
    Conv(images, weight_0, bias_0)  → x1
    BatchNorm(x1, …)               → x2
    SiLU(x2)                       → x3
    …hundreds more…
    Sigmoid(x_n)                   → output0
  output: output0 [batch, 84, 8400]
  initializers:
    weight_0: float32[64,3,3,3]    ← actual numbers
    bias_0:   float32[64]
    …
```

Every weight is embedded in the file. A YOLO11-L ONNX is ~100 MB because it
is carrying all those numbers.

Osprey needs an ONNX whose output layers match its custom parsers (in-graph
NMS via TRT plugins). The hosted exporter at
[ospreyai.dev/export](https://ospreyai.dev/export) produces exactly this format
from a plain `.pt` checkpoint — no local TensorRT install needed.

---

## What is a TensorRT Engine?

**TensorRT** is NVIDIA's inference optimizer and runtime. It takes a model
description (ONNX or its own API) and produces a **serialized engine** — a
binary blob that is:

1. **Compiled** for the exact GPU model and CUDA version on the machine
2. **Optimized** — layers fused, memory layout changed, unnecessary ops removed
3. **Precision-aware** — weights quantized to FP16 or INT8 if requested

The `.engine` file is the result of that compilation. It is opaque binary data
that only TensorRT can deserialize and run.

### Why not just run the ONNX directly?

You could use ONNX Runtime with the TensorRT execution provider and it would
internally do the same thing, but:

- The compilation happens **every time** the process starts (minutes of work)
- The result is not saved between runs
- DeepStream's `nvinfer` plugin requires a TensorRT engine natively

Pre-building the engine and saving it means:
- First run: slow (compile once)
- Every run after: fast (load pre-built binary in seconds)

---

## The ONNX → TensorRT Conversion Pipeline

```
ONNX file
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  TensorRT Builder                                   │
│                                                     │
│  1. Parse ONNX graph                                │
│     └─ Validate ops, resolve shapes                 │
│                                                     │
│  2. Apply optimizations                             │
│     ├─ Layer fusion (Conv+BN+ReLU → single kernel)  │
│     ├─ Kernel auto-tuning (picks fastest impl)      │
│     ├─ Memory layout reordering (NCHW ↔ NHWC)      │
│     └─ Dead code elimination                        │
│                                                     │
│  3. Precision conversion                            │
│     ├─ FP32 → as-is                                 │
│     ├─ FP16 → halve every weight, ~2× faster        │
│     └─ INT8 → quantize weights, needs calib data    │
│                                                     │
│  4. Build optimization profiles                     │
│     └─ min/opt/max batch shapes baked in            │
│                                                     │
│  5. Serialize → .engine file                        │
└─────────────────────────────────────────────────────┘
    │
    ▼
.engine file  (GPU-specific binary)
```

### Step 2 in detail — what "layer fusion" actually means

A typical YOLO bottleneck in ONNX has 3 separate nodes:

```
Conv → BatchNorm → SiLU
```

Each is a separate GPU kernel launch (overhead) and intermediate result
written to memory (bandwidth). TensorRT fuses them into a single kernel:

```
ConvBNSiLU  ← one kernel, no intermediate writes
```

On a YOLO11-L model this alone gives ~1.4× speedup before any precision change.

### Step 3 in detail — FP16

FP16 (half precision) stores each weight in 16 bits instead of 32.

- Model file: ~2× smaller (100 MB ONNX → ~50 MB engine)
- Arithmetic: modern NVIDIA GPUs (Turing+) have dedicated FP16 tensor cores
  that run **~2× faster** than FP32
- Accuracy: almost identical for inference. YOLO detection mAP drops < 0.1%
  in practice

The tradeoff: FP16 can overflow for very large activations, but YOLO models
are designed to be numerically stable at FP16.

### Step 4 in detail — optimization profiles

ONNX models can have **dynamic batch dimensions** (the batch axis is `?`):

```
input: images [?, 3, 640, 640]
```

TensorRT needs to know the range of batch sizes to optimize for. You provide
three points:

```
--minShapes=images:1x3x640x640    ← smallest batch you will ever run
--optShapes=images:10x3x640x640   ← the batch size to optimize hardest for
--maxShapes=images:10x3x640x640   ← largest batch you will ever run
```

TensorRT pre-generates GPU kernels for these sizes. At runtime the engine
accepts any batch between min and max, running the opt-tuned kernels for the
optimal size.

---

## What `build_engines.py` Does — Step by Step

```python
# 1. Read every GIE_N_CONFIG env var
GIE_0_CONFIG=/deepstream_app/deepstream/config/config_pgie_yolo_detct.txt
```

```python
# 2. Parse the nvinfer config to find:
onnx_file   = /deepstream_app/deepstream/models/yolo11l_bbox_v8-trt.onnx
engine_file = /deepstream_app/deepstream/models/yolo11l_bbox_v8-trt.onnx.engine
batch_size  = 10
network_mode = 0   # FP32
gpu_id       = 0
infer_dims   =     # empty → defaults to 3x640x640
```

```python
# 3. Check if the engine already exists
os.path.isfile(engine_file)  → True → skip
                             → False → build
```

```python
# 4. Detect the ONNX input tensor name
# uses onnxruntime to load the graph and read the first input name
# typical YOLO export: "images"
input_name = "images"
```

```python
# 5. Call trtexec
trtexec \
  --onnx=/deepstream_app/deepstream/models/yolo11l_bbox_v8-trt.onnx \
  --saveEngine=/deepstream_app/deepstream/models/yolo11l_bbox_v8-trt.onnx.engine \
  --device=0 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:10x3x640x640 \
  --maxShapes=images:10x3x640x640
  # no precision flag = FP32 (network-mode=0)
```

```
# 6. trtexec output (trimmed):
[I] Starting engine build...
[I] [TRT] Detected 1 input and 1 output.
[I] [TRT] Fusing conv_0 + bn_0 + relu_0 → ConvBNReLU
... (hundreds of fusion messages)
[I] [TRT] Total Activation Memory: 412.3 MB
[I] [TRT] Engine built in 487.2 seconds
[I] Saving engine to file...
```

```python
# 7. Engine is on disk, DeepStream loads it directly from now on
nvinfer:
  config-file-path = config_pgie_yolo_detct.txt
  # sees model-engine-file exists → deserializes in < 2 seconds
  # never calls engine-create-func-name again
```

---

## Engine File Naming Convention

DeepStream's `nvinfer` generates engine filenames automatically when
`model-engine-file` is **not** set:

```
<onnx_filename>_b<batch>_gpu<id>_<precision>.engine

yolo11l_bbox_v8-trt.onnx_b10_gpu0_fp32.engine
```

When `model-engine-file` **is** set (as in this project), DeepStream uses that
exact path instead. The builder writes to the same path so nvinfer loads it
without rebuilding.

---

## Why the Engine is GPU-Specific

TensorRT queries the GPU at build time:

- Compute capability (SM version)
- Available tensor cores
- Memory bandwidth
- Supported kernel variants

It picks the fastest kernel implementation for **that specific chip**. An
engine built on an RTX 3090 will not load on an RTX 4090 (different SM
version). DeepStream detects this and rebuilds automatically if needed, but
pre-building avoids that delay at startup.

---

## Re-ID Model (.etlt) — Why DeepStream Handles It

The NvDeepSORT Re-ID model ships as a `.etlt` file — an **encrypted TAO
model**. The encryption key (`nvidia_tao`) is embedded in
`libnvds_nvmultiobjecttracker.so`. No public tool can decrypt and convert it
without that library.

When DeepStream starts and the tracker initializes:

```
libnvds_nvmultiobjecttracker.so
    │
    ├─ decrypt resnet50_market1501.etlt  (key: "nvidia_tao")
    ├─ parse the decrypted network
    ├─ run TensorRT builder (FP16, batch=100, input=3×256×128)
    └─ save resnet50_market1501.etlt_b100_gpu0_fp16.engine
```

This happens once. On all subsequent starts DeepStream loads the cached
`.engine` directly, same as the ONNX models.

---

## Summary

| Format | Who writes it | Who reads it | Portable? | Build cost |
|--------|--------------|--------------|-----------|------------|
| `.onnx` | PyTorch / training framework | TensorRT, ONNX Runtime, … | Yes — any hardware | None (it's just a description) |
| `.engine` | TensorRT builder (`trtexec`) | TensorRT runtime only | No — GPU-specific | 5–20 min, paid once |
| `.etlt` | NVIDIA TAO toolkit | DeepStream tracker only | No | Paid on first tracker init |
