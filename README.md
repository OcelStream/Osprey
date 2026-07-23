# Osprey

Dynamic, multi-stream video-analytics library built on **NVIDIA DeepStream 8.0**
— installable with `pip`, run **bare-metal** on the host.

```python
import osprey.server as osprey

# Configure the model + tracker, then start the server (its own process).
osprey.configure(gie_config="/models/gie.txt", tracker="NvSORT")
osprey.serve()                                   # returns once the server is healthy
osprey.add_stream("rtsp://camera/stream", stream_id="cam1")

# Write your analytics with the client — no GStreamer/pyds knowledge needed.
from osprey.client import DeepStreamClient, FrameData

class VehicleCounter(DeepStreamClient):
    def _process_frame(self, frame: FrameData):
        for obj in frame.objects:
            self._draw_object(frame.surface, obj)  # box + tracking label

VehicleCounter().start()                          # serves RTSP for every discovered stream
```

One file: `configure` → `serve` runs the DeepStream inference pipeline + REST
control plane in a **separate process**, while your `DeepStreamClient` runs the
analytics and serves annotated RTSP. Prefer the CLI (`osprey-server` /
`osprey-client`) for a production two-service deployment — see below.

---

## What Osprey installs for you

Osprey is a Python layer on top of the DeepStream SDK. The GStreamer plugins
(`nvstreammux`, `nvinfer`, `nvtracker`, `nvunixfdsink/src`, `nvdsosd`, …) and the
`pyds` bindings are **not** on PyPI. Rather than make you set that up by hand,
Osprey ships an **end-to-end bootstrap** that installs the full DeepStream 8.0
stack bare-metal — so a fresh Ubuntu 24.04 box becomes a working Osprey host.

Target platform (DeepStream 8.0, x86/dGPU):

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 |
| NVIDIA driver | R570.133.20 |
| CUDA | 12.8 |
| TensorRT | 10.9.0.34 |
| cuDNN | 9.7.1 |
| GStreamer | 1.24.2 |

---

## Quick start (end-to-end)

```bash
# 1. Install the Python package (pure-Python + precompiled parser .so)
pip install ospreyai

# 2. Bootstrap the DeepStream stack bare-metal (needs root; ~one-time)
sudo osprey-bootstrap

# 3a. Run as two services (production)
osprey-server     # FastAPI control plane on :8000  (POST /api/v1/add …)
osprey-client     # discovers sockets, serves RTSP per stream

# 3b. …or a single Python file (see the example at the top)
python3 my_app.py   # configure() → serve() → your DeepStreamClient
```

Configuration is either **programmatic** (`osprey.configure(...)`, shown above)
or via **environment variables** (`GIE_0_CONFIG`, `DS_TRACKER`, `DS_MODEL_WIDTH`
…) for the `osprey-server` CLI. See [`examples/`](examples/) for a runnable
single-file app.

`osprey-bootstrap` runs five stages (each also runnable on its own; see below).
If the NVIDIA driver is (re)installed in stage 10, **reboot** before running the
pipeline.

### Bootstrap stages

| Stage | Does | Mirrors |
|---|---|---|
| `00_system_deps` | apt build toolchain + GStreamer runtime | image apt layer |
| `10_cuda_trt_cudnn` | driver R570 + CUDA 12.8 + cuDNN 9.7 + TensorRT 10.9 | base image |
| `20_deepstream_sdk` | DeepStream 8.0 SDK `.deb` from NGC → GStreamer plugins | base image |
| `30_pyds` | build + install `pyds` for your Python | image bindings build |
| `40_native_libs` | verify shipped parsers/serializer; TRT-plugin patch is **manual** | image compile step |

Useful toggles (read by the scripts, pass straight through):

```bash
sudo OSPREY_ASSUME_CUDA=1 osprey-bootstrap    # already have driver/CUDA/TRT/cuDNN
sudo OSPREY_ONLY=30 osprey-bootstrap          # run a single stage (e.g. just pyds)
sudo OSPREY_DS_VERSION=8.0 osprey-bootstrap   # override DeepStream version
```

> **Security note.** Osprey does **not** auto-download or overwrite your system
> TensorRT library. Some ONNX models embed end-to-end NMS (`EfficientNMS_TRT`)
> and need a patched `libnvinfer_plugin`; because that means replacing a system
> library with an external binary, Osprey leaves it as a deliberate manual step
> (see the reference `patch_libnvinfer.sh`). Standard YOLO/RT-DETR detection and
> segmentation need no patch.

---

## Python environment — `gi` + `pyds` must share the interpreter

Osprey needs three pieces **in the same Python interpreter**:

| Piece | Comes from | Lives in |
|---|---|---|
| `ospreyai` | `pip` | wherever you `pip install` |
| `gi` (PyGObject) | apt (`python3-gi`) | **system** Python |
| `pyds` | built by `osprey-bootstrap` | the Python active during bootstrap |

Because `gi` and `pyds` are **not** PyPI packages and live in system Python, a
**plain virtualenv can't see them** — that's the usual cause of
`ModuleNotFoundError: No module named 'gi'` (or `'pyds'`). Use one of:

```bash
# A) No venv (simplest) — install next to system gi/pyds
pip install --break-system-packages ospreyai

# B) A venv that can see system packages (any path)
python3 -m venv --system-site-packages ~/osprey-venv
source ~/osprey-venv/bin/activate
pip install ospreyai
```

A **plain** `python3 -m venv` (without `--system-site-packages`) will **not**
work — it hides system `gi`/`pyds`. The venv's Python must also be the same
minor version as system Python (3.12 on Ubuntu 24.04).

> **Running the bootstrap from a venv?** `sudo osprey-bootstrap` fails with
> `command not found` — `sudo` resets `PATH` and drops your venv. Run it by its
> full path instead:
> ```bash
> sudo $(command -v osprey-bootstrap)
> ```
> (`pyds` then builds into **system** Python, which a `--system-site-packages`
> venv sees.)

Verify any interpreter with:

```bash
osprey-doctor      # checks gi + pyds + osprey + plugins, prints the fix if not
```

---

## Already have DeepStream 8.0?

Skip the bootstrap and just use the library:

```bash
pip install ospreyai
python3 -c "from osprey.client import DeepStreamClient; print('ok')"
```

The bundled native libraries (`osprey/**/lib/*.so`) are compiled for
DeepStream 8.0 / CUDA 12.8 and match the platform table above.

---

## Supported Models

| Model | Task | Config |
|---|---|---|
| YOLO11 / YOLO26 detection | Object detection | `config_pgie_yolo_detct.txt` |
| YOLO11 / YOLO26 segmentation | Instance segmentation | `config_pgie_yolo_seg.txt` |
| RT-DETR-L | Object detection | `config_pgie_rtdetr_l.txt` |

Each task uses a dedicated `NvDsInferParseCustom*` parser that **ships with the
package** (`osprey/server/deepstream/lib/*.so`), loaded by DeepStream at runtime.
Osprey ships the parsers, **not** the weights — supply your own TensorRT-ready
ONNX whose output layers match the parser. See
[`examples/gie.txt`](examples/gie.txt) and
[`examples/make_gie_config.py`](examples/make_gie_config.py) for wiring a model
to a parser.

> **Have a `.pt` checkpoint?** Export it in your browser with the hosted
> [Osprey Platform](https://ospreyai.dev/export) — no TensorRT or CUDA toolchain
> needed. It returns a TRT-compatible ONNX with the right output layers for these
> parsers, plus the labels file and a ready-made nvinfer config. Browse
> community-exported models on the [Hub](https://ospreyai.dev/hub).

---

## CLIs

| Command | Purpose |
|---|---|
| `osprey-bootstrap` | Bare-metal end-to-end DeepStream install (root) |
| `osprey-doctor` | Check `gi` + `pyds` + `osprey` + plugins in the current interpreter |
| `osprey-server` | FastAPI control plane — add/remove streams at runtime |
| `osprey-client` | Discover sockets, run app logic, serve RTSP |
| `osprey-build-engines` | Pre-build TensorRT engines from `GIE_N_CONFIG` |

---

## Documentation

### Concepts — start here if you are new

| Document | Description |
|----------|-------------|
| [`docs/concepts/overview.md`](docs/concepts/overview.md) | What Osprey is, the problem it solves, the core philosophy |
| [`docs/concepts/deepstream-pipeline.md`](docs/concepts/deepstream-pipeline.md) | GStreamer elements, batch inference, NVMM memory model |
| [`docs/concepts/tensorrt-engines.md`](docs/concepts/tensorrt-engines.md) | ONNX and TensorRT engines, how ONNX→engine conversion works |
| [`docs/concepts/two-process-model.md`](docs/concepts/two-process-model.md) | The server/client process split and the dependency direction |
| [`docs/concepts/stream-lifecycle.md`](docs/concepts/stream-lifecycle.md) | Add/remove state machine, lock discipline, spot reuse |
| [`docs/concepts/ipc-unix-sockets.md`](docs/concepts/ipc-unix-sockets.md) | Zero-copy GPU buffer fd passing, metadata serialization |

### Architecture and guides

| Document | Description |
|----------|-------------|
| [`docs/architecture/arch.md`](docs/architecture/arch.md) | System architecture — processes, ports, data flow |
| [`docs/guides/engine-builder.md`](docs/guides/engine-builder.md) | TensorRT engine pre-builder — how it works, forcing a rebuild |
| [`docs/guides/building-apps.md`](docs/guides/building-apps.md) | Building applications on the `DeepStreamClient` base class |
| [`docs/guides/tracking-implementation.md`](docs/guides/tracking-implementation.md) | Multi-object tracking — concepts, 4 algorithms, full implementation |
| [`docs/guides/tracker-implementation.md`](docs/guides/tracker-implementation.md) | Gst-nvtracker integration — a concise walkthrough |
| [`docs/guides/metadata-guide.md`](docs/guides/metadata-guide.md) | DeepStream metadata model — the complete guide |
| [`docs/guides/metadata-structs-visual.md`](docs/guides/metadata-structs-visual.md) | Visual reference for the metadata structs |

### Server implementation

| Document | Description |
|----------|-------------|
| [`docs/server/fastapi-lifespan-startup.md`](docs/server/fastapi-lifespan-startup.md) | FastAPI lifespan startup + readiness probe |
| [`docs/server/pydantic-settings-config.md`](docs/server/pydantic-settings-config.md) | `PipelineSettings` — typed config with `pydantic-settings` |
| [`docs/server/element-factory.md`](docs/server/element-factory.md) | `DeepStreamElementFactory` — centralised element creation |

### Hosted platform (ospreyai.dev)

| Resource | Description |
|----------|-------------|
| [ospreyai.dev/export](https://ospreyai.dev/export) | Browser-based `.pt` → TRT-compatible ONNX exporter |
| [ospreyai.dev/hub](https://ospreyai.dev/hub) | Public gallery of community-exported models |
| [ospreyai.dev/docs](https://ospreyai.dev/docs) | Hosted docs — quickstart, export guide, REST/settings reference |
