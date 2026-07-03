# Osprey

Osprey is a dynamic, multi-stream video analytics platform built on NVIDIA
DeepStream 8.0. Add and remove RTSP/file sources at runtime via REST API — each stream gets
GPU-accelerated inference (YOLO-det, YOLO-seg, or RT-DETR), multi-object
tracking, and an independent RTSP output with zero downtime.

```
RTSP cameras / files ──▶ REST API ──▶ DeepStream pipeline ──▶ RTSP output
                            │     (YOLO-det / YOLO-seg / RT-DETR + tracking)
                            │                   │                  │
                         POST /add       Unix domain sockets    VLC / browser
                                          (zero-copy IPC)
```

---

## Architecture

Two Docker containers, one GPU, one shared socket directory:

```
┌────────────────────────────────────────────────────────────┐
│  deepstream container (server)                             │
│                                                            │
│  FastAPI (:7000) ──▶ DynamicRTSPPipeline                         │
│                       nvstreammux → YOLO GIE(s) → nvtracker → demux │
│                                                    │       │
│                                          nvunixfdsink      │
│                                               │            │
└───────────────────────────────────────────────┼────────────┘
                                                │
                                     /run/nvunixfd/*.sock
                                                │
┌───────────────────────────────────────────────┼────────────┐
│  ds_client container (client)                 │            │
│                                          nvunixfdsrc       │
│  DeepStreamClient (base class)                │            │
│    → discovers sockets automatically          │            │
│    → builds receive pipeline per stream       │            │
│    → serves RTSP on :8554 (host :8557)        │            │
└────────────────────────────────────────────────────────────┘
```

| Component | Responsibility |
|-----------|---------------|
| **Server** (`server/`) | REST API, DeepStream inference pipeline, writes frames + metadata to Unix sockets |
| **Client** (`client/`) | Discovers sockets, decodes frames, runs application logic, serves RTSP |
| **Sockets** (`./sockets/`) | Shared volume — zero-copy IPC between containers |

---

## Supported Models

| Model | Task | Config |
|---|---|---|
| YOLO11 / YOLO26 detection | Object detection | `config_pgie_yolo_detct.txt` |
| YOLO11 / YOLO26 segmentation | Instance segmentation | `config_pgie_yolo_seg.txt` |
| RT-DETR-L | Object detection | `config_pgie_rtdetr_l.txt` |

Each task ships with a dedicated `NvDsInferParseCustom*` parser library
(`server/deepstream/app/lib/*.so`) that DeepStream loads at runtime. Each config
expects a TensorRT-ready ONNX whose output layers match that parser. Sample models
are included under `server/deepstream/models/`; see that directory's README for how
to bring your own.

> **Have a `.pt` checkpoint?** Export it in your browser with the hosted
> [Osprey Platform](https://ospreyai.dev/export) — no TensorRT or CUDA toolchain
> needed. It returns a TRT-compatible ONNX with the right output layers for these
> parsers, plus the labels file and a ready-made nvinfer config. You can also
> browse community-exported models on the [Hub](https://ospreyai.dev/hub).

---

## Quick Start

### Prerequisites

- NVIDIA GPU with driver 535+
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Compose v2

### 1. Clone and configure

```bash
git clone <repo-url> && cd osprey
```

Edit `.env` to point at your model config and choose a tracker:

```env
GIE_0_CONFIG=/deepstream_app/deepstream/config/config_pgie_yolo_detct.txt
WIDTH_MODEL=1280
HEIGHT_MODEL=720
DS_TRACKER=NvSORT   # IOU / NvSORT / NvDCF / NvDeepSORT / off
```

### 2. Place your models

Osprey ships sample TensorRT-ready ONNX models in `server/deepstream/models/` so the
pipeline runs end-to-end out of the box. To use your own model, put its TensorRT-ready
ONNX in `server/deepstream/models/` and point `GIE_0_CONFIG` at the matching config
file. See [`server/deepstream/models/README.md`](server/deepstream/models/README.md)
for the per-config filenames and model-license details.

If your weights are still a `.pt` checkpoint, export them at
[ospreyai.dev/export](https://ospreyai.dev/export): it produces `<model>-trt.onnx`,
a `<model>-trt.txt` labels file, and a `<model>-trt-config.txt` nvinfer config —
copy the ONNX to `server/deepstream/models/`, the two `.txt` files to
`server/deepstream/config/`, and point `GIE_0_CONFIG` in `.env` at the config.

| Model type | Config file | Custom parser |
|---|---|---|
| YOLO detection | `config_pgie_yolo_detct.txt` | `nvdsinfer_yolo_det.so` |
| YOLO segmentation | `config_pgie_yolo_seg.txt` | `nvdsinfer_yolo_seg.so` |
| RT-DETR | `config_pgie_rtdetr_l.txt` | `nvdsinfer_rtdetr.so` |

For `NvDeepSORT`, also place the Re-ID model there. The Re-ID model is **not bundled** —
download it from NVIDIA NGC (see
[`docs/guides/tracking-implementation.md`](docs/guides/tracking-implementation.md)):

```
server/deepstream/models/
├── your-model.onnx                     ← TensorRT-ready ONNX
└── resnet50_market1501.etlt            ← Re-ID model (NvDeepSORT only, download from NGC)
```

### 3. Build and run

```bash
docker compose up --build
```

On **first run** the container builds all missing TensorRT engines before
starting the API. This can take **5–20 minutes** depending on model size and
GPU. Watch the logs:

```
[entrypoint] Building TensorRT engines...
Building ONNX engine: .../yolo11l.pt.onnx → ...engine
[entrypoint] Starting application...
```

On subsequent runs all engines already exist and the container starts in
seconds. See [`docs/guides/engine-builder.md`](docs/guides/engine-builder.md)
for details.

Wait for the server log `Pipeline is PLAYING` before adding streams.

### 3. Add a stream

```bash
curl -X POST http://localhost:7000/api/v1/add \
  -H "Content-Type: application/json" \
  -d '{
    "uri": "rtsp://your-camera/stream",
    "rtsp_output_width": 640,
    "rtsp_output_height": 640,
    "stream_id": "camera-1"
  }'
```

### 4. Watch the output

Open the RTSP stream in VLC or any RTSP player:

```
rtsp://localhost:8557/ds-testcamera-1
```

---

## REST API

All endpoints are under `http://localhost:7000/api/v1` (host port `7000` maps to
the container's `8000`).

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/add` | `{"uri", "rtsp_output_width", "rtsp_output_height", "stream_id"}` | Add a video source |
| `DELETE` | `/remove/{stream_id}` | — | Remove a video source |
| `GET` | `/streams` | — | List active streams |
| `GET` | `/health/ready` | — | Readiness probe — 200 when pipeline is PLAYING |

### Add stream — example response

```json
{
  "message": "Stream added",
  "uuid": "camera-1",
  "rtsp": "rtsp://localhost:8554/ds-testcamera-1"
}
```

---

## Project Structure

```
.
├── docker-compose.yml          # Two-service stack (server + client)
├── Dockerfile                  # Server image (DeepStream 8.0 + FastAPI)
├── .env                        # GIE configs, model resolution, env vars
├── sockets/                    # Shared IPC directory (auto-created)
│
├── server/
│   ├── backend/app/
│   │   ├── app.py              # FastAPI entry point
│   │   ├── models.py           # Pydantic request models
│   │   ├── core/context.py     # Pipeline singleton
│   │   └── api/v1/endpoints.py # REST endpoints
│   │
│   └── deepstream/
│       ├── app/
│       │   ├── deepstream.py           # DynamicRTSPPipeline (core engine)
│       │   ├── build_engines.py        # TensorRT engine pre-builder
│       │   ├── spotmanager.py          # Stream slot allocation
│       │   ├── source_bin_factory.py   # GStreamer source bin creation
│       │   ├── utils.py
│       │   └── lib/
│       │       ├── nvdsinfer_yolo_det.so        # Custom parser — YOLO detection (binary)
│       │       ├── nvdsinfer_yolo_seg.so        # Custom parser — YOLO segmentation (binary)
│       │       ├── nvdsinfer_rtdetr.so          # Custom parser — RT-DETR (binary)
│       │       └── serialize_meta.so            # Metadata serializer — IPC (binary)
│       ├── config/
│       │   ├── config_pgie_yolo_detct.txt       # YOLO detection PGIE config
│       │   ├── config_pgie_yolo_seg.txt         # YOLO segmentation PGIE config
│       │   ├── config_pgie_rtdetr_l.txt         # RT-DETR PGIE config
│       │   ├── labels_det.txt
│       │   ├── yolo11l-seg_labels.txt
│       │   ├── config_tracker_IOU.yml           # DS_TRACKER=IOU
│       │   ├── config_tracker_NvSORT.yml        # DS_TRACKER=NvSORT
│       │   ├── config_tracker_NvDCF_perf.yml    # DS_TRACKER=NvDCF
│       │   └── config_tracker_NvDeepSORT.yml    # DS_TRACKER=NvDeepSORT
│       └── models/                 # Sample ONNX models (+ your own)
│               └── README.md                       # Expected filenames + licenses
│
├── client/
│   ├── base_client.py              # DeepStreamClient base class
│   └── lib/
│       └── deserialize_meta.so     # Metadata deserializer — IPC (binary)
│
└── docs/
    ├── concepts/
    │   ├── overview.md                 # What this is and why it exists
    │   ├── deepstream-pipeline.md      # GStreamer/DeepStream concepts
    │   ├── tensorrt-engines.md         # ONNX → TensorRT engine conversion
    │   ├── two-container-model.md      # Why two containers, the boundary
    │   ├── stream-lifecycle.md         # Add/remove state machine
    │   └── ipc-unix-sockets.md         # Zero-copy GPU buffer IPC
    ├── architecture/
    │   └── arch.md                     # System architecture diagram
    ├── guides/
    │   ├── building-apps.md            # How to build apps on the client
    │   ├── engine-builder.md           # TensorRT engine pre-builder
    │   ├── tracking-implementation.md  # Multi-object tracking, A to Z
    │   ├── tracker-implementation.md   # Gst-nvtracker integration (concise)
    │   ├── metadata-guide.md           # DeepStream metadata, complete guide
    │   └── metadata-structs-visual.md  # Metadata struct reference
    └── server/
        ├── fastapi-lifespan-startup.md # FastAPI lifespan + readiness probe
        ├── pydantic-settings-config.md # PipelineSettings config
        └── element-factory.md          # DeepStreamElementFactory
```

---

## Building Applications on the Client

The client is designed as a base class. Subclass `DeepStreamClient` to build
custom video analytics applications without touching GStreamer or pyds.

Each detected object arrives as an `ObjectData` with a persistent tracking ID,
class label, and confidence already populated from the pipeline:

```python
@dataclass
class ObjectData:
    object_id: int    # persistent ID from nvtracker (0 = no tracker)
    label: str        # class name from nvinfer (e.g. "car")
    confidence: float
    class_id: int
    left: int
    top: int
    width: int
    height: int
```

The base class draws `ID:42 car 0.91` above each box by default (class-colored).
Override `_draw_object()` to change the rendering.

```python
from base_client import DeepStreamClient, FrameData

class VehicleCounter(DeepStreamClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counts = {}

    def _process_frame(self, frame_data: FrameData):
        vehicles = [o for o in frame_data.objects if o.label in ("car", "truck")]
        self.counts[frame_data.source_id] = len(vehicles)

        for v in vehicles:
            self._draw_object(frame_data.surface, v)   # draws box + tracking label

if __name__ == "__main__":
    VehicleCounter().start()
```

Override hooks for custom behavior:

| Hook | Receives | Purpose |
|------|----------|---------|
| `_process_frame(frame_data)` | `FrameData` (surface + objects) | Per-frame application logic |
| `_draw_object(surface, obj)` | numpy array, `ObjectData` | Custom bounding box rendering |
| `_draw_info_text(surface, frame_data)` | numpy array, `FrameData` | Custom HUD overlay |
| `_on_stream_added(record)` | `StreamRecord` | React to new camera |
| `_on_stream_removed(record)` | `StreamRecord` | React to camera disconnect |

See [`docs/guides/building-apps.md`](docs/guides/building-apps.md) for the full guide with
a complete parking monitor example.

---

## Configuration

### Environment variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GIE_0_CONFIG` | — | Path to primary inference config (YOLO detection) |
| `GIE_1_CONFIG` | — | Path to secondary inference config (segmentation, optional) |
| `WIDTH_MODEL` | `640` | Model input width |
| `HEIGHT_MODEL` | `640` | Model input height |
| `DS_BATCHED_PUSH_TIMEOUT` | `66666` | Streammux batch timeout (microseconds) |
| `USE_NEW_NVSTREAMMUX` | `yes` | Use new nvstreammux plugin |
| `DS_TRACKER` | `NvSORT` | Tracker algorithm — `IOU` / `NvSORT` / `NvDCF` / `NvDeepSORT` / `off` |
| `SAVE_FRAMES` | `0` | Save frames to disk (client) |
| `DRAWING_TYPE` | `native` | Drawing mode (client) |
| `SHOW_INFO_TEXT` | `1` | Show HUD overlay (client) |

### Ports

| Host Port | Container Port | Service | Protocol |
|-----------|---------------|---------|----------|
| `7000` | `8000` | Server — FastAPI | HTTP |
| `8554` | `8554` | Server — RTSP | RTSP |
| `8557` | `8554` | Client — RTSP | RTSP |

---

## Data Flow

```
1. POST /api/v1/add { uri: "rtsp://camera/stream", stream_id: "cam-1" }
                │
                ▼
2. FastAPI → DynamicRTSPPipeline.add_source()
                │
                ▼
3. GStreamer pipeline:
   nvurisrcbin → nvvideoconvert → nvstreammux → nvinfer (YOLO) → nvtracker
                                                                      │
                                                               nvstreamdemux
                                                                      │
                                                               nvunixfdsink
                                                    │
                                        /run/nvunixfd/cam-1.sock
                                                    │
4. DeepStreamClient discovers socket              ◄─┘
                │
                ▼
5. Receive pipeline:
   nvunixfdsrc → nvstreammux → nvvideoconvert → nvdsosd → h264enc → RTSP
                                      │
                                  _osd_probe()
                                      │
                               _process_frame(FrameData)
                                      │
                              Your application logic
```

---

## Documentation

### Concepts — start here if you are new

| Document | Description |
|----------|-------------|
| [`docs/concepts/overview.md`](docs/concepts/overview.md) | What this system is, the problem it solves, the core philosophy |
| [`docs/concepts/deepstream-pipeline.md`](docs/concepts/deepstream-pipeline.md) | GStreamer elements, batch inference, NVMM memory model |
| [`docs/concepts/tensorrt-engines.md`](docs/concepts/tensorrt-engines.md) | What ONNX and TensorRT engines are, how ONNX→engine conversion works, what `build_engines.py` does step by step |
| [`docs/concepts/two-container-model.md`](docs/concepts/two-container-model.md) | Why two containers, what each owns, the dependency direction |
| [`docs/concepts/stream-lifecycle.md`](docs/concepts/stream-lifecycle.md) | Add/remove state machine, lock discipline, spot reuse |
| [`docs/concepts/ipc-unix-sockets.md`](docs/concepts/ipc-unix-sockets.md) | Zero-copy GPU buffer fd passing, metadata serialization |

### Architecture and guides

| Document | Description |
|----------|-------------|
| [`docs/architecture/arch.md`](docs/architecture/arch.md) | System architecture — containers, ports, data flow |
| [`docs/guides/engine-builder.md`](docs/guides/engine-builder.md) | TensorRT engine pre-builder — how it works, what it builds, how to force a rebuild |
| [`docs/guides/building-apps.md`](docs/guides/building-apps.md) | How to build applications on the client base class |
| [`docs/guides/tracking-implementation.md`](docs/guides/tracking-implementation.md) | Multi-object tracking — concepts, 4 algorithms, full implementation A to Z |
| [`docs/guides/tracker-implementation.md`](docs/guides/tracker-implementation.md) | Gst-nvtracker integration — a concise walkthrough |
| [`docs/guides/metadata-guide.md`](docs/guides/metadata-guide.md) | DeepStream metadata model — the complete guide |
| [`docs/guides/metadata-structs-visual.md`](docs/guides/metadata-structs-visual.md) | Visual reference for the metadata structs |

### Hosted platform (ospreyai.dev)

| Resource | Description |
|----------|-------------|
| [ospreyai.dev/export](https://ospreyai.dev/export) | Browser-based `.pt` → TRT-compatible ONNX exporter — returns ONNX + labels + ready-made nvinfer config |
| [ospreyai.dev/hub](https://ospreyai.dev/hub) | Public gallery of community-exported models, ready to drop into this pipeline |
| [ospreyai.dev/docs](https://ospreyai.dev/docs) | Hosted docs — quickstart, export guide, REST API and settings reference |

### Server implementation

| Document | Description |
|----------|-------------|
| [`docs/server/fastapi-lifespan-startup.md`](docs/server/fastapi-lifespan-startup.md) | FastAPI lifespan startup, readiness probe, Docker healthcheck |
| [`docs/server/pydantic-settings-config.md`](docs/server/pydantic-settings-config.md) | `PipelineSettings` — typed config with `pydantic-settings` |
| [`docs/server/element-factory.md`](docs/server/element-factory.md) | `DeepStreamElementFactory` — centralised element creation |

---

## Requirements

- NVIDIA GPU (Turing or later recommended)
- NVIDIA Driver 535+
- Docker 24+ with NVIDIA Container Toolkit
- Docker Compose v2
- DeepStream 8.0 base image (`ilkaybrahim/deepstream_app:8.0`)
