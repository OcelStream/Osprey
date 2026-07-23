## Osprey Architecture

Osprey (`ospreyai` on PyPI, imported as `osprey`) is a pip-installable Python
library. It runs as **two processes on a single host** that communicate via Unix
domain sockets:

- **Server process** (`osprey-server`, or `osprey.serve()` in-process): hosts the
  DeepStream pipeline and a FastAPI control plane.
- **Client process** (`osprey-client`, or a `DeepStreamClient` subclass): discovers
  Unix sockets created by the server and exposes each stream as RTSP.

There are no containers. Both processes run directly on the host GPU.

### Host requirements

Both processes run on the host GPU, so the host must satisfy the DeepStream 8.0
baseline:

- NVIDIA GPU (Turing or later recommended)
- **NVIDIA driver 570+ with CUDA 13** — the baseline for DeepStream 8.0
- DeepStream 8.0 SDK installed on the host

Install the library and run its one-time host setup:

```bash
pip install ospreyai
sudo osprey-bootstrap
```

`osprey-bootstrap` prepares the host — creating the socket directory
`/run/nvunixfd` used for inter-process communication and wiring up the runtime
prerequisites the pipeline needs.

### Processes and components

- **Server process (`osprey-server`)**
  - Runs the NVIDIA DeepStream pipeline plus a FastAPI application
    (`osprey/server/app.py`).
  - Exposes ports directly on the host:
    - `8000` – FastAPI control plane (REST API).
    - `8554` – RTSP output.
  - Writes per-stream Unix sockets into `/run/nvunixfd` for the client to consume.

- **Client process (`osprey-client`)**
  - Runs `DeepStreamClient` from `osprey/client/base_client.py`, which:
    - Watches `/run/nvunixfd` for Unix socket files created by the server pipeline.
    - Builds GStreamer pipelines dynamically for each detected socket.
    - Serves each stream via RTSP on port `8554`.
  - Applications subclass `DeepStreamClient` to add their own per-frame logic.

Both processes share the host filesystem directory `/run/nvunixfd`, so the sockets
one process creates are directly visible to the other — no network hop or volume
mount involved.

The `osprey` package is laid out as:

- `osprey/server` – FastAPI app, settings/context, API endpoints.
- `osprey/server/deepstream` – the DeepStream pipeline and its GStreamer helpers.
- `osprey/client` – the `DeepStreamClient` base class and IPC/receive plumbing.

### Backend API and DeepStream pipeline

- **FastAPI application**
  - Entry point: `osprey/server/app.py`.
  - Includes router from `osprey/server/api/v1/endpoints.py` under `/api/v1`.
  - Starts the DeepStream pipeline in a background thread inside a FastAPI
    `lifespan` context manager. FastAPI does not accept any requests until the
    pipeline signals readiness via `pipeline._ready` (`threading.Event`). The
    server aborts startup if the pipeline does not reach `PLAYING` state within
    30 seconds. See [fastapi-lifespan-startup.md](../server/fastapi-lifespan-startup.md)
    for full details.

- **API endpoints (`/api/v1`)**
  - `POST /add`
    - Accepts a `StreamRequest` with `uri`, output width/height, and a `stream_id`.
    - Calls `pipeline.add_source(...)` to register a new RTSP source in the DeepStream pipeline.
    - Returns a unique `uuid` and an RTSP URL such as `rtsp://localhost:8554/ds-test{uuid}`.
  - `GET /labels/status`
    - Exposes the current label visibility/status (from `pipeline.get_labels_status()`).
  - `POST /hide_class_name`, `POST /enable_class_name`
    - Toggle visibility of specific detection classes in the pipeline.
  - `GET /streams`
    - Returns a list of active streams.

- **DeepStream pipeline**
  - Implemented under `osprey/server/deepstream` (e.g. `pipeline.py`, `spotmanager.py`, etc.).
  - Uses configuration from `osprey/server/config`:
    - `config_pgie_yolo_detct.txt` – YOLO detection PGIE.
    - `config_pgie_yolo_seg.txt` – YOLO instance segmentation PGIE.
    - `config_tracker_IOU.yml`, `config_tracker_NvSORT.yml`, `config_tracker_NvDCF_perf.yml`, `config_tracker_NvDeepSORT.yml` – tracker algorithm configs.
  - The GStreamer element order is: `nvstreammux → nvinfer (YOLO) → nvtracker → nvstreamdemux`.
    YOLO writes bounding boxes and class labels into `NvDsObjectMeta`. The tracker reads those boxes,
    runs data association across frames, and writes a persistent `object_id` back into each
    `NvDsObjectMeta` before the metadata flows downstream.
  - The active tracker algorithm is selected at startup via the `DS_TRACKER` environment variable
    (`IOU` / `NvSORT` / `NvDCF` / `NvDeepSORT` / `off`). All algorithms use the same shared library
    (`libnvds_nvmultiobjecttracker.so`); the YAML config controls which one activates.
  - NvDeepSORT requires a Re-ID model (`/run/model/resnet50_market1501.etlt`).
    TensorRT compiles an optimized engine on first run and caches it in the same directory.
  - Writes video frames and metadata (including tracking IDs) to Unix domain sockets in
    `/run/nvunixfd` for the client to consume.

### Data and control flow

1. A user or external service calls the FastAPI control plane (`:8000`) to add a new RTSP source via `POST /api/v1/add`.
2. The server registers the source in the DeepStream pipeline and starts processing it.
3. The pipeline pushes each stream through
   `source → nvstreammux → nvinfer → nvtracker → nvstreamdemux → nvunixfdsink`,
   writing buffers and metadata to a dedicated Unix socket `/run/nvunixfd/<id>.sock`.
4. The `DeepStreamClient` (client process) watches `/run/nvunixfd`, discovers new sockets,
   and for each one builds a receive pipeline
   `nvunixfdsrc → OSD → encode → RTSP`.
5. The client exposes each stream as an RTSP endpoint on port `8554`, allowing external
   RTSP viewers to connect at e.g. `rtsp://localhost:8554/ds-test<N>`.

### Configuration and environment

- **Environment variables**
  - `USE_NEW_NVSTREAMMUX` – toggles DeepStream muxer behavior.
  - `DS_TRACKER` – selects the tracking algorithm (`IOU` / `NvSORT` / `NvDCF` / `NvDeepSORT` / `off`). Default: `NvSORT`.
  - `SAVE_FRAMES`, `DRAWING_TYPE`, `SHOW_INFO_TEXT` – used by `DeepStreamClient` to control frame saving and overlay behavior.

- **Shared state on the host**
  - `/run/nvunixfd` – the IPC directory holding one Unix domain socket per active stream. Both the server and client processes read and write here on the host filesystem.
  - Inference/tracker **configs** ship inside the installed package under `osprey/server/config/`. **Models** are user-supplied (Osprey ships parsers, not weights) — place your ONNX/engine wherever you like (e.g. `/run/model/`), including the Re-ID model for NvDeepSORT.

This architecture cleanly separates **pipeline/processing (server)** from **RTSP presentation (client)** while using shared Unix sockets for high-throughput, low-latency inter-process communication.

For a full explanation of how tracking fits into this architecture — concepts, algorithm comparison, implementation details — see [`docs/guides/tracking-implementation.md`](../guides/tracking-implementation.md).
