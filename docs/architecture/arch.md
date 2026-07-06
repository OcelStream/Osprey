## Osprey Architecture

This project is split into two main components that run in separate Docker services and communicate via Unix domain sockets:

- **Server (`deepstream` service)**: Hosts the DeepStream pipeline and the FastAPI backend.
- **Client (`ds_client` service)**: Runs a GStreamer-based DeepStream client that discovers Unix sockets and exposes RTSP streams.

### Host requirements

Both services run on the host GPU through the NVIDIA Container Toolkit, so the host must satisfy the DeepStream 8.0 baseline:

- NVIDIA GPU (Turing or later recommended)
- **NVIDIA driver 570+ with CUDA 13** — the baseline for the DeepStream 8.0 base image (`ilkaybrahim/deepstream_app:8.0`)
- Docker 24+ with NVIDIA Container Toolkit, and Docker Compose v2

### Containers and services

- **`deepstream` (server)**
  - Built from the repository’s `Dockerfile` with `./server` mounted to `/deepstream_app`.
  - Runs the NVIDIA DeepStream pipeline plus a FastAPI application (`server/backend/app/app.py`).
  - Exposes ports:
    - `8554` – RTSP output from the DeepStream pipeline.
    - `5000`, `8888`, `4000:8000` – HTTP/utility ports (e.g. FastAPI, notebooks, or tooling as configured).
  - Shares the `./sockets` directory with the client as `/run/nvunixfd` for IPC.

- **`ds_client` (client)**
  - Uses the `ilkaybrahim/deepstream_app:8.0` image and mounts `./client` to `/client`.
  - Runs `DeepStreamClient` from `client/base_client.py`, which:
    - Watches `/run/nvunixfd` for Unix socket files created by the server pipeline.
    - Builds GStreamer pipelines dynamically for each detected socket.
    - Serves each stream via RTSP on port `8554` (mapped to `8557` on the host).

Both containers are attached to the same `deepstream_network` bridge network for communication.

### Backend API and DeepStream pipeline

- **FastAPI application**
  - Entry point: `server/backend/app/app.py`.
  - Includes router from `server/backend/app/api/v1/endpoints.py` under `/api/v1`.
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
  - Implemented under `server/deepstream/app` (e.g. `deepstream.py`, `spotmanager.py`, etc.).
  - Uses configuration from `server/deepstream/config`:
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
  - NvDeepSORT requires a Re-ID model (`server/deepstream/models/resnet50_market1501.etlt`).
    TensorRT compiles an optimized engine on first run and caches it in the same directory.
  - Writes video frames and metadata (including tracking IDs) to Unix domain sockets in
    `/run/nvunixfd` for the client to consume.

### Data and control flow

1. A user or external service calls the FastAPI backend (`deepstream` container) to add a new RTSP source via `POST /api/v1/add`.
2. The backend’s `pipeline` registers the source in the DeepStream pipeline and starts processing it.
3. The DeepStream application writes buffers/metadata for each stream to a dedicated Unix socket in `/run/nvunixfd`.
4. The `DeepStreamClient` (in the `ds_client` container) watches `/run/nvunixfd`, discovers new sockets, and creates corresponding GStreamer pipelines.
5. The client exposes each stream as an RTSP endpoint on its own `8554` port (mapped to `8557` on the host), allowing external RTSP viewers to connect.

### Configuration and environment

- **Environment variables**
  - `USE_NEW_NVSTREAMMUX` – forwarded into the `ds_client` to toggle DeepStream muxer behavior.
  - `DS_TRACKER` – selects the tracking algorithm (`IOU` / `NvSORT` / `NvDCF` / `NvDeepSORT` / `off`). Default: `NvSORT`.
  - `SAVE_FRAMES`, `DRAWING_TYPE`, `SHOW_INFO_TEXT` – used by `DeepStreamClient` to control frame saving and overlay behavior.

- **Volumes**
  - `./server` → `/deepstream_app` (server container): DeepStream and backend code, configs, and models (including Re-ID model for NvDeepSORT).
  - `./client` → `/client` (client container): DeepStream client code.
  - `./sockets` → `/run/nvunixfd` (both containers): IPC directory for Unix domain sockets.

This architecture cleanly separates **pipeline/processing (server)** from **RTSP presentation (client)** while using shared Unix sockets for high‑throughput, low‑latency inter‑process communication.

For a full explanation of how tracking fits into this architecture — concepts, algorithm comparison, implementation details — see [`docs/guides/tracking-implementation.md`](../guides/tracking-implementation.md).

