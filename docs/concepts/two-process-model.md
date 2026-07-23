# The Server / Client Process Model

> Why this system is split into two processes, what each owns,
> and the philosophy behind the boundary between them.

---

## The Split

```
┌──────────────────────────────────────────┐
│  Process 1: osprey-server                 │
│                                          │
│  Owns: GPU inference + REST API          │
│  Users: don't touch this                 │
│  Changes: rarely                         │
└──────────────────────┬───────────────────┘
                       │
              Unix domain sockets
              /run/nvunixfd/*.sock
                       │
┌──────────────────────▼───────────────────┐
│  Process 2: osprey-client                 │
│                                          │
│  Owns: presentation + application logic  │
│  Users: subclass DeepStreamClient here   │
│  Changes: every sprint                   │
└──────────────────────────────────────────┘
```

Both run on the **same host** — `osprey-server` (or `osprey.serve()`) and
`osprey-client` (a `DeepStreamClient` subclass). They are two ordinary
processes, not containers, and they talk over host-local Unix domain sockets.

---

## Why Not One Process?

One process — everything in one Python interpreter — is the simplest option.
The reason it fails here is the **extension problem**.

`DeepStreamClient` is designed as an SDK: users subclass it, override
`_process_frame`, add business logic, and deploy. If the inference pipeline
and the application logic live in the same process, the user must:

- Restart the entire GPU pipeline on every application change
- Risk breaking the inference pipeline with every code change
- Understand GStreamer internals just to change what gets drawn on screen
- Wait for the DeepStream pipeline to warm up (engine load, source wiring) on
  every iteration

Separating them means the user's application is a plain Python program. It
starts in seconds and can be restarted freely. They never touch the inference
process.

---

## Why Not Three Processes (separate API)?

A common pattern in microservices is to put the REST API in its own process:

```
Process 1: FastAPI (API gateway)
Process 2: DeepStream pipeline
Process 3: Client
```

This fails here because **the API is not a gateway — it is a direct controller**.

`add_source()` calls:
```python
src_bin = Gst.Bin.new(bin_name)
self._pipeline.add(src_bin)
pad = self._streammux.get_request_pad(f"sink_{spot}")
src_bin.get_static_pad("src").link(pad)
```

These are in-process Python/C bindings to GStreamer objects. They cannot be
called over a network. Making the API remote means building a command protocol
(gRPC, message queue), handling partial failures differently, and losing the
ability to do atomic rollback on error.

For 5 endpoints at low QPS (adding/removing cameras a few times per minute),
this adds complexity that buys nothing. The API lives in the same process as
the pipeline, served on `:8000`.

---

## What Each Process Owns

### Process 1 — Inference (server)

**Responsibility:** Run batched GPU inference on all streams simultaneously.

| What it owns | Details |
|---|---|
| GStreamer pipeline | nvstreammux → nvinfer → nvstreamdemux |
| Source management | Runtime add/remove of nvurisrcbin elements |
| NVIDIA plugins | nvinfer, nvdsosd, nvunixfdsink, nvstreammux |
| REST API | POST /add, DELETE /remove, GET /streams, GET /health/ready (on :8000) |
| Stream state | `DynamicRTSPPipeline._streams` — the single source of truth |

**Who runs here:** The infrastructure. No user code runs in this process.

**When does it change:** When the inference pipeline, NVIDIA API, or REST
contract changes. This is rare.

### Process 2 — Presentation (client)

**Responsibility:** Receive inference results and deliver them to consumers.

| What it owns | Details |
|---|---|
| Socket discovery | Watches `/run/nvunixfd` for new sockets |
| Per-stream pipelines | nvunixfdsrc → OSD → encoder → RTSP |
| Application logic | User's `_process_frame` override |
| Drawing | Bounding boxes, zones, HUDs |
| RTSP output | GstRtspServer, one mount per stream (on :8554) |

**Who runs here:** User application code. `DeepStreamClient` subclasses.

**When does it change:** Every feature sprint. Drawing changes, new analytics,
new output formats — all happen here without touching the server process.

---

## The Boundary — What Crosses It

The only thing that crosses the boundary between the two processes is:

1. **GPU frame buffers** — via Unix domain socket file descriptor passing (`nvunixfdsink` → `nvunixfdsrc`)
2. **DeepStream metadata** — serialised into the buffer stream by `serialize_meta.so`, deserialised by `deserialize_meta.so`
3. **Socket files** — the `.sock` files in `/run/nvunixfd` that signal a new stream is available

Nothing else crosses. The processes share no database, no message queue, no
in-process objects. The only coupling is the socket file and the metadata
format.

This means:
- The client process can crash and restart without affecting inference
- The server process can restart without affecting the client (it will
  reconnect to new sockets)
- The two processes can be on different versions as long as the metadata
  serialization format is compatible

---

## The Dependency Direction

```
Client process  ──depends on──►  Server process sockets
                                 (waits for /run/nvunixfd/*.sock)
```

The client waits for the server's sockets to appear before it starts a
receiving pipeline. The server never knows about the client. This is the
correct dependency direction: the consumer depends on the producer, not the
other way around.

---

## The Socket Directory

`/run/nvunixfd` is a host-local directory (created once by
`sudo osprey-bootstrap`) that both processes use. This is the only shared
state. When the server creates `stream_id.sock`, the client's watcher sees it
and starts a pipeline. When the server deletes it, the client detects the stale
socket and tears down.

---

## Summary

| Question | Answer |
|----------|--------|
| Why two processes? | Separate stable inference from changeable application logic |
| Why not one? | Users need to extend and iterate without touching the GPU pipeline |
| Why not three? | API is a direct in-process controller of GStreamer — not a gateway |
| What crosses the boundary? | Only GPU buffers (fd passing) + metadata (serialised) + socket files |
| Who depends on whom? | Client depends on server. Server doesn't know client exists. |
