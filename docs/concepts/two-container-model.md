# The Two-Container Model

> Why this system is split into two containers, what each owns,
> and the philosophy behind the boundary between them.

---

## The Split

```
┌──────────────────────────────────────────┐
│  Container 1: deepstream (server)         │
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
│  Container 2: ds_client (client)          │
│                                          │
│  Owns: presentation + application logic  │
│  Users: subclass DeepStreamClient here   │
│  Changes: every sprint                   │
└──────────────────────────────────────────┘
```

---

## Why Not One Container?

One container — everything in one process — is the simplest option. The reason
it fails here is the **extension problem**.

`DeepStreamClient` is designed as an SDK: users subclass it, override
`_process_frame`, add business logic, and deploy. If the inference pipeline
and the application logic live in the same container, the user must:

- Modify and rebuild the NVIDIA DeepStream container on every application change
- Risk breaking the inference pipeline with every code change
- Understand GStreamer internals just to change what gets drawn on screen
- Be blocked by the DeepStream container's build time (several minutes) on
  every iteration

Separating them means the user's application container is a plain Python image.
It builds in seconds. They never touch the inference container.

---

## Why Not Three Containers (separate API)?

A common pattern in microservices is to put the REST API in its own container:

```
Container 1: FastAPI (API gateway)
Container 2: DeepStream pipeline
Container 3: Client
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
this adds complexity that buys nothing.

---

## What Each Container Owns

### Container 1 — Inference (server)

**Responsibility:** Run batched GPU inference on all streams simultaneously.

| What it owns | Details |
|---|---|
| GStreamer pipeline | nvstreammux → nvinfer → nvstreamdemux |
| Source management | Runtime add/remove of nvurisrcbin elements |
| NVIDIA plugins | nvinfer, nvdsosd, nvunixfdsink, nvstreammux |
| REST API | POST /add, DELETE /remove, GET /streams, GET /health/ready |
| Stream state | `DynamicRTSPPipeline._streams` — the single source of truth |

**Who runs here:** The infrastructure. No user code runs in this container.

**When does it change:** When the inference pipeline, NVIDIA API, or REST
contract changes. This is rare.

### Container 2 — Presentation (client)

**Responsibility:** Receive inference results and deliver them to consumers.

| What it owns | Details |
|---|---|
| Socket discovery | Watches `/run/nvunixfd` for new sockets |
| Per-stream pipelines | nvunixfdsrc → OSD → encoder → RTSP |
| Application logic | User's `_process_frame` override |
| Drawing | Bounding boxes, zones, HUDs |
| RTSP output | GstRtspServer, one mount per stream |

**Who runs here:** User application code. `DeepStreamClient` subclasses.

**When does it change:** Every feature sprint. Drawing changes, new analytics,
new output formats — all happen here without touching Container 1.

---

## The Boundary — What Crosses It

The only thing that crosses the boundary between containers is:

1. **GPU frame buffers** — via Unix domain socket file descriptor passing (`nvunixfdsink` → `nvunixfdsrc`)
2. **DeepStream metadata** — serialised into the buffer stream by `serialize_meta.so`, deserialised by `deserialize_meta.so`
3. **Socket files** — the `.sock` files in the shared `./sockets` volume that signal a new stream is available

Nothing else crosses. The containers share no database, no message queue, no
in-process objects. The only coupling is the socket file and the metadata
format.

This means:
- The client container can crash and restart without affecting inference
- The server container can restart without affecting the client (it will
  reconnect to new sockets)
- The two containers can be on different versions as long as the metadata
  serialization format is compatible

---

## The Dependency Direction

```
Client container  ──depends on──►  Server container sockets
                                   (service_healthy condition in docker-compose)
```

The client waits for the server to be healthy before starting. The server
never knows about the client. This is the correct dependency direction:
the consumer depends on the producer, not the other way around.

---

## The Shared Volume

```yaml
# docker-compose.yml
volumes:
  - ./sockets:/run/nvunixfd   # shared between deepstream and ds_client
```

`./sockets` on the host is mounted as `/run/nvunixfd` in both containers. This
is the only shared state. When the server creates `stream_id.sock`, the client's
watcher sees it and starts a pipeline. When the server deletes it, the client
detects the stale socket and tears down.

---

## Summary

| Question | Answer |
|----------|--------|
| Why two containers? | Separate stable inference from changeable application logic |
| Why not one? | Users need to extend and iterate without touching the GPU pipeline |
| Why not three? | API is a direct in-process controller of GStreamer — not a gateway |
| What crosses the boundary? | Only GPU buffers (fd passing) + metadata (serialised) + socket files |
| Who depends on whom? | Client depends on server. Server doesn't know client exists. |
