# FastAPI Lifespan — Pipeline Startup & Readiness

> **Status:** Implemented  
> **Files changed:** `server/backend/app/app.py`, `server/deepstream/app/deepstream.py`  
> **Replaces:** `time.sleep(3)` race condition  
> **Related:** [Startup Race Condition (detail)](../local/startup-race-condition.md) · [Architecture Proposal §6](../local/architecture-proposal.md#6-control-plane--fastapi--pipeline-lifecycle)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Implementation](#3-implementation)
4. [Startup Sequence](#4-startup-sequence)
5. [Why `lifespan` over `@app.on_event`](#5-why-lifespan-over-appon_event)
6. [Shutdown](#6-shutdown)
7. [What This Enables Next](#7-what-this-enables-next)

---

## 1. Problem Statement

The original `app.py` started the DeepStream pipeline at **module import time**:

```python
# app.py — old (DO NOT USE)
threading.Thread(target=pipeline.start, daemon=True).start()
time.sleep(3)   # blindly hope the pipeline is ready in 3 seconds
```

This had two compounding problems:

### Problem A — Race condition

Pipeline startup time depends on GPU state, TensorRT engine loading, and system
load. It is not a fixed constant.

```
Slow GPU (race condition triggers):
  t=0s   pipeline.start() thread begins
  t=3s   Main thread wakes, FastAPI starts accepting requests
  t=3s   Client hits POST /add → 400 "pipeline not playing"
  t=12s  Pipeline reaches PLAYING  ← too late

Fast GPU (works, but wastes time):
  t=0s   pipeline.start() thread begins
  t=1s   Pipeline reaches PLAYING
  t=3s   Main thread wakes  ← 2 seconds wasted waiting for nothing
```

### Problem B — Module-level side effects

Code at module level runs at import time. Under `gunicorn` or `uvicorn` with
multiple workers, the module is imported once per worker — so the pipeline
starts once per worker. Two pipeline instances competing over the same GPU and
the same Unix sockets is undefined behavior.

---

## 2. Solution Overview

The fix has two parts that work together:

| Part | Where | What it does |
|------|-------|-------------|
| `threading.Event` | `deepstream.py` | Pipeline signals when it reaches `PLAYING` |
| FastAPI `lifespan` | `app.py` | Startup code runs exactly once, before any request is accepted |

Together they guarantee: **no HTTP request is accepted until the pipeline
signals it is ready**, with a 30-second hard timeout if something is broken.

---

## 3. Implementation

### 3.1 `DynamicRTSPPipeline` — `deepstream.py`

**In `__init__`** — add the readiness event next to the existing lock:

```python
# Serializes concurrent add/remove from different threads
self._lock = threading.Lock()

# Signals that the pipeline has reached PLAYING state
self._ready = threading.Event()
```

**In `start()`** — set the event immediately after `set_state(PLAYING)`:

```python
def start(self, ...):
    ...
    time.sleep(1)
    self._pipeline.set_state(Gst.State.PLAYING)
    self._ready.set()          # ← pipeline is now in PLAYING state
    try:
        self._loop.run()       # blocks here — GLib main loop runs
    ...
```

`self._ready.set()` is called once, before `self._loop.run()` blocks. Any
thread calling `self._ready.wait()` is unblocked at this exact moment.

### 3.2 `app.py` — FastAPI lifespan

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.app.api.v1.endpoints import router
from backend.app.core.context import pipeline
import threading


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=pipeline.start, daemon=True)
    thread.start()
    pipeline._ready.wait(timeout=30)
    yield
    # pipeline.stop()  ← optional graceful shutdown


app = FastAPI(title="DeepStream API", lifespan=lifespan)
app.include_router(router, prefix="/api/v1", tags=["deepstream"])
```

Key points:

- The pipeline thread is started inside `lifespan`, not at module level —
  it runs exactly once, regardless of how many times the module is imported.
- `pipeline._ready.wait(timeout=30)` blocks until the pipeline signals ready.
  If the pipeline never reaches `PLAYING` within 30 seconds, the server fails
  to start, which is visible immediately rather than silently serving 400s.
- `yield` is where FastAPI runs. Everything before `yield` is startup;
  everything after is shutdown.
- The router is still registered at module level (`include_router`) — this is
  safe because registration only adds routes to a table, it doesn't execute
  any pipeline code.

---

## 4. Startup Sequence

```
uvicorn starts
    │
    ▼
app.py imported
    │  pipeline object created (context.py)
    │  router routes registered
    │
    ▼
lifespan() begins
    │
    ├──► Thread: pipeline.start()
    │        │
    │        ├── Gst.init()
    │        ├── GStreamer elements created
    │        ├── time.sleep(1)
    │        ├── set_state(PLAYING)
    │        ├── self._ready.set()  ──────────────────────────────────────┐
    │        └── self._loop.run()  (blocks here)                          │
    │                                                                     │
    ├── pipeline._ready.wait(timeout=30)  ◄── unblocked by set() ────────┘
    │
    ▼
yield  ← FastAPI begins accepting HTTP requests
    │
    │   [server runs]
    │
    ▼
shutdown signal
    │
    ├── (optional) pipeline.stop()
    └── lifespan() exits
```

The `yield` point acts as a gate. No request handler is reachable until
`pipeline._ready.wait()` returns.

---

## 5. Why `lifespan` over `@app.on_event`

FastAPI deprecated `@app.on_event("startup")` in favour of `lifespan` in
version 0.93.0 (released Feb 2023). The `lifespan` pattern:

| | `@app.on_event` | `lifespan` |
|---|---|---|
| Status | Deprecated | Recommended |
| Startup + shutdown in one place | No (two decorators) | Yes (before/after `yield`) |
| Exception propagates to uvicorn | Partially | Yes — startup failure aborts the server |
| Works with Starlette directly | No | Yes |
| Async context manager | No | Yes |

The old commented-out pattern in the original file:

```python
# @app.on_event("startup")
# async def startup_event():
#     await pipeline.rabbitmq_manager.connect()
#     asyncio.create_task(pipeline._processing_worker_loop())
```

Any future async startup work (RabbitMQ connection, processing loops) should
go inside the `lifespan` function before the `yield`, not in a separate
`@app.on_event` decorator.

---

## 6. Shutdown

> **Status: Implemented.**

The `lifespan` function calls `pipeline.stop()` after `yield` — this runs
when uvicorn receives a shutdown signal (SIGTERM, SIGINT, or Ctrl-C).

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=pipeline.start, daemon=True)
    thread.start()
    pipeline._ready.wait(timeout=30)
    yield
    pipeline.stop()
```

`DynamicRTSPPipeline.stop()` is implemented in `deepstream.py`:

```python
def stop(self) -> None:
    """Signal the GLib main loop to quit and set the pipeline to NULL."""
    self._loop.quit()
    self._pipeline.set_state(Gst.State.NULL)
    logger.info("Pipeline stopped")
```

`self._loop.quit()` wakes the GLib main loop from `_loop.run()`, causing
`start()` to exit its `try` block and fall through to the `finally` clause
(which also calls `set_state(NULL)`). `stop()` therefore reaches `NULL` state
from two directions — `_loop.quit()` path and the `finally` path — which is
safe because setting `NULL` on an already-`NULL` pipeline is a no-op.

The pipeline thread is a daemon thread (`daemon=True`). If the process is
killed hard (SIGKILL, OOM) rather than via uvicorn's graceful shutdown, the
daemon thread is killed automatically by the OS without `stop()` being called.
In-flight stream teardown is not guaranteed in that case.

---

## 7. What This Enables Next

> **Status: Implemented.**

### `/health/ready` endpoint

Added to `server/backend/app/api/v1/endpoints.py`:

```python
@router.get("/health/ready")
def ready():
    """Readiness probe — returns 200 only after the pipeline has reached PLAYING state."""
    if pipeline._ready.is_set():
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Pipeline not ready yet")
```

Returns `200 {"status": "ready"}` once the pipeline is `PLAYING`.
Returns `503` with a detail message until then.

This endpoint is intentionally simple — it checks a single boolean flag.
There is no lock contention and the response is always fast.

### Docker Compose `healthcheck` + `depends_on`

`docker-compose.yml` now has:

```yaml
deepstream:
  ...
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/ready"]
    interval: 10s
    timeout: 5s
    retries: 5
    start_period: 30s

ds_client:
  depends_on:
    deepstream:
      condition: service_healthy
  ...
```

Docker Compose will:
1. Start the `deepstream` container and begin polling `/api/v1/health/ready`.
2. Mark it `healthy` only after 5 consecutive `200` responses.
3. Only then start the `ds_client` container.

This eliminates the last remaining startup race — the client will never try
to read Unix sockets before the server pipeline is `PLAYING`.

---

## Files Changed

| File | Change |
|------|--------|
| [app.py](../../server/backend/app/app.py) | Replaced module-level `Thread + sleep` with `lifespan` context manager |
| [deepstream.py](../../server/deepstream/app/deepstream.py) | Added `self._ready = threading.Event()` in `__init__`; added `self._ready.set()` in `start()` after `set_state(PLAYING)` |
