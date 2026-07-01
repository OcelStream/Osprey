# Stream Lifecycle

> What happens from the moment you call POST /add to the moment
> frames are flowing — and what happens when a stream ends.

---

## The States

```
                POST /add
                    │
                    ▼
             ┌──────────┐
             │ CREATING  │  spot acquired, elements being wired
             └─────┬─────┘
                   │ pad linked, state synced
                   ▼
             ┌──────────┐
             │  ACTIVE   │  frames flowing, FPS being measured
             └─────┬─────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
  DELETE /remove  EOS     bus error
        │          │          │
        ▼          ▼          ▼
             ┌──────────┐
             │ REMOVING  │  elements being torn down
             └─────┬─────┘
                   │
                   ▼
             ┌──────────┐
             │  REMOVED  │  spot released, socket deleted
             └───────────┘
```

---

## Adding a Stream — Step by Step

### 1. API layer — `POST /api/v1/add`

```json
{
  "stream_id": "camera-north",
  "uri": "rtsp://192.168.1.10:554/stream",
  "rtsp_output_width": 1280,
  "rtsp_output_height": 720
}
```

`endpoints.py` calls `pipeline.add_source(uri, width, height, stream_id)`.

### 2. Lock acquired

`add_source()` acquires `self._lock`. All concurrent add/remove operations
queue behind it. The lock is held for the entire wiring sequence.

### 3. Spot allocated — `SpotManager.acquire()`

`SpotManager` returns a free integer `spot` (0, 1, 2, ...). This spot is the
index used for `nvstreammux` pad naming (`sink_N`) and `nvstreamdemux` pad
naming (`src_N`). If no spot is free (all 64 hardware slots occupied),
`add_source` raises `RuntimeError`.

### 4. Source bin created — `SourceBinFactory`

A `Gst.Bin` wrapping an `nvurisrcbin` is created and added to the pipeline.
`nvurisrcbin` properties are set: URI, reconnect interval, protocol, audio
disabled.

### 5. Preprocessing chain wired

```
nvurisrcbin
    → nvvideoconvert  (resize to model_width × model_height, set nvbuf-memory-type)
    → capsfilter      (enforce NV12 format + dimensions)
    → nvstreammux.sink_N
```

Elements are added to the pipeline and their state is synced with the parent
(`sync_state_with_parent()`). This brings them to `PLAYING` without stopping
the rest of the pipeline.

### 6. Output branch wired

```
nvstreamdemux.src_N
    → queue (q_demux)
    → nvvideoconvert (NV12 → RGBA)
    → capsfilter (RGBA, output_width × output_height)
    → nvdsosd
    → nvvideoconvert (RGBA → NV12)
    → capsfilter (NV12)
    → queue (q_fd)
    → nvunixfdsink  →  /run/nvunixfd/camera-north.sock
```

The FPS probe is added to the OSD sink pad to start measuring throughput.

### 7. StreamRecord stored

```python
self._streams["camera-north"] = StreamRecord(
    stream_id="camera-north",
    uri="rtsp://...",
    spot=N,
    source_bin=src_bin,
    branch_elements={...},
)
```

`StreamRecord` is the single source of truth for everything about this stream.
No parallel lists, no separate dicts.

### 8. Lock released. API returns.

The response goes back to the caller. The socket file `/run/nvunixfd/camera-north.sock`
is created by `nvunixfdsink` as soon as it starts pushing buffers — typically
within a few hundred milliseconds.

### 9. Client picks up the socket

The `ds_client` container's watcher loop detects the new `.sock` file and
creates a receiving pipeline (`nvunixfdsrc` → ... → RTSP mount).

---

## Removing a Stream — Step by Step

### Via `DELETE /api/v1/remove/camera-north`

1. Lock acquired
2. `StreamRecord` looked up by `stream_id`
3. Source teardown:
   - `source_bin.set_state(Gst.State.NULL)` — stops the source
   - `nvstreammux` pad unlinked and released
   - Preprocessing elements removed from pipeline
4. Branch teardown:
   - `nvstreamdemux` request pad unlinked and released
   - All branch elements set to `NULL` and removed from pipeline
5. Spot released back to `SpotManager`
6. `StreamRecord` removed from `self._streams`
7. Lock released

The socket file at `/run/nvunixfd/camera-north.sock` is deleted by
`nvunixfdsink` when it reaches `NULL` state. The client detects the stale
socket and tears down its receiving pipeline.

---

## Natural Endings — EOS and Errors

When a stream ends naturally (file:// source reaches end of file) or a
connection drops permanently (reconnect attempts exhausted), the GStreamer bus
delivers a message to `_on_bus_message`.

The bus handler calls `remove_source(stream_id)` internally — the same
sequence as the REST delete, but triggered from the GLib event loop thread.
The `_lock` protects this from racing with any concurrent API call.

This is why `GET /streams` must delegate to `pipeline.get_active_streams()`
and must never use a separate list — the pipeline removes streams without
notifying the API layer directly. Any API-level list would go stale.

---

## The Lock Discipline

Every path that modifies `self._streams` holds `self._lock`:

| Operation | Acquires lock? |
|-----------|---------------|
| `add_source()` | Yes |
| `remove_source()` | Yes |
| `get_active_streams()` | Yes |
| Bus message handler → `remove_source()` | Yes (via remove_source) |
| `_process_frame` probe (GLib thread) | No — read-only, separate data |

The lock is a `threading.Lock` (not RLock). It is never held across I/O or
blocking calls — only across the in-memory state mutation and GStreamer element
wiring steps.

---

## Spot Reuse

`SpotManager` reuses released spots before allocating fresh ones. If camera
at spot 3 is removed and a new camera is added, it gets spot 3 again.
This avoids gaps in the `nvstreammux` batch tensor — gaps waste GPU computation.

The spot index is separate from the `stream_id`. Two streams can have the same
spot at different points in time, but never simultaneously.
