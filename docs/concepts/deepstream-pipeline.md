# The DeepStream Pipeline — Concepts

> How GStreamer and NVIDIA DeepStream work, and how this project uses them.

---

## GStreamer in One Paragraph

GStreamer is a pipeline framework. You create **elements** (sources, filters,
sinks), connect them via **pads** (input/output ports), and data flows through
them as **buffers**. The framework handles threading, synchronisation, and
negotiating the format (caps) that flows between elements. DeepStream is a set
of NVIDIA-specific GStreamer plugins that run on GPU memory instead of CPU
memory.

---

## The Element Chain

The full server pipeline for N streams looks like this:

```
[nvurisrcbin_0]──┐
[nvurisrcbin_1]──┤
[nvurisrcbin_2]──┤──► [nvstreammux] ──► [nvinfer] ──► [nvstreamdemux]
       ...        │                                          │
[nvurisrcbin_N]──┘                              ┌───────────┼───────────┐
                                                │           │           │
                                          [branch_0]  [branch_1]  [branch_N]
                                                │           │           │
                                          [fdsink_0] [fdsink_1] [fdsink_N]
```

### Sources — `nvurisrcbin`

Each camera stream has one `nvurisrcbin`. This NVIDIA element handles:
- RTSP connection and reconnection (configurable retry interval)
- RTP protocol selection (TCP fallback)
- File looping for file:// URIs
- Format decoding and output as NVMM GPU buffers

Between each `nvurisrcbin` and `nvstreammux` there is a short preprocessing
chain:

```
nvurisrcbin → nvvideoconvert → capsfilter(NV12, model_width × model_height) → nvstreammux sink_N
```

The `nvvideoconvert` resizes and converts the frame to NV12 at the model's
input dimensions. The `capsfilter` enforces the format so GStreamer never
sends a wrong-sized frame to the muxer.

### Muxer — `nvstreammux`

`nvstreammux` is the key to efficient multi-stream inference. It collects one
frame from each active source and packs them into a single **batch tensor**
that fits the GPU's execution model.

Key properties:
- `batch-size` — how many sources can be batched at once (set to 64, the hardware limit)
- `batched-push-timeout` — how long to wait for a full batch before sending a partial one

This is why running 10 streams is not 10× slower than running 1 stream: the
GPU processes all 10 frames in one inference call.

### Inference — `nvinfer`

`nvinfer` runs the TensorRT engine against the batch. Each frame gets bounding
boxes, class IDs, and confidence scores attached as DeepStream metadata
(`NvDsBatchMeta`). The metadata travels with the buffer downstream — no
separate channel needed.

Multiple `nvinfer` elements can be chained: a PGIE (Primary GIE) for detection,
then one or more SGIEs (Secondary GIEs) for classification on detected objects.
Which GIEs are active is controlled by `GIE_N_CONFIG` environment variables.

### Demuxer — `nvstreamdemux`

`nvstreamdemux` is the inverse of `nvstreammux`. It splits the batch back into
N individual streams, one per source pad (`src_0`, `src_1`, ..., `src_N`).
Each stream gets its own output branch.

### Output Branch — per stream

Each stream's output branch wires:

```
nvstreamdemux src_N
    → queue (q_demux)
    → nvvideoconvert (NV12 → RGBA)   ← DeepStreamElementFactory, nvbuf-memory-type set
    → capsfilter (RGBA, output_w × output_h)
    → nvdsosd                         ← draws bounding boxes if enabled
    → nvvideoconvert (RGBA → NV12)   ← _create_element, NO nvbuf-memory-type (IPC boundary)
    → capsfilter (NV12)
    → queue (q_fd)
    → nvunixfdsink                    ← writes to /run/nvunixfd/<stream_id>.sock
```

The RGBA conversion is needed for `nvdsosd` (OSD drawing requires CPU-readable
RGBA). The second NV12 conversion produces the format expected by
`nvunixfdsink`. Note: the second converter intentionally has no explicit
`nvbuf-memory-type` — see [ipc-unix-sockets.md](ipc-unix-sockets.md).

---

## Runtime Source Management

Standard GStreamer pipelines are static. Adding a source requires stopping
the pipeline, adding elements, and restarting — which interrupts all other
streams.

`DynamicRTSPPipeline` solves this with a carefully ordered sequence:

**Add:**
1. Acquire a free spot index from `SpotManager`
2. Create `nvurisrcbin` + preprocessing elements
3. Add elements to the pipeline and sync their state with the parent
4. Link the preprocessing chain to `nvstreammux sink_N`
5. Create the output branch and link from `nvstreamdemux src_N`

**Remove:**
1. Flush and set source to `NULL` state
2. Unlink from `nvstreammux`
3. Release preprocessing elements
4. Unlink from `nvstreamdemux` and release the request pad
5. Set output branch elements to `NULL` and remove them
6. Release the spot back to `SpotManager`

All of this happens while the rest of the pipeline is `PLAYING`. The other
streams never pause. This is only safe because every add/remove operation is
serialised by `self._lock`.

---

## Memory Model — NVMM

DeepStream buffers live in **NVMM** (NVIDIA Memory Manager) — GPU-side memory
that is not accessible from the CPU without an explicit copy. This is why:

- `nvvideoconvert` needs `nvbuf-memory-type` set to tell it which NVMM pool to use
- `nvdsosd` metadata drawing works via DeepStream's metadata API (not direct pixel access)
- Moving frames between processes uses file descriptor passing (not memcpy)

On x86, `NVBUF_MEM_CUDA_UNIFIED` allocates in unified memory that both CPU
and GPU can address — necessary for the CPU-side metadata operations. On Jetson,
`NVBUF_MEM_DEFAULT` uses the Tegra NVMM pool.

---

## The GLib Main Loop

GStreamer's bus (event/message system) and all its async callbacks run inside a
**GLib `MainLoop`**. This is a separate event loop from Python's asyncio and from
FastAPI's uvicorn.

`DynamicRTSPPipeline.start()` calls `self._loop.run()` — this blocks the
calling thread permanently, running the GLib event loop. This is why `start()`
runs in a daemon thread, and why `threading.Event._ready` is needed to signal
FastAPI that the pipeline is up before accepting requests.
