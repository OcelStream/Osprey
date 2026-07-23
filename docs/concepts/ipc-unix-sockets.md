# IPC — Unix Sockets and Zero-Copy GPU Buffers

> How frames travel from the server process to the client process
> without being copied through the CPU.

---

## The Problem with Copying

A 1280×720 RGBA frame is 3.5 MB. At 30 FPS across 10 streams, that is
**1.05 GB/s** of frame data. Copying that through the CPU (read GPU memory,
write to shared memory, read in client process) would saturate a CPU memory
bus and add latency.

The system avoids this entirely using GPU buffer file descriptor passing.

---

## How `nvunixfdsink` / `nvunixfdsrc` Work

```
Server process (osprey-server)                Client process (osprey-client)
                                              
 GStreamer pipeline                            GStreamer pipeline
 ...                                           nvunixfdsrc
 nvunixfdsink ──► /run/nvunixfd/X.sock ──────► (fd passing)
                                               ...
```

`nvunixfdsink` is an NVIDIA GStreamer sink that:
1. Creates a Unix domain socket at a configured path (`socket-path`)
2. For each buffer, passes a **CUDA IPC file descriptor** over the socket

`nvunixfdsrc` is the corresponding source that:
1. Connects to the socket
2. Receives the file descriptor
3. Opens the GPU memory referenced by that fd (`cudaIpcOpenMemHandle`)
4. Makes the buffer available in its own pipeline — **without any copy**

The GPU memory is allocated once in the server. The client receives a handle
to the same physical GPU memory. No memcpy occurs.

### What travels over the socket

```
┌─────────────────────────────────────┐
│  NvBufSurface descriptor            │  GPU buffer handle
│  CUDA IPC handle                    │  pointer to GPU memory
│  Serialized NvDsBatchMeta           │  bounding boxes, class IDs, etc.
│  Timestamp                          │
└─────────────────────────────────────┘
```

The metadata (inference results — bounding boxes, class IDs, confidence scores)
travels alongside the buffer as serialised bytes. The server uses
`serialize_meta.so` to pack the `NvDsBatchMeta` structure into the buffer.
The client uses `deserialize_meta.so` to unpack it. No separate IPC channel
is needed for metadata.

---

## The Socket Directory

Both processes use the same directory, `/run/nvunixfd`, on the host. This
serves two purposes:

1. **Socket file discovery** — the client's watcher thread calls `os.listdir("/run/nvunixfd")` to detect new streams
2. **Socket connection** — `nvunixfdsrc` connects to the socket path to receive buffers

The directory is a plain host-local path (created by `sudo osprey-bootstrap`).
The socket files themselves are not the data channel — they are just the
rendezvous point. The actual GPU buffer handles flow through the socket
connection.

---

## The Shared IPC Namespace

CUDA IPC (`cudaIpcOpenMemHandle`) shares GPU memory between processes using the
OS's IPC namespace. Because the server and the client run as two ordinary
processes on the **same host**, they already share the host's IPC namespace —
so the client can open memory handles created by the server with no extra
configuration. (This is the constraint that a container deployment would have
had to recover with an `ipc: host` setting; running natively on one host, it
is automatic.)

---

## The `conv2` Memory Type Rule

The last `nvvideoconvert` before `nvunixfdsink` (called `conv2` in the output
branch) must **not** have `nvbuf-memory-type` set explicitly:

```python
# output branch — _build_output_branch
"conv2": self._create_element("nvvideoconvert", f"conv2_{stream_id}"),
# ← intentionally _create_element, NOT element_factory.nvvideoconvert()
```

If `NVBUF_MEM_CUDA_UNIFIED` is set on `conv2`, the output buffer uses unified
memory. `nvunixfdsink` then passes a unified-memory IPC handle. On the client
side, `nvunixfdsrc` calls `cudaIpcOpenMemHandle` — but unified memory handles
behave differently from device memory handles, causing:

```
NvBufSurfaceCudaMemImport: cudaIpcOpenMemHandle err: 1
streaming stopped, reason error (-5)
```

Leaving `nvbuf-memory-type` unset lets GStreamer negotiate the memory type
that `nvunixfdsink` and the IPC mechanism prefer — device memory, not unified
memory.

**Rule:** Any `nvvideoconvert` whose output feeds `nvunixfdsink` must use
default memory type (no explicit `nvbuf-memory-type`).

---

## Metadata Serialization

The bounding box data (inference results) is attached to each buffer as
`NvDsBatchMeta` — a C struct managed by DeepStream. This struct contains:
- Per-frame metadata (`NvDsFrameMeta`)
- Per-object metadata (`NvDsObjectMeta`)
- Bounding box coordinates, class IDs, confidence scores
- Tracker IDs (if a tracker is in the pipeline)

To pass this across a process boundary, it must be serialised into bytes.
`nvunixfdsink` calls `serialize_meta.so` (a custom C shared library in
`osprey/server/deepstream/lib/`) to pack the struct into the buffer. `nvunixfdsrc`
calls `deserialize_meta.so` to unpack it on the client side.

The path to the serialization library is configured via:
```
DS_META_SERIALIZATION_LIB=osprey/server/deepstream/lib/serialize_meta.so
```

---

## Why Unix Sockets, Not TCP or Shared Memory

| Mechanism | Throughput | Metadata | NVMM support | Complexity |
|-----------|-----------|----------|--------------|------------|
| **nvunixfdsink** (this system) | High | Native | Yes | Low |
| TCP (rtsp between processes) | Medium | Separate channel | No | Medium |
| shmsink (CPU shared memory) | High | Manual | No | Medium |
| ZeroMQ (Savant approach) | Medium | Custom protocol | No | High |

`nvunixfdsink` wins because it is NVIDIA's official mechanism for exactly this
use case. It handles NVMM buffers and metadata serialization out of the box.
The only constraint is that server and client must be on the **same physical
host** (they share the same GPU). Multi-host distribution would require a
different IPC mechanism.
