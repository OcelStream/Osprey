# Element Factory — Centralised GStreamer Element Creation

> **Status:** Implemented  
> **New file:** `osprey/server/deepstream/element_factory.py`  
> **Changed:** `osprey/server/deepstream/pipeline.py` — `_attach_preprocessing`, `_build_output_branch`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [The Code Before — Full Inline Blocks](#2-the-code-before--full-inline-blocks)
3. [The Code After — With Factory](#3-the-code-after--with-factory)
4. [Solution Overview](#4-solution-overview)
5. [DeepStreamElementFactory — API Reference](#5-deepstreamelementfactory--api-reference)
6. [Platform Support — x86 vs Jetson](#6-platform-support--x86-vs-jetson)
7. [Where the Factory Is Used](#7-where-the-factory-is-used)
8. [The conv2 Exception — IPC Boundary Rule](#8-the-conv2-exception--ipc-boundary-rule)
9. [Adding a New Helper Method](#9-adding-a-new-helper-method)

---

## 1. Problem Statement

Before this refactoring, `nvbuf-memory-type` and queue settings were set
inline in three separate methods:

```
_attach_preprocessing       → conv.set_property("nvbuf-memory-type", CUDA_UNIFIED)
_build_output_branch        → conv1.set_property("nvbuf-memory-type", CUDA_UNIFIED)
                            → for q in (q_demux, q_fd): q.set_property("leaky", 0) ...
```

This created two concrete problems:

### Problem A — Platform scatter

When porting to Jetson, `nvbuf-memory-type` must change from
`NVBUF_MEM_CUDA_UNIFIED` to `NVBUF_MEM_DEFAULT`. With the old code, every
occurrence had to be found and changed manually — across multiple methods,
spread across 150+ lines of pipeline code. A single missed occurrence produces
a silent memory mismatch that is hard to diagnose.

### Problem B — Queue settings drift

Queue elements (`q_demux`, `q_fd`) had `leaky`, `max-size-buffers`,
`max-size-bytes`, and `max-size-time` set in a loop inside `_build_output_branch`.
Any new queue added elsewhere in the file would get different defaults unless
the developer found and copied the loop — a copy-paste trap.

---

## 2. The Code Before — Full Inline Blocks

These are the exact blocks that existed in `pipeline.py` before this
refactoring. They are preserved here so you can see precisely what was
duplicated and why the factory was introduced.

### `_attach_preprocessing` — before

```python
# pipeline.py — _attach_preprocessing (old)

conv = self._create_element("nvvideoconvert", f"conv_{stream_id}")
caps = self._create_element("capsfilter", f"capsfilter_{stream_id}")

# ← platform-specific property set inline — must find this when porting to Jetson
conv.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))

# ← caps set inline — model dimensions come from config
caps.set_property(
    "caps",
    Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM), format=NV12, "
        f"width={self._config.model_width}, height={self._config.model_height}"
    ),
)
```

### `_build_output_branch` — before

```python
# pipeline.py — _build_output_branch (old)

elems = {
    "q_demux": self._create_element("queue",          f"q_demux_{stream_id}"),
    "conv1":   self._create_element("nvvideoconvert",  f"conv1_{stream_id}"),
    "caps_rgba": self._create_element("capsfilter",    f"caps_rgba_{stream_id}"),
    "osd":     self._create_element("nvdsosd",         f"osd_{stream_id}"),
    "conv2":   self._create_element("nvvideoconvert",  f"conv2_{stream_id}"),
    "caps_nv12": self._create_element("capsfilter",    f"caps_nv12_{stream_id}"),
    "q_fd":    self._create_element("queue",           f"q_fd_{stream_id}"),
    "fdsink":  self._create_element("nvunixfdsink",    f"fdsink_{stream_id}"),
}

# ← same platform-specific property again — second occurrence to find and change
elems["conv1"].set_property(
    "nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED)
)

# ← caps set inline
elems["caps_rgba"].set_property(
    "caps",
    Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM),format=RGBA,width={width},height={height}"
    ),
)
elems["caps_nv12"].set_property(
    "caps",
    Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"),
)

elems["osd"].set_property("display-bbox", 0)
elems["osd"].set_property("display-mask", 0)
elems["osd"].set_property("display-text", 0)

# ← queue settings copy-pasted as a loop — must copy this loop every time
#   a new queue is added anywhere in the file
for key in ("q_demux", "q_fd"):
    q = elems[key]
    q.set_property("leaky", 0)
    q.set_property("max-size-buffers", 20)
    q.set_property("max-size-bytes", 0)
    q.set_property("max-size-time", 5_000)
```

### What was wrong

| Duplicated thing | Occurrences before | Risk |
|------------------|--------------------|------|
| `nvbuf-memory-type = NVBUF_MEM_CUDA_UNIFIED` | 2 (one per method) | Miss one when porting to Jetson → silent crash |
| `pyds.NVBUF_MEM_CUDA_UNIFIED` import | Used in `pipeline.py` just for this | Unnecessary GStreamer/pyds coupling |
| Queue 4-property block | 1 loop covering 2 queues | Any new queue misses it unless the loop is found and copied |
| Caps set as raw string inline | 3 occurrences | No central place to validate format strings |

---

## 3. The Code After — With Factory

### `_attach_preprocessing` — after

```python
# pipeline.py — _attach_preprocessing (now)

conv = self._element_factory.nvvideoconvert(f"conv_{stream_id}")
caps = self._element_factory.capsfilter(
    f"capsfilter_{stream_id}",
    f"video/x-raw(memory:NVMM), format=NV12, "
    f"width={self._config.model_width}, height={self._config.model_height}",
)
```

6 lines → 2. `nvbuf-memory-type` is set inside the factory — one place.

### `_build_output_branch` — after

```python
# pipeline.py — _build_output_branch (now)

elems = {
    "q_demux":   self._element_factory.queue(f"q_demux_{stream_id}"),
    "conv1":     self._element_factory.nvvideoconvert(f"conv1_{stream_id}"),
    "caps_rgba": self._element_factory.capsfilter(
        f"caps_rgba_{stream_id}",
        f"video/x-raw(memory:NVMM),format=RGBA,width={width},height={height}",
    ),
    "osd":       self._create_element("nvdsosd", f"osd_{stream_id}"),
    "conv2":     self._create_element("nvvideoconvert", f"conv2_{stream_id}"),  # ← IPC boundary, see §8
    "caps_nv12": self._element_factory.capsfilter(
        f"caps_nv12_{stream_id}",
        "video/x-raw(memory:NVMM),format=NV12",
    ),
    "q_fd":      self._element_factory.queue(f"q_fd_{stream_id}"),
    "fdsink":    self._create_element("nvunixfdsink", f"fdsink_{stream_id}"),
}

elems["osd"].set_property("display-bbox", 0)
elems["osd"].set_property("display-mask", 0)
elems["osd"].set_property("display-text", 0)
```

The `nvbuf-memory-type` block is gone. The 4-property queue loop is gone.
The caps inline assignments are gone. OSD display flags remain — they are
element-specific, not a factory concern.

---

## 4. Solution Overview

`DeepStreamElementFactory` is a thin wrapper around `Gst.ElementFactory.make`
that applies platform-aware and consistent defaults at the point of creation.

| Before | After |
|--------|-------|
| `conv.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))` | `self._element_factory.nvvideoconvert(name)` |
| `caps.set_property("caps", Gst.Caps.from_string(...))` | `self._element_factory.capsfilter(name, caps_string)` |
| 4-line loop per queue (`leaky`, `max-size-*`) | `self._element_factory.queue(name)` |
| `pyds` imported in `pipeline.py` for memory type | `pyds` used only in `element_factory.py` |

Porting to Jetson now requires changing one line: the `platform` argument
passed to `DeepStreamElementFactory(platform=...)`.

---

## 5. DeepStreamElementFactory — API Reference

**File:** `osprey/server/deepstream/element_factory.py`

### `__init__(platform: str = "x86")`

Sets the internal `_mem_type` used by `nvvideoconvert()`:

| `platform` | `nvbuf-memory-type` | Use case |
|------------|---------------------|----------|
| `"x86"` (default) | `NVBUF_MEM_CUDA_UNIFIED` | Desktop GPU (dGPU) |
| `"jetson"` | `NVBUF_MEM_DEFAULT` | Jetson / iGPU |

---

### `make(factory, name) → Gst.Element`

Primitive. Creates any GStreamer element by factory name. Raises `RuntimeError`
if the element cannot be created (plugin not installed, wrong name), instead of
returning `None` silently.

```python
elem = self._element_factory.make("nvdsosd", "osd_cam1")
```

Use `make()` for elements that need no shared default properties
(`nvdsosd`, `nvunixfdsink`, `nvinfer`, etc.).

---

### `nvvideoconvert(name) → Gst.Element`

Creates an `nvvideoconvert` element with `nvbuf-memory-type` set for the
target platform.

```python
conv = self._element_factory.nvvideoconvert("conv_cam1")
```

**Only use this for converters that feed into further GPU processing** —
see [Section 8](#8-the-conv2-exception--ipc-boundary-rule) for the IPC
boundary exception.

---

### `capsfilter(name, caps_string) → Gst.Element`

Creates a `capsfilter` element with caps pre-applied from a string.

```python
caps = self._element_factory.capsfilter(
    "caps_nv12_cam1",
    "video/x-raw(memory:NVMM),format=NV12",
)
```

---

### `queue(name, max_buffers=20, max_time=5_000) → Gst.Element`

Creates a `queue` element with consistent defaults:

| Property | Value | Meaning |
|----------|-------|---------|
| `leaky` | `0` | No dropping — apply backpressure |
| `max-size-buffers` | `20` (overridable) | Buffer count limit |
| `max-size-bytes` | `0` | No byte limit |
| `max-size-time` | `5_000` ns (overridable) | Time-based cap (5 µs) |

```python
q = self._element_factory.queue("q_demux_cam1")

# Custom limits for a slower pipeline section:
q = self._element_factory.queue("q_slow", max_buffers=10, max_time=16_000)
```

---

## 6. Platform Support — x86 vs Jetson

`nvbuf-memory-type` controls where NVIDIA video buffers are allocated:

| Constant | Value | Platform |
|----------|-------|----------|
| `NVBUF_MEM_CUDA_UNIFIED` | unified CPU/GPU memory | x86 dGPU |
| `NVBUF_MEM_DEFAULT` | platform default (NVMM) | Jetson iGPU |

On x86, `NVBUF_MEM_CUDA_UNIFIED` allows both CPU and GPU access to the same
buffer — necessary for OSD drawing (which uses CPU-side nvds metadata). On
Jetson, `NVBUF_MEM_DEFAULT` allocates in the Tegra NVMM pool, which is the
correct format for the iGPU encoder pipeline.

### To port to Jetson

Change the single instantiation in `pipeline.py`:

```python
# pipeline.py — __init__
# Before (x86 default):
self._element_factory = DeepStreamElementFactory()

# After (Jetson):
self._element_factory = DeepStreamElementFactory(platform="jetson")
```

That is the only change required for memory type. No other file needs to be
touched.

---

## 7. Where the Factory Is Used

```
DynamicRTSPPipeline.__init__
    self._element_factory = DeepStreamElementFactory()

_attach_preprocessing (source → streammux path)
    conv   ← element_factory.nvvideoconvert()   NV12 color convert
    caps   ← element_factory.capsfilter()        NV12 + model dimensions

_build_output_branch (demux → IPC socket path)
    q_demux    ← element_factory.queue()
    conv1      ← element_factory.nvvideoconvert()   NV12 → RGBA for OSD
    caps_rgba  ← element_factory.capsfilter()
    osd        ← _create_element()              (no factory — no shared props)
    conv2      ← _create_element()              (IPC boundary — see §8)
    caps_nv12  ← element_factory.capsfilter()
    q_fd       ← element_factory.queue()
    fdsink     ← _create_element()              (no factory — no shared props)
```

Elements that stay with `_create_element` (`nvdsosd`, `nvunixfdsink`) have no
shared default properties — there is nothing for the factory to centralise.

---

## 8. The `conv2` Exception — IPC Boundary Rule

`conv2` converts RGBA → NV12 immediately before `nvunixfdsink`. It is
intentionally created with `_create_element` rather than
`element_factory.nvvideoconvert`:

```python
"conv2": self._create_element("nvvideoconvert", f"conv2_{stream_id}"),
```

### Why

Setting `nvbuf-memory-type=NVBUF_MEM_CUDA_UNIFIED` on `conv2` changes the
output buffer's memory type. `nvunixfdsink` passes a CUDA IPC handle for this
buffer to the client process's `nvunixfdsrc`. If the memory type does not
match what `nvunixfdsrc` expects, the IPC handle import fails:

```
NvBufSurfaceCudaMemImport: cudaIpcOpenMemHandle err: 1
streaming stopped, reason error (-5)
```

`conv2` must output in the memory type that `nvunixfdsink`/`nvunixfdsrc`
negotiate by default — which is not `NVBUF_MEM_CUDA_UNIFIED`. Leaving
`nvbuf-memory-type` unset on `conv2` lets GStreamer use the sink's preferred
type, keeping IPC working correctly.

### Rule

> Any `nvvideoconvert` whose output crosses the IPC boundary
> (`nvunixfdsink`) must **not** have `nvbuf-memory-type` set explicitly.
> Use `_create_element` for it, not the factory.

---

## 9. Adding a New Helper Method

If a new element type needs consistent defaults across the pipeline, add a
method to `DeepStreamElementFactory`:

```python
def nvjpegenc(self, name: str, quality: int = 85) -> Gst.Element:
    """nvjpegenc with consistent quality default."""
    elem = self.make("nvjpegenc", name)
    elem.set_property("quality", quality)
    return elem
```

Then use it anywhere in `pipeline.py`:

```python
enc = self._element_factory.nvjpegenc("jpeg_cam1")
```

The factory is the single place to change when defaults need to evolve.
