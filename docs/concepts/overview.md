# What Is Osprey?

> The philosophy, the problem it solves, and why it is built the way it is.

---

## The Problem

Running AI video analytics on live camera streams is hard in three specific ways:

**1. GPU inference is batch-oriented, but cameras are independent.**  
A GPU processes frames most efficiently when they are batched — many frames
from many cameras processed in one shot. But each RTSP camera produces its
own independent stream at its own pace. Something has to bridge that mismatch.

**2. Inference infrastructure and application logic change at different rates.**  
The GStreamer pipeline, NVIDIA plugins, TensorRT engines, and IPC mechanisms
are infrastructure — they should be stable and rarely changed. The drawing,
alerting, business rules, and output format are application logic — they
change every sprint. Mixing them in one codebase means every application
change risks breaking the GPU pipeline.

**3. Adding and removing cameras while the system is running is the default.**  
A parking lot has fixed cameras, but a construction site gets new cameras
weekly. Security deployments scale up and down. The system must support
runtime source management without restarting the entire pipeline.

---

## What This System Does

`Osprey` is a **dynamic multi-stream DeepStream video analytics platform**.

It takes any number of RTSP camera streams, runs YOLO object detection on all
of them simultaneously on a single GPU, and delivers the annotated results as
new RTSP streams that any viewer can connect to.

```
RTSP cameras (any number)
        │
        ▼
  ┌─────────────────────────────────┐
  │  Server process                 │
  │                                 │
  │  REST API — add / remove        │
  │  streams at runtime             │
  │                                 │
  │  NVIDIA DeepStream pipeline     │
  │  batched GPU inference          │
  │  YOLO detection / segmentation  │
  └──────────────┬──────────────────┘
                 │ Unix sockets (zero-copy GPU buffers)
  ┌──────────────▼──────────────────┐
  │  Client process                 │
  │                                 │
  │  Your application code          │
  │  Drawing, alerting, analytics   │
  │                                 │
  │  RTSP output                    │
  └─────────────────────────────────┘
        │
        ▼
  RTSP viewers, dashboards, recorders
```

---

## The Core Philosophy

### Infrastructure should be invisible

The GPU pipeline, NVIDIA plugins, GStreamer element wiring, memory management,
and IPC are infrastructure. Users of this platform should never need to
understand them. They should write Python business logic and receive clean
data objects.

This is why `DeepStreamClient` exists as a base class with hook methods:
`_process_frame`, `_on_stream_added`, `_on_stream_removed`. A user who wants
to count vehicles never writes a GStreamer pad probe. They override one method
and receive a `FrameData` with a Python list of `ObjectData`.

### Separate what changes from what doesn't

| Stable (infrastructure) | Changing (application) |
|------------------------|------------------------|
| GStreamer pipeline topology | Drawing / OSD overlays |
| NVIDIA plugin configuration | Business rules (zones, thresholds) |
| Batch inference engine | Alert destinations |
| IPC mechanism | Output format |
| Thread safety model | Stream selection logic |

These two groups live in different processes, different files, and different
abstraction layers — intentionally.

### One GPU, many streams

A single `nvstreammux` batches frames from all active streams into one tensor.
One `nvinfer` call processes them all. `nvstreamdemux` splits results back per
stream. This is the only architecture that uses the GPU efficiently across many
sources — running one inference per stream would be 10x slower.

### Runtime mutability as a first-class concern

`DynamicRTSPPipeline` is the design centrepiece. A standard DeepStream pipeline
is static — you define sources at startup. This pipeline supports `add_source()`
and `remove_source()` at any time while the pipeline is `PLAYING`. Every design
decision (the lock, the SpotManager, the per-stream `StreamRecord`, the branch
teardown sequence) exists to make this safe.

---

## What This Is Not

- Not a general-purpose video server (it is purpose-built for DeepStream on NVIDIA GPUs)
- Not a cloud-native platform (it runs on a single host with a single GPU)
- Not a real-time notification system (it has no event bus yet — that is a planned enhancement)
- Not a model training pipeline (it runs inference on pre-built TensorRT engines)

---

## Where to Go Next

| Question | Document |
|----------|----------|
| How does the GStreamer pipeline work? | [deepstream-pipeline.md](deepstream-pipeline.md) |
| Why two processes? | [two-process-model.md](two-process-model.md) |
| What happens when I add a stream? | [stream-lifecycle.md](stream-lifecycle.md) |
| How do GPU buffers move between processes? | [ipc-unix-sockets.md](ipc-unix-sockets.md) |
| How do I build an application on this? | [guides/building-apps.md](../guides/building-apps.md) |
| What is the full architecture? | [architecture/arch.md](../architecture/arch.md) |
