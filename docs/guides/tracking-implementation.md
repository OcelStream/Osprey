# Multi-Object Tracking — Full Implementation Guide

A to Z explanation of how tracking was designed, built, and wired into this platform —
from the concept of what tracking is, through every file that was changed or created.

---

## Table of Contents

1. [What is Multi-Object Tracking?](#1-what-is-multi-object-tracking)
2. [The Problem Without a Tracker](#2-the-problem-without-a-tracker)
3. [How DeepStream's Tracker Works](#3-how-deepstreams-tracker-works)
4. [The Four Algorithms — Concepts](#4-the-four-algorithms--concepts)
5. [The Full Data Flow — End to End](#5-the-full-data-flow--end-to-end)
6. [What Was Already in Place](#6-what-was-already-in-place)
7. [Implementation — Step by Step](#7-implementation--step-by-step)
   - [Step 1 — YAML Config Files](#step-1--yaml-config-files-serverdeeepstreamconfig)
   - [Step 2 — Element Factory](#step-2--element-factory-serverdeeepstreamapplement_factorypy)
   - [Step 3 — Settings](#step-3--settings-serverbackendappcoresettingspy)
   - [Step 4 — Pipeline Wiring](#step-4--pipeline-wiring-serverdeeepstreamappdeeepstreamy)
   - [Step 5 — IPC serialization (no change needed)](#step-5--ipc-serialization-no-change-needed)
   - [Step 7 — Client Drawing](#step-7--client-drawing-clientbase_clientpy)
8. [The Re-ID Model Setup (NvDeepSORT)](#8-the-re-id-model-setup-nvdeepsort)
9. [Choosing an Algorithm — Decision Guide](#9-choosing-an-algorithm--decision-guide)
10. [Config File Reference](#10-config-file-reference)
11. [Bug: Init Order Crash](#11-bug-init-order-crash)

---

## 1. What is Multi-Object Tracking?

Object detection (what YOLO does) answers a single question per frame:
> "What objects are in this frame, and where are they?"

It does this independently for every frame. Each frame is processed from scratch.
The output is a list of bounding boxes with a class label and confidence score.

**Multi-Object Tracking (MOT)** answers a different question across frames:
> "Is the person in box A on frame 100 the same person as the one in box B on frame 101?"

It connects detections over time and assigns each real-world object a **persistent unique ID**.
That ID stays the same as long as the object remains in the scene — even if the detector
briefly misses it, or if it becomes partially occluded.

Without tracking, you have a list of boxes. With tracking, you have a list of *objects*
with identities that persist through time.

---

## 2. The Problem Without a Tracker

Before tracking was added, this pipeline produced output like:

```
Frame 10:  [box@(100,200), class=car, conf=0.91]
Frame 11:  [box@(103,202), class=car, conf=0.88]
Frame 12:  [box@(108,205), class=truck, conf=0.72]  ← misclassified!
Frame 13:  [box@(112,208), class=car, conf=0.90]
```

Problems:
- No way to know frames 10, 11, 12, 13 are all the same person.
- Frame 12's misclassification (YOLO is not perfect) creates a ghost object.
- Counting is impossible — you can't count unique people.
- Any trajectory or time-based analytics is impossible.

With tracking the same frames look like:

```
Frame 10:  [ID:7, box@(100,200), class=car, conf=0.91]
Frame 11:  [ID:7, box@(103,202), class=car, conf=0.88]
Frame 12:  [ID:7, box@(108,205), class=truck, conf=0.72]  ← same ID, different label
Frame 13:  [ID:7, box@(112,208), class=car, conf=0.90]
```

Object identity is now stable. Analytics become possible.

---

## 3. How DeepStream's Tracker Works

DeepStream's `Gst-nvtracker` plugin is a GStreamer element that sits between the
detector (`nvinfer`) and the demuxer (`nvstreamdemux`) in the pipeline.

It receives two things from the upstream element per frame:
1. **Video frame buffer** — raw pixel data (NV12 or RGBA), used by visual trackers.
2. **`NvDsBatchMeta`** — the list of `NvDsObjectMeta` structs produced by YOLO.

The plugin delegates all actual tracking math to a **low-level tracker library**
loaded at runtime via `ll-lib-file`. In this project that library is always:
```
/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
```

This single `.so` implements all four tracking algorithms. The YAML config file
specified in `ll-config-file` tells it which algorithm to activate and how to tune it.

After the tracker runs, it writes a `uint64_t object_id` into each `NvDsObjectMeta`.
That ID is what flows downstream — through the serializer, over the Unix socket,
through the deserializer, and finally onto the RTSP frame drawn by OpenCV.

### The tracker's internal API

The plugin negotiates with the library through three main calls:

| Call | When | Purpose |
|------|------|---------|
| `NvMOT_Query` | Startup | Asks library what input format it needs (NV12/RGBA, CPU/GPU memory) |
| `NvMOT_Init` | Per context | Creates a tracker instance, pre-allocates GPU memory |
| `NvMOT_Process` | Per frame batch | Feeds detections + video frames, gets back object IDs |
| `NvMOT_DeInit` | Shutdown | Releases resources |

---

## 4. The Four Algorithms — Concepts

All four live in `libnvds_nvmultiobjecttracker.so`. They share common modules
(target management, state estimation, data association) and differ in what
additional modules they enable.

### IOU Tracker

The simplest possible tracker. It answers:
> "Which box in frame N+1 overlaps the most with box X from frame N?"

- Computes **Intersection-Over-Union** between every pair of old and new boxes.
- Uses **greedy matching** — assigns the best-scoring pair first, then the next, etc.
- No GPU. No Kalman filter. No visual features.

**Limitation**: If two objects cross paths, or if the detector misses an object for
even one frame, the ID is lost. No memory of where the object was moving.

**Config**: `config_tracker_IOU.yml`

---

### NvSORT (Simple Online Realtime Tracking — NVIDIA enhanced)

Adds two things on top of IOU:

1. **Kalman Filter state estimator** — models each object as having a position,
   size, and *velocity*. When a frame arrives, the filter first *predicts* where
   each known object should be now, then *updates* that prediction with the
   new detection. This means the tracker can stay locked onto a fast-moving object
   even if the detector produces a slightly displaced box.

2. **Cascaded data association** — instead of one greedy matching pass, it runs
   multiple matching stages: first match high-confidence detections against validated
   targets, then lower-confidence detections against remaining targets. This reduces
   ID switches on crowded scenes.

- No GPU. Very low CPU.
- Best for YOLO running at `interval=0` (every frame), medium-density scenes.

**Config**: `config_tracker_NvSORT.yml`

---

### NvDCF (Discriminative Correlation Filter)

Adds **visual tracking** on top of NvSORT. A DCF tracker learns a correlation
filter that describes what a specific object *looks like* in terms of color and
gradient features. On each frame it searches a region around the object's predicted
location and produces a confidence map — a 2D map where peaks indicate where the
target is most likely to be.

This means NvDCF can track an object **even when the detector misses it entirely**
(false negative, occlusion) for many consecutive frames. It uses its learned
visual filter to keep finding the object and update the Kalman filter.

The correlation filter also gives a per-object confidence score (`tracker_confidence`)
— useful for detecting when the tracker itself is losing the object.

- **Needs GPU** — feature extraction (ColorNames, HOG) and FFT-based filter
  correlation run on CUDA.
- Configured with `visualTrackerType: 2` (NvDCF_VPI — NVIDIA VPI-accelerated).

**Config**: `config_tracker_NvDCF_perf.yml`

---

### NvDeepSORT

Replaces DCF visual features with a **Re-ID neural network**. Re-ID (Re-Identification)
is a deep learning task where a network learns to produce a compact feature vector
(embedding) for each person/vehicle such that:
- Two images of the same person → similar embeddings (small cosine distance).
- Two images of different people → dissimilar embeddings (large cosine distance).

NvDeepSORT uses these embeddings for data association. When it needs to decide
whether detection A belongs to tracked target B, it computes the cosine similarity
between A's embedding and the gallery of embeddings stored for B.

This makes it the most robust tracker for **re-identification after long occlusion** —
if a person disappears behind a pillar for 50 frames and reappears, NvDeepSORT can
match them back to their original ID because their appearance matches the stored gallery.

- **Needs GPU** — Re-ID network inference runs via TensorRT.
- Requires a pre-trained Re-ID model (`resnet50_market1501.etlt`).
- TensorRT builds an optimized engine on first run (can take 2–5 minutes).

**Config**: `config_tracker_NvDeepSORT.yml`

---

## 5. The Full Data Flow — End to End

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SERVER CONTAINER (deepstream_app-8.0)                                    │
│                                                                           │
│  nvurisrcbin (RTSP/file)                                                  │
│       │                                                                   │
│       ▼                                                                   │
│  nvvideoconvert ──► capsfilter (NV12, 1280×720)                          │
│       │                                                                   │
│       ▼                                                                   │
│  nvstreammux  ◄── (up to 64 streams merged into one batch)               │
│       │                                                                   │
│       ▼                                                                   │
│  nvinfer (YOLO detector)                                                  │
│       │  Writes NvDsObjectMeta per detected box:                         │
│       │    class_id, confidence, rect_params, obj_label                  │
│       │    object_id = INVALID (not set yet)                             │
│       │                                                                   │
│       ▼                                                                   │
│  nvtracker  ◄────────────────── libnvds_nvmultiobjecttracker.so         │
│       │                         config_tracker_<ALGO>.yml                │
│       │  Reads: rect_params from each NvDsObjectMeta                    │
│       │  Reads: video frame pixels (NvDCF, NvDeepSORT)                  │
│       │  Runs:  data association + state estimation                      │
│       │  Writes: object_id (uint64) into each NvDsObjectMeta            │
│       │                                                                   │
│       ▼                                                                   │
│  nvstreamdemux  (splits batch back to per-stream buffers)                │
│       │                                                                   │
│  ┌────┴────────────────────────────────────────┐                        │
│  │  Per-stream output branch:                   │                        │
│  │  queue → nvvideoconvert → nvdsosd            │                        │
│  │       → nvvideoconvert → nvunixfdsink        │                        │
│  │                              │               │                        │
│  │            serialize_meta.c ─┘               │                        │
│  │            Reads NvDsObjectMeta:             │                        │
│  │              class_id, confidence            │                        │
│  │              rect_params (bbox)              │                        │
│  │              object_id  ← tracking ID!       │                        │
│  │              obj_label                       │                        │
│  │            Packs into binary blob            │                        │
│  └──────────────────────────────────────────────┘                        │
│                         │                                                 │
│              Unix domain socket /run/nvunixfd/<stream_id>.sock            │
└─────────────────────────┼────────────────────────────────────────────────┘
                          │  (shared volume: ./sockets)
┌─────────────────────────┼────────────────────────────────────────────────┐
│  CLIENT CONTAINER (ds_client)                                             │
│                         │                                                 │
│              nvunixfdsrc (reads socket)                                   │
│                         │                                                 │
│              deserialize_meta.c                                           │
│              Reads binary blob → reconstructs NvDsObjectMeta:            │
│                class_id, confidence, rect_params                         │
│                object_id  ← tracking ID preserved!                       │
│                obj_label                                                  │
│                         │                                                 │
│  identity → caps_nv12 → nvstreammux → nvvideoconvert → caps_rgba         │
│                                                          │                │
│                                               Pad probe (_osd_probe)     │
│                                               Calls _extract_objects()   │
│                                               Builds ObjectData:         │
│                                                 .object_id (track ID)   │
│                                                 .label  (obj_label)      │
│                                                 .confidence              │
│                                                 .left/top/width/height   │
│                                                          │                │
│                                               _process_frame()           │
│                                               _draw_object()  ─────────► OpenCV draws on RGBA surface:
│                                                                           "ID:42 car 0.91"
│  nvdsosd → nvvideoconvert → nvv4l2h264enc → h264parse → rtph264pay      │
│       → udpsink → RTSP server → rtsp://localhost:8554/ds-test<N>        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. What Was Already in Place

Before tracking was added, these components already handled `object_id` correctly.
No changes were needed to them.

### IPC serialization — server → client

The `object_id` and `obj_label` already cross the IPC boundary as part of Osprey's
metadata serialization, handled by the prebuilt `serialize_meta.so` /
`deserialize_meta.so` libraries. Tracking required **no change** to the wire
format — the tracking ID simply rides the field that previously carried
`INVALID_TRACKING_ID`, and the client reconstructs it into `NvDsObjectMeta`.

### `base_client.py` — `_extract_objects()`

```python
# Already extracted object_id from pyds metadata into ObjectData:
ObjectData(
    object_id=obj_meta.object_id,   # ← tracking ID
    label=obj_meta.obj_label,        # ← label name
    ...
)
```

The entire IPC chain was already wired for tracking. The tracking ID was passing
through as zero (`INVALID_TRACKING_ID`) because no tracker element existed.

---

## 7. Implementation — Step by Step

### Step 1 — YAML Config Files (`server/deepstream/config/`)

Four YAML files were created, one per algorithm. Each file configures the
`NvMultiObjectTracker` library — it reads these on startup via `NvMOT_Query`.

All configs are based on **NVIDIA's own reference configs** from
`/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/` inside the
DS 8.0 container, with one deliberate change: `checkClassMatch: 0` in all configs.

**Why `checkClassMatch: 0`?**
NVIDIA's defaults set this to `1`, which means the tracker only associates
detections with targets of the same class. This is wrong for YOLO because YOLO
can misclassify the same real-world object as a different class on different frames
(e.g., `car` on frame 10, `truck` on frame 12). With `checkClassMatch: 1`,
frame 12 would start a new track instead of continuing the existing one.

#### `config_tracker_IOU.yml`

```yaml
DataAssociator:
  associationMatcherType: 0   # greedy — O(n²) but faster than cascaded
  checkClassMatch: 0
  matchingScoreWeight4Iou: 0.6
  matchingScoreWeight4SizeSimilarity: 0.4

StateEstimator:
  # No StateEstimator section → dummy estimator (pure IOU, no Kalman)
```

Key difference from others: no `StateEstimator` section means `stateEstimatorType: 0`
(dummy) — position is not predicted between frames.

#### `config_tracker_NvSORT.yml`

```yaml
DataAssociator:
  associationMatcherType: 1   # cascaded — multi-stage matching
  usePrediction4Assoc: 1      # uses Kalman-predicted position for matching

StateEstimator:
  stateEstimatorType: 2       # regular-bbox KF: tracks {x,y,w,h,dx,dy,dw,dh}
  noiseWeightVar4Loc: 0.0301
  noiseWeightVar4Vel: 0.0017
  useAspectRatio: 1           # tracks aspect ratio, not raw width
```

The Kalman filter state for each target is `{x, y, aspect_ratio, height, vx, vy, va, vh}`.
When the detector misses a frame, the filter predicts where the object is now
based on its last known velocity. Association is done against predicted positions,
not the last observed position.

#### `config_tracker_NvDCF_perf.yml`

```yaml
VisualTracker:
  visualTrackerType: 2        # NvDCF_VPI (NVIDIA VPI-accelerated)
  useColorNames: 1            # 10-channel color descriptor
  useHog: 0                   # HOG disabled (slower, not needed for perf profile)
  featureImgSizeLevel: 2      # feature map 18×18 pixels per channel
  featureFocusOffsetFactor_y: -0.2  # shift attention window up (top of person)
  filterLr: 0.075             # DCF filter learning rate

DataAssociator:
  minMatchingScore4VisualSimilarity: 0.5356
  matchingScoreWeight4VisualSimilarity: 0.3370
  matchingScoreWeight4Iou: 0.3656
  matchingScoreWeight4SizeSimilarity: 0.4354
```

The total association score per (detection, target) pair:
```
score = 0.3370 * visual_similarity
      + 0.3656 * iou
      + 0.4354 * size_similarity
```

#### `config_tracker_NvDeepSORT.yml`

```yaml
ReID:
  reidType: 1                 # NvDeepSORT mode
  inferDims: [3, 256, 128]    # input: 3 channels, 256×128 pixels (portrait crop)
  networkMode: 1              # FP16 inference via TensorRT
  inputOrder: 0               # NCHW tensor layout
  offsets: [123.675, 116.28, 103.53]  # ImageNet mean subtraction
  netScaleFactor: 0.01735207  # 1/57.63 (ImageNet std normalization)
  addFeatureNormalization: 1  # L2-normalize output embeddings
  tltEncodedModel: "/deepstream_app/deepstream/models/resnet50_market1501.etlt"
  modelEngineFile: "/deepstream_app/deepstream/models/resnet50_market1501.etlt_b100_gpu0_fp16.engine"
```

The preprocessing formula applied to each crop before inference:
```
pixel_out = (pixel_in - offset) * netScaleFactor
```

The model is ResNet-50 trained on Market-1501 + CUHK03 + DukeMTMC datasets —
large person Re-ID benchmarks. Despite being trained on people, it generalizes
reasonably to other object types.

---

### Step 2 — Element Factory (`server/deepstream/app/element_factory.py`)

A new method was added to `DeepStreamElementFactory`:

```python
def nvtracker(
    self,
    name: str,
    ll_lib_file: str,
    ll_config_file: str,
    tracker_width: int = 640,
    tracker_height: int = 384,
) -> Gst.Element:
    elem = self.make("nvtracker", name)
    elem.set_property("tracker-width", tracker_width)
    elem.set_property("tracker-height", tracker_height)
    elem.set_property("ll-lib-file", ll_lib_file)
    elem.set_property("ll-config-file", ll_config_file)
    elem.set_property("gpu-id", 0)
    elem.set_property("display-tracking-id", 1)
    return elem
```

**`tracker-width` / `tracker-height`**: The tracker plugin internally scales
video frames to this resolution before feeding them to the visual tracking algorithm
(NvDCF, NvDeepSORT). This is separate from the YOLO input resolution (1280×720).
`640×384` gives the tracker enough detail without excessive GPU memory usage.

**`display-tracking-id: 1`**: Tells the plugin to copy the tracking ID into the
text label used by `nvdsosd`. Since OSD display is disabled (`display-bbox: 0`,
`display-text: 0`) in the output branch, this has no visible effect — but it
ensures the ID is accessible in metadata downstream.

---

### Step 3 — Settings (`server/backend/app/core/settings.py`)

The original design had two separate env vars (`DS_TRACKER_CONFIG`, `DS_TRACKER_LL_LIB`).
These were replaced with a single `DS_TRACKER` env var that resolves to a preset:

```python
_TRACKER_PRESETS = {
    "IOU":        "/deepstream_app/deepstream/config/config_tracker_IOU.yml",
    "NvSORT":     "/deepstream_app/deepstream/config/config_tracker_NvSORT.yml",
    "NvDCF":      "/deepstream_app/deepstream/config/config_tracker_NvDCF_perf.yml",
    "NvDeepSORT": "/deepstream_app/deepstream/config/config_tracker_NvDeepSORT.yml",
}

@property
def tracker_config(self) -> str:
    name = self.tracker.strip()
    if not name or name.lower() == "off":
        return ""            # disables tracker in _build_tracker()
    if name in _TRACKER_PRESETS:
        return _TRACKER_PRESETS[name]
    return name              # treat raw value as a file path (custom config)

@property
def tracker_ll_lib(self) -> str:
    return "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
```

`tracker_ll_lib` is a property (not a field) because it never changes — all four
algorithms live in the same shared library. Only the YAML config changes.

**Usage in `.env`:**
```bash
DS_TRACKER=NvDeepSORT   # or IOU / NvSORT / NvDCF / off
```

---

### Step 4 — Pipeline Wiring (`server/deepstream/app/deepstream.py`)

#### Init order bug (found and fixed)

The first attempt crashed with:
```
AttributeError: 'DynamicRTSPPipeline' object has no attribute '_element_factory'
```

The root cause: `_build_tracker()` was called at line 106 but `_element_factory`
was instantiated at line 125. The fix was to move all helpers to the top of
`__init__`, before pipeline element construction:

```python
# WRONG (original order):
self._gies = self._build_inference_chain()
self._tracker = self._build_tracker()       # ← uses _element_factory
...
self._element_factory = DeepStreamElementFactory()  # ← too late

# CORRECT (fixed order):
self._element_factory = DeepStreamElementFactory()  # ← created first
...
self._gies = self._build_inference_chain()
self._tracker = self._build_tracker()               # ← now safe
```

#### `_build_tracker()` method

```python
def _build_tracker(self) -> Optional[Gst.Element]:
    if not self._config.tracker_config:
        logger.info("Tracker disabled (DS_TRACKER=off)")
        return None
    tracker = self._element_factory.nvtracker(
        "tracker",
        ll_lib_file=self._config.tracker_ll_lib,
        ll_config_file=self._config.tracker_config,
        tracker_width=self._config.tracker_width,
        tracker_height=self._config.tracker_height,
    )
    self._pipeline.add(tracker)
    logger.info("Tracker enabled: DS_TRACKER=%s → %s",
                self._config.tracker, self._config.tracker_config)
    return tracker
```

#### Pipeline linking

```python
tail = self._gies[-1] if self._gies else self._streammux

if self._tracker:
    tail.link(self._tracker)       # nvinfer → nvtracker
    self._tracker.link(self._demux)  # nvtracker → nvstreamdemux
else:
    tail.link(self._demux)         # fallback when DS_TRACKER=off
```

The tracker is a **single shared element** across all streams. The batched
frame from `nvstreammux` (containing frames from all streams) enters the tracker
as a whole batch. Internally, `NvMultiObjectTracker` tracks targets per-stream
using `streamID` to separate them — no state leaks between streams.

---

### Step 5 — IPC serialization (no change needed)

Osprey's metadata serialization — the prebuilt `serialize_meta.so` /
`deserialize_meta.so` libraries — already carries `object_id` across the
server → client boundary and reconstructs it on the client. Before the tracker
was added, that field always held `INVALID_TRACKING_ID` (a large sentinel);
after the tracker runs, it holds the real persistent 64-bit ID. The wire format
did not change — the receiver interprets whatever value is present, so no
serializer or deserializer changes were required.

---

### Step 7 — Client Drawing (`client/base_client.py`)

Three methods were updated or added:

#### `_draw_object()` — updated

```python
def _draw_object(self, surface, obj: ObjectData) -> None:
    color = self._class_color(obj.class_id)   # consistent color per class
    self._draw_bounding_box(
        surface, obj.left, obj.top, obj.width, obj.height, color=color
    )
    self._draw_label(surface, obj, color=color)
```

#### `_draw_bounding_box()` — updated

Added `color` parameter (backwards-compatible, defaults to green):
```python
def _draw_bounding_box(self, surface, left, top, width, height, color=(0,255,0)):
    cv2.rectangle(surface, (left, top), (left+width, top+height), color, 2)
```

#### `_draw_label()` — new

Draws a filled background rectangle + text above the bbox:
```python
def _draw_label(self, surface, obj: ObjectData, color=(0,255,0)) -> None:
    parts = []
    if obj.object_id > 0:
        parts.append(f"ID:{obj.object_id}")   # from nvtracker
    if obj.label:
        parts.append(obj.label)               # from obj_meta.obj_label
    if obj.confidence > 0:
        parts.append(f"{obj.confidence:.2f}") # from nvinfer
    label_text = " ".join(parts)
    # ... measure text size, draw filled rect in bbox color,
    # auto-pick black/white text for contrast, putText
```

Example output per object: `ID:42 car 0.91`

Why `obj.label` and not a labels file on the client? Because the label is already
serialized inside `obj_label` by `nvinfer` (using `labelfile-path` in the PGIE
config) and transported through the entire IPC chain. No duplication needed.

#### `_class_color()` — new

```python
@staticmethod
def _class_color(class_id: int) -> tuple:
    palette = (
        (0, 255, 0),    # green   — class 0
        (255, 128, 0),  # orange  — class 1
        (0, 128, 255),  # blue    — class 2
        (255, 0, 128),  # pink    — class 3
        (128, 255, 0),  # lime    — class 4
        (0, 255, 255),  # cyan    — class 5
        (255, 255, 0),  # yellow  — class 6
        (128, 0, 255),  # purple  — class 7
    )
    return palette[class_id % len(palette)]
```

This makes every class visually distinct on-screen without any configuration.

---

## 8. The Re-ID Model Setup (NvDeepSORT)

NvDeepSORT requires a ResNet-50 Re-ID model trained on person re-identification
datasets. NVIDIA provides this as a TAO-encoded `.etlt` file on NGC.

### What was done

1. Downloaded the model inside the running container:
   ```bash
   docker exec deepstream_app-8.0 bash -c "
     mkdir -p /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
     wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt' \
          -P /opt/nvidia/deepstream/deepstream/samples/models/Tracker/
   "
   ```

2. Copied it from the container's filesystem into the host project via the volume:
   ```bash
   docker exec deepstream_app-8.0 cp \
     /opt/nvidia/deepstream/deepstream/samples/models/Tracker/resnet50_market1501.etlt \
     /deepstream_app/deepstream/models/resnet50_market1501.etlt
   ```

3. Updated `config_tracker_NvDeepSORT.yml` to point to the volume-mounted path:
   ```yaml
   tltEncodedModel: "/deepstream_app/deepstream/models/resnet50_market1501.etlt"
   modelEngineFile: "/deepstream_app/deepstream/models/resnet50_market1501.etlt_b100_gpu0_fp16.engine"
   ```

### What happens on first run

TensorRT reads `tltEncodedModel`, decrypts it using `tltModelKey: "nvidia_tao"`,
and compiles an optimized GPU engine. The engine is saved at `modelEngineFile`.
On subsequent runs, TensorRT loads the engine directly (fast). The first build
takes 2–5 minutes depending on GPU.

### Why the model lives in `deepstream/models/`

The volume mount `./server:/deepstream_app` makes the model persistent across
container restarts. If the model stayed only in the container's filesystem
(`/opt/nvidia/...`), it would be lost every time the container is recreated.

---

## 9. Choosing an Algorithm — Decision Guide

```
Does your YOLO run every frame (interval=0)?
├─ Yes → Do you need to track through occlusions?
│        ├─ No  → NvSORT  (fast, accurate enough)
│        └─ Yes → Do you need re-identification after long disappearance?
│                 ├─ No  → NvDCF  (visual tracking, handles occlusion)
│                 └─ Yes → NvDeepSORT  (Re-ID network, strongest re-ID)
└─ No  → Always use NvDCF or NvDeepSORT (they can track between detections)
```

Set in `.env`:
```bash
DS_TRACKER=NvSORT        # default — no GPU, Kalman filter
DS_TRACKER=IOU           # absolute minimum — no GPU, no prediction
DS_TRACKER=NvDCF         # GPU, visual tracking, good for dense scenes
DS_TRACKER=NvDeepSORT    # GPU, Re-ID, best for long occlusions
DS_TRACKER=off           # no tracker — raw YOLO output, no IDs
```

---

## 10. Config File Reference

| File | Algorithm | GPU | CPU | Use case |
|------|-----------|-----|-----|----------|
| `config_tracker_IOU.yml` | IOU | None | Minimal | Sparse objects, baseline |
| `config_tracker_NvSORT.yml` | NvSORT | None | Low | YOLO every frame, general use |
| `config_tracker_NvDCF_perf.yml` | NvDCF | Medium | Low | Occlusion, PGIE interval > 0 |
| `config_tracker_NvDeepSORT.yml` | NvDeepSORT | High | Low | Re-ID after disappearance |

### Common parameters across all configs

| Parameter | What it does |
|-----------|-------------|
| `minDetectorConfidence` | Detections below this confidence are ignored by the tracker. Lower = more sensitive, more ghost tracks. |
| `probationAge` | New targets are not reported for this many frames. Filters out single-frame false positives. |
| `maxShadowTrackingAge` | How many frames to keep tracking an object the detector stopped seeing. Higher = more robust to occlusion. |
| `earlyTerminationAge` | If a tentative (unconfirmed) target is missed this many times, kill it immediately. |
| `maxTargetsPerStream` | Hard cap. When reached, new detections are ignored. Pre-allocates GPU memory. |
| `associationMatcherType` | `0` = greedy (fast), `1` = cascaded (accurate). |
| `checkClassMatch` | `0` = associate across classes (correct for YOLO). `1` = same-class only. |

---

## 11. Bug: Init Order Crash

**Error:**
```
File "deepstream.py", line 106, in __init__
    self._tracker = self._build_tracker()
AttributeError: 'DynamicRTSPPipeline' object has no attribute '_element_factory'
```

**Root cause:** `_build_tracker()` uses `self._element_factory.nvtracker(...)`,
but `_element_factory` was instantiated 19 lines later in `__init__`.

**Fix:** Moved all helper/factory instantiations to the top of `__init__`,
before any pipeline element construction. The corrected order:

```python
def __init__(self, ...):
    # 1. Helpers first — everything else depends on these
    self._element_factory = DeepStreamElementFactory()
    self._source_factory  = SourceBinFactory()
    self._spot_manager    = SpotManager(...)
    self._perf_data       = PERF_DATA()
    self._loop            = GLib.MainLoop()

    # 2. Pipeline elements — use the factories above
    self._pipeline  = Gst.Pipeline()
    self._streammux = self._create_element("nvstreammux", "stream-mux")
    self._gies      = self._build_inference_chain()
    self._tracker   = self._build_tracker()      # safe now
    self._demux     = self._create_element("nvstreamdemux", "stream-demux")

    # 3. Link elements
    ...
```
