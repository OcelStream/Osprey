# Tracker Implementation — Gst-nvtracker

This document explains how the NvSORT multi-object tracker was added to the DeepStream pipeline.

---

## What the tracker does

The `nvtracker` GStreamer plugin sits after the PGIE (detector) in the pipeline. It takes the bounding boxes produced by YOLO and assigns each detected object a **persistent tracking ID** across frames. Without the tracker, each frame's detections are independent — object 5 in frame 10 has no relationship to object 5 in frame 11. With the tracker, the same physical object keeps the same ID as long as it remains in the scene.

---

## Files changed

| File | What changed |
|------|-------------|
| `server/deepstream/config/config_tracker_NvSORT.yml` | New — low-level tracker YAML config |
| `server/deepstream/app/element_factory.py` | Added `nvtracker()` factory method |
| `server/backend/app/core/settings.py` | Added 4 tracker settings |
| `server/deepstream/app/deepstream.py` | Inserted tracker into the pipeline |

---

## Pipeline before and after

**Before:**
```
Source → nvvideoconvert → capsfilter → nvstreammux → nvinfer(YOLO) → nvstreamdemux → output branches
```

**After:**
```
Source → nvvideoconvert → capsfilter → nvstreammux → nvinfer(YOLO) → nvtracker(NvSORT) → nvstreamdemux → output branches
```

The tracker is a single shared element — all streams pass through it together in a batched frame, which is how DeepStream achieves efficient multi-stream tracking on one GPU.

---

## 1. Tracker YAML config — `config_tracker_NvSORT.yml`

This file is the low-level configuration consumed by the `NvMultiObjectTracker` library. The algorithm chosen is **NvSORT** — the NVIDIA-enhanced SORT tracker. It was chosen because:

- Your YOLO detector runs with `interval=0` (every frame gets inference), so visual tracking between frames is not needed.
- NvSORT is lightweight — it uses only CPU, no GPU compute for tracking itself.
- It adds a Kalman filter and cascaded data association on top of bare IOU matching, which significantly reduces ID switches.

### Key parameters explained

```yaml
BaseConfig:
  minDetectorConfidence: 0.2    # Detections below this are ignored by the tracker.
                                 # Matches your PGIE pre-cluster-threshold of 0.45
                                 # (set lower to be safe).

TargetManagement:
  probationAge: 3               # A new detection must survive 3 frames before
                                 # being reported downstream. Prevents false-positive
                                 # ghost tracks from single-frame noise.
  maxShadowTrackingAge: 30      # If the detector misses an object, the tracker keeps
                                 # it alive for up to 30 frames before terminating it.
                                 # Handles brief occlusions.
  earlyTerminationAge: 1        # A tentative target (in probationAge period) is
                                 # dropped after 1 missed frame — keeps ghost tracks short.

DataAssociator:
  associationMatcherType: 1     # Cascaded matching (more accurate than greedy).
                                 # Splits detections into confirmed/tentative and
                                 # matches in multiple stages.
  checkClassMatch: 0            # Allow cross-class association. YOLO can misclassify
                                 # the same object on different frames — this prevents
                                 # ID switches when the class label flips.
  matchingScoreWeight4Iou: 0.5
  matchingScoreWeight4SizeSimilarity: 0.5  # Combined IOU + size similarity score.

StateEstimator:
  stateEstimatorType: 1         # Simple-bbox Kalman Filter.
                                 # Tracks {x, y, w, h, dx, dy} — position, size,
                                 # and velocity. Predicts where an object will be
                                 # in the next frame, improving association accuracy.
```

### Target lifecycle

```
Detected → [Tentative: probationAge=3 frames]
                ↓ confirmed (matched 3 frames)       ↓ missed (earlyTerminationAge=1)
           [Active: reports tracking ID]              [Terminated]
                ↓ detector misses object
           [Shadow tracking: up to maxShadowTrackingAge=30 frames]
                ↓ exceeded / matched again
           [Terminated / reactivated]
```

---

## 2. Element factory — `element_factory.py`

A new `nvtracker()` method was added to `DeepStreamElementFactory`:

```python
def nvtracker(
    self,
    name: str,
    ll_lib_file: str,     # path to libnvds_nvmultiobjecttracker.so
    ll_config_file: str,  # path to the YAML config above
    tracker_width: int = 640,
    tracker_height: int = 384,
) -> Gst.Element:
```

The tracker internally scales video frames to `tracker_width × tracker_height` before processing. This is separate from the model input resolution — it controls how much GPU memory the tracker uses for frame crops. `640×384` is the recommended balance for performance.

`display-tracking-id=1` is set so that the OSD downstream can render the tracking ID on each bounding box.

---

## 3. Settings — `settings.py`

Four new fields were added to `PipelineSettings`, all overridable via environment variables:

| Setting | Env var | Default |
|---------|---------|---------|
| `tracker_config` | `DS_TRACKER_CONFIG` | `/deepstream_app/deepstream/config/config_tracker_NvSORT.yml` |
| `tracker_ll_lib` | `DS_TRACKER_LL_LIB` | `/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so` |
| `tracker_width` | `DS_TRACKER_WIDTH` | `640` |
| `tracker_height` | `DS_TRACKER_HEIGHT` | `384` |

To disable the tracker at runtime, set `DS_TRACKER_CONFIG=` (empty string). The pipeline then falls back to direct GIE → demux linking.

To switch to a more accurate tracker (NvDCF), set:
```
DS_TRACKER_CONFIG=/deepstream_app/deepstream/config/config_tracker_NvDCF_perf.yml
```

---

## 4. Pipeline wiring — `deepstream.py`

### `_build_tracker()` method

```python
def _build_tracker(self) -> Optional[Gst.Element]:
    if not self._config.tracker_config:
        return None                          # tracker disabled
    tracker = self._element_factory.nvtracker(
        "tracker",
        ll_lib_file=self._config.tracker_ll_lib,
        ll_config_file=self._config.tracker_config,
        tracker_width=self._config.tracker_width,
        tracker_height=self._config.tracker_height,
    )
    self._pipeline.add(tracker)
    return tracker
```

### Linking in `__init__`

```python
self._gies = self._build_inference_chain()  # creates nvinfer elements
self._tracker = self._build_tracker()        # creates nvtracker

tail = self._gies[-1] if self._gies else self._streammux

if self._tracker:
    tail.link(self._tracker)       # PGIE → tracker
    self._tracker.link(self._demux)  # tracker → demux
else:
    tail.link(self._demux)         # fallback: PGIE → demux
```

The tracker is a single shared element — all streams in the batch pass through it together. The tracker internally separates them by stream ID.

---

## How tracking ID reaches your application

After the tracker runs, each `NvDsObjectMeta` in the batch metadata has its `object_id` field populated with a persistent 64-bit tracking ID. You can read it in any downstream pad probe:

```python
def my_probe(pad, info, user_data):
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            tracking_id = obj_meta.object_id   # <-- persistent ID from nvtracker
            class_id    = obj_meta.class_id
            confidence  = obj_meta.confidence
            l_obj = l_obj.next
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK
```

---

## Switching tracker algorithms

| Use case | Config file | Notes |
|----------|------------|-------|
| Best performance (current) | `config_tracker_NvSORT.yml` | No GPU, Kalman filter + cascaded matching |
| Occlusion robustness | `config_tracker_NvDCF_perf.yml` | Needs GPU, visual correlation filter |
| Re-identification | `config_tracker_NvDeepSORT.yml` | Needs Re-ID model download from NGC |

Change via env var — no code change needed:
```bash
DS_TRACKER_CONFIG=/deepstream_app/deepstream/config/config_tracker_NvDCF_perf.yml
```
