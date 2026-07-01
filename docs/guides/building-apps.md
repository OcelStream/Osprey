# Building Applications on `DeepStreamClient`

**Scope:** This document shows how to build a real application on top of
`DeepStreamClient` — the base class in `client/base_client.py`. It is written
so that a developer who has never touched GStreamer or pyds can build a
working video analytics application by only writing plain Python.

> **Prerequisite documents**
>
> | Document | Purpose |
> |----------|---------|
> | `arch.md` | System topology (server ↔ client via Unix sockets) |
> | `../concepts/ipc-unix-sockets.md` | How frames and metadata cross the socket boundary |
> | **This document** | How to build applications on the base class |

---

## Table of Contents

1. [What the Base Class Gives You](#1--what-the-base-class-gives-you)
2. [The Data Types](#2--the-data-types)
3. [Minimal Subclass](#3--minimal-subclass)
4. [Custom Configuration](#4--custom-configuration)
5. [Custom Drawing](#5--custom-drawing)
6. [Stream Lifecycle Hooks](#6--stream-lifecycle-hooks)
7. [Available Utilities](#7--available-utilities)
8. [Complete Example: Parking Monitor](#8--complete-example-parking-monitor)
9. [Hook Reference](#9--hook-reference)

---

## 1 — What the Base Class Gives You

`DeepStreamClient` handles **all infrastructure**:

| Concern | Handled by base class |
|---------|----------------------|
| Watching `/run/nvunixfd` for socket files | `_watcher_loop`, `_scan_directory` |
| Checking if sockets are active vs stale | `_is_socket_active`, `_should_check_socket` |
| Building GStreamer receive pipelines | `create_pipeline`, `_create_pipeline_elements` |
| Linking GStreamer elements | `_link_pipeline`, `_SOURCE_CHAIN`, `_OUTPUT_CHAIN` |
| Setting up RTSP output | `_setup_rtsp_mount`, `_teardown_rtsp_mount` |
| Iterating pyds batch/frame/object metadata | `_osd_probe`, `_extract_objects` |
| Thread safety | `_lock`, `_stale_lock` |
| Bus message handling (EOS, errors) | `_on_bus_message` |

**Your subclass only writes application logic.** You receive clean Python
data — no GStreamer, no pyds, no linked-list iteration.

---

## 2 — The Data Types

### `FrameData` — one video frame

```python
@dataclass
class FrameData:
    frame_num: int              # monotonically increasing frame counter
    source_id: int              # which camera/stream produced this frame
    batch_id: int               # position in the batch
    pad_index: int              # streammux pad index
    surface: numpy.ndarray      # writable RGBA frame — draw with OpenCV
    objects: List[ObjectData]   # all detected objects in this frame
    raw_meta: object            # escape hatch: original pyds frame_meta
```

### `ObjectData` — one detected object

```python
@dataclass
class ObjectData:
    class_id: int           # model class index (e.g. 0=car, 1=truck)
    confidence: float       # detection confidence 0.0–1.0
    left: int               # bounding box — pixel coordinates
    top: int
    width: int
    height: int
    object_id: int          # tracker ID (0 if no tracker)
    label: str              # class name string (e.g. "car")
    raw_meta: object        # escape hatch: original pyds obj_meta
```

### `StreamRecord` — one active stream

```python
@dataclass
class StreamRecord:
    socket_path: str        # /run/nvunixfd/<uuid>.sock
    uuid: str               # extracted from socket filename
    index: int              # numeric pipeline index
    pipeline: object        # GStreamer pipeline (rarely needed)
```

---

## 3 — Minimal Subclass

The simplest possible application — log object counts per frame:

```python
from base_client import DeepStreamClient, FrameData

class ObjectCounter(DeepStreamClient):
    def _process_frame(self, frame_data: FrameData) -> None:
        count = len(frame_data.objects)
        if count > 0:
            print(f"Frame {frame_data.frame_num}: {count} objects detected")

if __name__ == "__main__":
    client = ObjectCounter()
    client.start()
```

That's it. No GStreamer. No pyds. No linked-list iteration. The base class
discovers sockets, builds pipelines, iterates buffers, extracts metadata,
and hands you a `FrameData` with everything ready.

### What happens if you DON'T override `_process_frame`?

The base implementation draws bounding boxes and an info-text HUD — exactly
what the original client did before the refactoring:

```python
def _process_frame(self, frame_data):
    if self._config.show_info_text:
        self._draw_info_text(frame_data.surface, frame_data)
    for obj in frame_data.objects:
        self._draw_object(frame_data.surface, obj)
```

---

## 4 — Custom Configuration

### Step 1: Subclass `ClientConfig`

```python
from dataclasses import dataclass
from base_client import ClientConfig

@dataclass
class ParkingConfig(ClientConfig):
    parking_zones_file: str = "/config/zones.json"
    alert_webhook: str = ""
    occupancy_threshold: float = 0.8

    @classmethod
    def from_env(cls, rtsp_port="8554", watch_dir="/run/nvunixfd"):
        cfg = super().from_env(rtsp_port=rtsp_port, watch_dir=watch_dir)
        cfg.parking_zones_file = os.environ.get(
            "PARKING_ZONES", "/config/zones.json"
        )
        cfg.alert_webhook = os.environ.get("ALERT_WEBHOOK", "")
        cfg.occupancy_threshold = float(
            os.environ.get("OCCUPANCY_THRESHOLD", "0.8")
        )
        return cfg
```

### Step 2: Point your subclass at it

```python
class ParkingClient(DeepStreamClient):
    _config_class = ParkingConfig
```

Now `self._config` is a `ParkingConfig` with all base fields plus your
custom ones. The base class calls `self._config_class.from_env()` automatically.

### Alternative: pass config directly

```python
cfg = ParkingConfig(
    rtsp_port="8554",
    parking_zones_file="/test/zones.json",
    occupancy_threshold=0.5,
)
client = ParkingClient(config=cfg)
```

---

## 5 — Custom Drawing

### Override `_draw_object` — change how each object looks

```python
class ColorByClass(DeepStreamClient):
    COLORS = {
        0: (0, 255, 0),    # car → green
        1: (0, 0, 255),    # truck → red
        2: (255, 165, 0),  # bus → orange
    }

    def _draw_object(self, surface, obj):
        color = self.COLORS.get(obj.class_id, (255, 255, 255))
        cv2.rectangle(
            surface,
            (obj.left, obj.top),
            (obj.left + obj.width, obj.top + obj.height),
            color, 2,
        )
        cv2.putText(
            surface,
            f"{obj.label} {obj.confidence:.0%}",
            (obj.left, obj.top - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )
```

### Override `_draw_info_text` — change the HUD

```python
class CustomHUD(DeepStreamClient):
    def _draw_info_text(self, surface, frame_data):
        count = len(frame_data.objects)
        cv2.putText(
            surface,
            f"Objects: {count} | Stream: {frame_data.source_id}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
```

### Override `_process_frame` — full control

When you override `_process_frame`, you control everything that happens per
frame. The base `_draw_object` and `_draw_info_text` are NOT called unless
you call them yourself:

```python
class FullControl(DeepStreamClient):
    def _process_frame(self, frame_data):
        # Your logic runs — base drawing is skipped entirely
        cars = [o for o in frame_data.objects if o.label == "car"]
        for car in cars:
            self._draw_bounding_box(
                frame_data.surface,
                car.left, car.top, car.width, car.height,
            )
        # Call base HUD if you still want it
        self._draw_info_text(frame_data.surface, frame_data)
```

---

## 6 — Stream Lifecycle Hooks

### `_on_stream_added(record)` — a new camera connected

Called after the pipeline is PLAYING and the `StreamRecord` is stored.

```python
class MyApp(DeepStreamClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cameras = {}

    def _on_stream_added(self, record):
        self.cameras[record.uuid] = {"status": "active", "frame_count": 0}
        logger.info("Camera %s registered", record.uuid)

    def _on_stream_removed(self, record):
        self.cameras.pop(record.uuid, None)
        logger.info("Camera %s unregistered", record.uuid)
```

### When are the hooks called?

```
_add_stream()
    ├── acquire lock, increment counter
    ├── create_pipeline(...)
    ├── pipeline.set_state(PLAYING)
    ├── store StreamRecord in self._streams
    ├── release lock
    └── _on_stream_added(record)    ◄── HERE (after lock released)

_remove_stream()
    ├── acquire lock, pop StreamRecord
    ├── release lock
    ├── pipeline.set_state(NULL)
    ├── teardown RTSP mount
    ├── remove socket file
    └── _on_stream_removed(record)  ◄── HERE (after cleanup)
```

The hooks run *outside* the lock, so you can safely do slow work (HTTP
requests, database writes, etc.) without blocking other streams.

---

## 7 — Available Utilities

The base class provides drawing utilities you can call from your hooks:

| Method | What it does |
|--------|-------------|
| `self._draw_bounding_box(surface, left, top, w, h)` | Green rectangle |
| `self._draw_mask(surface, obj)` | Segmentation mask overlay (requires `obj.raw_meta`) |
| `self._draw_info_text(surface, frame_data)` | Stream/frame HUD with text border |

These are instance methods, not static — you can override them in a subclass
to change colors, fonts, or behavior.

---

## 8 — Complete Example: Parking Monitor

```python
#!/usr/bin/env python3
"""
Parking lot monitor — counts occupied spots per zone,
draws zone overlays, sends alerts when occupancy is high.
"""

import json
import logging
import os
from dataclasses import dataclass

import cv2
import requests

from base_client import ClientConfig, DeepStreamClient, FrameData, StreamRecord

logger = logging.getLogger(__name__)


@dataclass
class ParkingConfig(ClientConfig):
    zones_file: str = "/config/zones.json"
    alert_url: str = ""
    threshold: float = 0.8

    @classmethod
    def from_env(cls, rtsp_port="8554", watch_dir="/run/nvunixfd"):
        cfg = super().from_env(rtsp_port=rtsp_port, watch_dir=watch_dir)
        cfg.zones_file = os.environ.get("ZONES_FILE", "/config/zones.json")
        cfg.alert_url = os.environ.get("ALERT_URL", "")
        cfg.threshold = float(os.environ.get("OCCUPANCY_THRESHOLD", "0.8"))
        return cfg


class ParkingMonitor(DeepStreamClient):
    _config_class = ParkingConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zones = self._load_zones()
        self.occupancy = {}

    def _load_zones(self):
        path = self._config.zones_file
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
        return []

    def _on_stream_added(self, record: StreamRecord):
        self.occupancy[record.uuid] = 0
        logger.info("Monitoring parking for camera %s", record.uuid)

    def _on_stream_removed(self, record: StreamRecord):
        self.occupancy.pop(record.uuid, None)

    def _process_frame(self, frame_data: FrameData):
        vehicles = [
            o for o in frame_data.objects
            if o.label in ("car", "truck", "bus")
        ]

        occupied = 0
        for zone in self.zones:
            zone_occupied = any(
                self._is_in_zone(v, zone) for v in vehicles
            )
            color = (0, 0, 255) if zone_occupied else (0, 255, 0)
            pts = zone["polygon"]
            cv2.polylines(frame_data.surface, [pts], True, color, 2)
            if zone_occupied:
                occupied += 1

        total = len(self.zones) or 1
        ratio = occupied / total
        self.occupancy[str(frame_data.source_id)] = ratio

        if ratio >= self._config.threshold and self._config.alert_url:
            self._send_alert(frame_data.source_id, ratio)

        for v in vehicles:
            self._draw_bounding_box(
                frame_data.surface, v.left, v.top, v.width, v.height
            )

        cv2.putText(
            frame_data.surface,
            f"Occupied: {occupied}/{len(self.zones)} ({ratio:.0%})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    @staticmethod
    def _is_in_zone(obj, zone):
        cx = obj.left + obj.width // 2
        cy = obj.top + obj.height // 2
        return cv2.pointPolygonTest(zone["polygon"], (cx, cy), False) >= 0

    def _send_alert(self, source_id, ratio):
        try:
            requests.post(self._config.alert_url, json={
                "source_id": source_id,
                "occupancy": ratio,
            }, timeout=2)
        except Exception as exc:
            logger.warning("Alert failed: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ParkingMonitor().start()
```

### What this example demonstrates

| Pattern | Where |
|---------|-------|
| Custom config | `ParkingConfig` extends `ClientConfig` |
| `_config_class` | `ParkingMonitor._config_class = ParkingConfig` |
| Lifecycle hooks | `_on_stream_added` / `_on_stream_removed` manage occupancy dict |
| `_process_frame` | Receives clean `FrameData`, iterates `objects` as a Python list |
| Custom drawing | Draws zone polygons and bounding boxes with OpenCV |
| Custom HUD | `cv2.putText` with occupancy percentage |
| Base utilities | `self._draw_bounding_box` reused from base class |
| No pyds anywhere | The entire file is pure Python — no GStreamer imports |

---

## 9 — Hook Reference

### Hooks (override in subclass)

| Hook | Called when | Receives | Base behavior |
|------|-----------|----------|---------------|
| `_process_frame(frame_data)` | Every frame | `FrameData` | Draws info text + bounding boxes |
| `_draw_object(surface, obj)` | Per object (from base `_process_frame`) | surface, `ObjectData` | Green bounding box |
| `_draw_info_text(surface, frame_data)` | Per frame (from base `_process_frame`) | surface, `FrameData` | "Stream: N \| Frame: M" |
| `_on_stream_added(record)` | After pipeline starts playing | `StreamRecord` | No-op |
| `_on_stream_removed(record)` | After pipeline torn down | `StreamRecord` | No-op |

### Public API (use, don't override)

| Method | Purpose |
|--------|---------|
| `start(wait_for_sockets=True)` | Block and run the client |
| `stop()` | Stop all pipelines |
| `add_stream(socket_path)` | Manually add a stream |
| `get_active_streams()` | Get dict of active streams |

### Utilities (call from hooks)

| Method | Purpose |
|--------|---------|
| `self._draw_bounding_box(surface, l, t, w, h)` | Green rectangle |
| `self._draw_mask(surface, obj)` | Segmentation mask overlay |
| `self._config` | Access configuration |
| `self._streams` | Access active stream records (under `self._lock`) |
