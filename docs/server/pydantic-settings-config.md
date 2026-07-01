# Pipeline Configuration — Pydantic `BaseSettings`

> **Status:** Implemented  
> **Files changed:** `server/backend/app/core/settings.py` (new), `server/deepstream/app/deepstream.py`, `server/backend/app/core/context.py`, `server/backend/requirements.txt`  
> **Replaces:** `PipelineConfig` dataclass + `os.getenv` in `deepstream.py`  
> **Related:** [Architecture Proposal §10](../local/architecture-proposal.md#10-configuration-architecture) · [Refactoring Suggestions §8](../local/refactoring-suggestions.md)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [PipelineSettings — Field Reference](#3-pipelinesettings--field-reference)
4. [Environment Variable Names](#4-environment-variable-names)
5. [GIE Configs and Hidden Classes](#5-gie-configs-and-hidden-classes)
6. [How max_sources Was Removed](#6-how-max_sources-was-removed)
7. [How the Settings Flow Through the Code](#7-how-the-settings-flow-through-the-code)
8. [Hidden Class Names at Runtime](#8-hidden-class-names-at-runtime)
9. [Adding a New Config Value](#9-adding-a-new-config-value)

---

## 1. Problem Statement

The original `PipelineConfig` in `deepstream.py` had three compounding problems:

### Problem A — Fragile manual parsing

```python
# deepstream.py — old (REMOVED)
cfg = cls(
    max_sources=int(os.getenv("MAX_RESOURCES", "40")),
    batched_push_timeout=int(os.getenv("batched_push_timeout", "66666")),
    model_width=os.getenv("WIDTH_MODEL", "640"),   # ← returns str, not int
    model_height=os.getenv("HEIGHT_MODEL", "640"),  # ← returns str, not int
)
```

Issues:
- `os.getenv` always returns `str`. Manual `int()` cast has no error message if
  the env var contains a typo — it raises `ValueError: invalid literal for int()`.
- Missing vars silently fall back to defaults with no warning at all.
- `model_width` / `model_height` were typed as `str` on the dataclass but used
  as numeric values in GStreamer caps strings — inconsistent types throughout.

### Problem B — Duplicate source of truth for `max_sources`

```python
# Three places, three different values:
max_sources: int = 40          # PipelineConfig default
max_sources: int = 5           # DynamicRTSPPipeline.__init__ default
pipeline = DynamicRTSPPipeline(max_sources=40)  # context.py override
```

The actual value was the `context.py` override, making the other two
defaults dead code that confused anyone reading the code.

### Problem C — Config lived inside the DeepStream module

`PipelineConfig` was defined in `deepstream.py` — the GStreamer pipeline module.
Configuration is a concern of the application entry point, not the pipeline
implementation. This made it impossible to load and validate config without
importing the entire GStreamer-heavy `deepstream` module.

---

## 2. Solution Overview

| Before | After |
|--------|-------|
| `PipelineConfig` dataclass in `deepstream.py` | `PipelineSettings(BaseSettings)` in `settings.py` |
| `os.getenv` with manual `int()` casts | Pydantic validates and coerces types automatically |
| `max_sources` in three places | Removed — replaced with `_NVSTREAMMUX_BATCH_SIZE = 64` constant |
| Config created inside `deepstream.py` | Config created at app startup, injected into pipeline |
| No validation — bad env vars crash at runtime | Pydantic raises a clear error at startup with field name |

---

## 3. PipelineSettings — Field Reference

**File:** [server/backend/app/core/settings.py](../../server/backend/app/core/settings.py)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    batched_push_timeout: int = 66_666
    model_width: int = 640
    model_height: int = 640
    codec: str = "H265"
    bitrate: int = 4_000_000
    meta_serialization_lib: str = "/deepstream_app/deepstream/app/lib/serialize_meta.so"
    perf_interval_ms: int = 5_000
```

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `batched_push_timeout` | `int` | `66_666` | `nvstreammux` batched-push-timeout (µs) |
| `model_width` | `int` | `640` | YOLO input width for caps negotiation |
| `model_height` | `int` | `640` | YOLO input height for caps negotiation |
| `codec` | `str` | `"H265"` | Output codec for RTSP encoding |
| `bitrate` | `int` | `4_000_000` | Output bitrate (bps) |
| `meta_serialization_lib` | `str` | see above | Path to `serialize_meta.so` for nvunixfdsink |
| `perf_interval_ms` | `int` | `5_000` | FPS reporting interval (ms) |

---

## 4. Environment Variable Names

Pydantic reads env vars **case-insensitively** and matches them to field names.
Fields that had different names in the old code use `AliasChoices` for backward
compatibility — both the old name and the new `DS_` prefixed name are accepted:

| Field | Old env var | New env var | Notes |
|-------|-------------|-------------|-------|
| `batched_push_timeout` | `batched_push_timeout` | `DS_BATCHED_PUSH_TIMEOUT` | Either accepted |
| `model_width` | `WIDTH_MODEL` | `DS_MODEL_WIDTH` | Either accepted |
| `model_height` | `HEIGHT_MODEL` | `DS_MODEL_HEIGHT` | Either accepted |
| `codec` | *(not configurable before)* | `DS_CODEC` | New |
| `bitrate` | *(not configurable before)* | `DS_BITRATE` | New |
| `perf_interval_ms` | *(not configurable before)* | `DS_PERF_INTERVAL_MS` | New |
| `meta_serialization_lib` | *(not configurable before)* | `DS_META_SERIALIZATION_LIB` | New |

`MAX_RESOURCES` is removed. See [Section 6](#6-how-max_sources-was-removed).

### `.env` file example

```dotenv
# Inference model dimensions
WIDTH_MODEL=1280
HEIGHT_MODEL=720

# nvstreammux tuning
DS_BATCHED_PUSH_TIMEOUT=100000

# Performance reporting
DS_PERF_INTERVAL_MS=3000
```

---

## 5. GIE Configs and Hidden Classes

`gie_configs` and `hidden_class_names` use a dynamic `GIE_N_*` numbering scheme
that Pydantic fields cannot capture with a static schema. They are implemented
as `@property` methods that scan `os.environ` on each access:

```python
@property
def gie_configs(self) -> List[str]:
    """Return ordered list of GIE config paths from GIE_N_CONFIG env vars."""
    gie_map: dict = {}
    for key, value in os.environ.items():
        m = re.match(r"^GIE_(\d+)_CONFIG$", key)
        if m:
            gie_map[int(m.group(1))] = value.strip()
    return [gie_map[i] for i in sorted(gie_map)]

@property
def hidden_class_names(self) -> Set[str]:
    """Return initial set of hidden class names from GIE_N_HIDE_CLASS_NAMES env vars."""
    names: Set[str] = set()
    for key, value in os.environ.items():
        if re.match(r"^GIE_(\d+)_HIDE_CLASS_NAMES$", key):
            names.update(v.strip() for v in value.split(",") if v.strip())
    return names
```

### `.env` example for GIE

```dotenv
GIE_0_CONFIG=/deepstream_app/deepstream/config/config_pgie_yolo_detct.txt
GIE_1_CONFIG=/deepstream_app/deepstream/config/config_sgie.txt
GIE_0_HIDE_CLASS_NAMES=person,bicycle
```

`gie_configs` returns `["/deepstream_app/...detct.txt", "/deepstream_app/...sgie.txt"]` —
sorted by the numeric index, so order is deterministic regardless of how env
vars are declared.

---

## 6. How `max_sources` Was Removed

`max_sources` served two purposes:
1. Set `nvstreammux` `batch-size` — the fixed number of pad slots the muxer
   allocates at creation time.
2. Limit `SpotManager` — how many concurrent streams are allowed.

Both are now driven by a single module-level constant in `deepstream.py`:

```python
# deepstream.py
_NVSTREAMMUX_BATCH_SIZE = 64
```

**Why 64?** `nvstreammux` requires a fixed `batch-size` at pipeline creation —
it cannot be dynamic. 64 covers the practical limit of a single high-end NVIDIA
GPU running DeepStream. The actual number of streams you can run is
hardware-limited (GPU memory, compute, RTSP network bandwidth) — not a config
knob. Removing it from config prevents false confidence that bumping a number
in `.env` gives you more capacity.

If you're porting to Jetson or another platform with a lower practical limit,
change `_NVSTREAMMUX_BATCH_SIZE` in `deepstream.py` to match the hardware.

---

## 7. How the Settings Flow Through the Code

```
.env file  +  environment variables
           │
           ▼
   PipelineSettings()          ← pydantic validates + coerces types
   server/backend/app/core/settings.py
           │
           │  settings singleton
           ▼
   context.py
   pipeline = DynamicRTSPPipeline(settings=settings)
           │
           ▼
   deepstream.py  DynamicRTSPPipeline.__init__(settings)
       self._config = settings         ← immutable validated config
       self._hidden_class_names = set(settings.hidden_class_names)
                                        ← mutable runtime copy
```

`deepstream.py` receives the settings object through its constructor — it does
not import `PipelineSettings` directly. This keeps the GStreamer module free
of any dependency on the FastAPI application layer.

---

## 8. Hidden Class Names at Runtime

`hidden_class_names` on `PipelineSettings` is a read-only property — it
reflects what's in the env vars at the moment it's called. But the pipeline
needs to support `POST /hide_class_name` and `POST /enable_class_name` at
runtime, which mutate the hidden set.

To handle this, `DynamicRTSPPipeline.__init__` creates a **mutable copy**:

```python
# deepstream.py
self._hidden_class_names: set = set(settings.hidden_class_names)
```

All runtime hide/show operations work on `self._hidden_class_names`, not on the
settings object. The settings object gives the initial state from env vars;
after that, `self._hidden_class_names` is the live source of truth.

```
startup:   self._hidden_class_names = {"person", "bicycle"}  ← from env
runtime:   POST /hide_class_name?class_name=truck
           self._hidden_class_names = {"person", "bicycle", "truck"}

restart:   self._hidden_class_names = {"person", "bicycle"}  ← env again
```

Runtime changes do not persist across restarts. If you need persistence,
write `self._hidden_class_names` to a file or database on mutation.

---

## 9. Adding a New Config Value

To add a new configurable value (e.g., a maximum reconnect timeout):

**Step 1** — Add the field to `PipelineSettings`:

```python
# settings.py
rtsp_reconnect_attempts: int = 10
```

Pydantic will read `DS_RTSP_RECONNECT_ATTEMPTS` from the environment automatically.

**Step 2** — Use it in `deepstream.py` via `self._config`:

```python
# source_bin_factory or wherever needed
nvurisrc.set_property("rtsp-reconnect-attempts", self._config.rtsp_reconnect_attempts)
```

**Step 3** — Add it to `.env` if the default needs overriding:

```dotenv
DS_RTSP_RECONNECT_ATTEMPTS=20
```

That's all. No `os.getenv`, no manual type casting, no new constructor parameters.
