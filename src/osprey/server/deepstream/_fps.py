"""Minimal per-stream FPS accounting.

Drop-in replacement for the ``GETFPS`` / ``PERF_DATA`` helpers that the
DeepStream Python sample apps ship under ``apps/common/FPS.py``. Vendoring
a tiny equivalent here removes Osprey's dependency on a hardcoded path into
the DeepStream SDK sources (``.../deepstream_python_apps/apps/common``), so
the library works on any host that has the SDK, regardless of where the
Python samples live.

Interface consumed by :mod:`osprey.server.deepstream.pipeline`:
    PERF_DATA().all_stream_fps      -> dict[str, GETFPS]
    PERF_DATA().update_fps(key)     -> tick the counter for one stream
    GETFPS(key).get_fps()           -> float FPS since the last call
"""

from __future__ import annotations

import time
from threading import Lock


class GETFPS:
    """Frames-per-second counter for a single stream.

    ``get_fps()`` returns the average FPS over the window since the previous
    ``get_fps()`` call and resets the window.
    """

    def __init__(self, stream_id: str) -> None:
        self.stream_id = stream_id
        self._lock = Lock()
        self._frame_count = 0
        self._start_time = time.time()
        self._first = True

    def update_fps(self) -> None:
        with self._lock:
            if self._first:
                # Anchor the window on the first observed frame.
                self._start_time = time.time()
                self._first = False
            else:
                self._frame_count += 1

    def get_fps(self) -> float:
        now = time.time()
        with self._lock:
            elapsed = now - self._start_time
            fps = self._frame_count / elapsed if elapsed > 0 else 0.0
            self._frame_count = 0
            self._start_time = now
        return round(fps, 2)


class PERF_DATA:
    """Container tracking a :class:`GETFPS` per stream key."""

    def __init__(self, num_streams: int = 1) -> None:
        self.all_stream_fps: dict[str, GETFPS] = {}

    def update_fps(self, stream_index: str) -> None:
        counter = self.all_stream_fps.get(stream_index)
        if counter is not None:
            counter.update_fps()
