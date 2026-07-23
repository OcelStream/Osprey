"""
DeepStream GStreamer element factory.

Centralises element creation so that platform-specific properties
(nvbuf-memory-type) and repeated queue settings (leaky, max-size-*)
are defined in one place.

Supported platforms
-------------------
"x86"    — uses NVBUF_MEM_CUDA_UNIFIED  (desktop GPU, default)
"jetson" — uses NVBUF_MEM_DEFAULT       (Jetson / embedded GPU)

To target Jetson, set DS_PLATFORM=jetson in the environment or .env file
and pass platform="jetson" when constructing DeepStreamElementFactory.
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds


class DeepStreamElementFactory:
    """Factory for GStreamer elements with consistent platform-aware defaults."""

    def __init__(self, platform: str = "x86") -> None:
        self._mem_type = (
            int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            if platform == "x86"
            else int(pyds.NVBUF_MEM_DEFAULT)
        )

    # ------------------------------------------------------------------
    # Primitive
    # ------------------------------------------------------------------

    def make(self, factory: str, name: str) -> Gst.Element:
        """Create a GStreamer element, raising RuntimeError if it fails."""
        elem = Gst.ElementFactory.make(factory, name)
        if elem is None:
            raise RuntimeError(
                f"Failed to create GStreamer element '{factory}' as '{name}'"
            )
        return elem

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def nvvideoconvert(self, name: str) -> Gst.Element:
        """nvvideoconvert with nvbuf-memory-type set for the target platform."""
        elem = self.make("nvvideoconvert", name)
        elem.set_property("nvbuf-memory-type", self._mem_type)
        return elem

    def capsfilter(self, name: str, caps_string: str) -> Gst.Element:
        """capsfilter pre-loaded with a caps string."""
        elem = self.make("capsfilter", name)
        elem.set_property("caps", Gst.Caps.from_string(caps_string))
        return elem

    def nvtracker(
        self,
        name: str,
        ll_lib_file: str,
        ll_config_file: str,
        tracker_width: int = 640,
        tracker_height: int = 384,
    ) -> Gst.Element:
        """nvtracker configured with a low-level tracker library and YAML config."""
        elem = self.make("nvtracker", name)
        elem.set_property("tracker-width", tracker_width)
        elem.set_property("tracker-height", tracker_height)
        elem.set_property("ll-lib-file", ll_lib_file)
        elem.set_property("ll-config-file", ll_config_file)
        elem.set_property("gpu-id", 0)
        elem.set_property("display-tracking-id", 1)
        return elem

    def queue(
        self,
        name: str,
        max_buffers: int = 20,
        max_time: int = 5_000,
    ) -> Gst.Element:
        """queue with consistent leaky / max-size-* defaults.

        leaky=0          — no dropping (backpressure)
        max-size-bytes=0 — unlimited bytes
        max-size-time    — cap by time (nanoseconds); default 5 ms
        """
        elem = self.make("queue", name)
        elem.set_property("leaky", 0)
        elem.set_property("max-size-buffers", max_buffers)
        elem.set_property("max-size-bytes", 0)
        elem.set_property("max-size-time", max_time)
        return elem
