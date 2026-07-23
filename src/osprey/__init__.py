"""Osprey — a DeepStream 8.0 video-analytics library.

Two layers, installable as one package on a host that already has the
DeepStream 8.0 SDK (GStreamer plugins + ``pyds``) present:

    osprey.client  — DeepStreamClient base class (subclass to build apps)
    osprey.server  — DynamicRTSPPipeline + FastAPI control plane
"""

__version__ = "0.1.0"
