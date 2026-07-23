"""Osprey client SDK — subclass :class:`DeepStreamClient` to build apps.

Example::

    from osprey.client import DeepStreamClient, FrameData

    class VehicleCounter(DeepStreamClient):
        def _process_frame(self, frame: FrameData):
            for obj in frame.objects:
                self._draw_object(frame.surface, obj)

    VehicleCounter().start()
"""

from .base_client import (
    ClassifierResult,
    ClientConfig,
    DeepStreamClient,
    FrameData,
    LabelInfo,
    ObjectData,
    StreamRecord,
)

__all__ = [
    "DeepStreamClient",
    "ClientConfig",
    "StreamRecord",
    "FrameData",
    "ObjectData",
    "LabelInfo",
    "ClassifierResult",
]
