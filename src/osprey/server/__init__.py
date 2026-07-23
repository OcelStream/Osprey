"""Osprey server — DeepStream inference pipeline + FastAPI control plane.

Requires the ``[server]`` extra (FastAPI/uvicorn/pydantic) plus the
DeepStream 8.0 SDK (``pyds``, GStreamer plugins) on the host.

Single-file usage — the server runs in its own (forked) process, the client
runs in yours::

    import osprey.server as osprey

    osprey.configure(gie_config="/models/gie.txt", tracker="NvSORT")
    osprey.serve()                         # forks the server; returns when healthy

    from osprey.client import DeepStreamClient, FrameData
    class VehicleCounter(DeepStreamClient):
        def _process_frame(self, frame: FrameData):
            for obj in frame.objects:
                self._draw_object(frame.surface, obj)
    VehicleCounter().start()               # serves RTSP for every stream

Importing this module and calling :func:`configure` is light — the heavy
``pyds``/GStreamer pipeline is only built when :func:`serve`/:func:`start` (or
:func:`get_pipeline`) is called. Use :func:`start` to run the server blocking
in the current process (service mode / the ``osprey-server`` CLI).
"""

from osprey.server.core.context import (
    ServerHandle,
    add_stream,
    configure,
    get_pipeline,
    get_settings,
    list_streams,
    remove_stream,
    serve,
    start,
    stop,
)

__all__ = [
    "configure",
    "serve",
    "start",
    "stop",
    "add_stream",
    "remove_stream",
    "list_streams",
    "get_pipeline",
    "get_settings",
    "ServerHandle",
]
