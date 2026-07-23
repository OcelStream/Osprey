"""Programmatic configuration + lifecycle for the Osprey server pipeline.

Instead of the ``osprey-server`` CLI and environment variables, a developer
configures the model/tracker and launches the control plane from Python — in a
**single file** — while their client analytics run alongside it::

    import osprey.server as osprey

    # 1. configure + start the server (forked into its own process)
    osprey.configure(gie_config="/models/gie.txt", tracker="NvSORT")
    osprey.serve()                        # returns once the server is healthy

    # 2. write your analytics with the client, in the same file
    from osprey.client import DeepStreamClient, FrameData

    class VehicleCounter(DeepStreamClient):
        def _process_frame(self, frame: FrameData):
            for obj in frame.objects:
                self._draw_object(frame.surface, obj)

    VehicleCounter().start()              # serves RTSP for every stream

:func:`serve` runs the DeepStream inference pipeline + FastAPI control plane in
a **separate (forked) process**, so its GStreamer/GLib main loop never collides
with the client's in the parent. The developer never has to manage that split.

Two lower-level entry points are also available:

* :func:`start` — run the server **blocking, in the current process** (service
  mode; this is what the ``osprey-server`` CLI uses).
* :func:`get_pipeline` — the lazily-built singleton pipeline.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from osprey.server.core.settings import PipelineSettings, settings as _env_settings

logger = logging.getLogger(__name__)

# Guarded singletons (per-process).
_settings: Optional[PipelineSettings] = None
_config_kwargs: Dict[str, Any] = {}       # raw inputs to configure(), for forking
_pipeline = None  # DynamicRTSPPipeline — typed loosely to avoid an eager import
_lock = threading.RLock()

# Track forked server processes so we can reap them at interpreter exit.
_servers: "list[ServerHandle]" = []

# URL of the most recently started server — default target for the REST helpers.
_default_url: str = "http://127.0.0.1:8000"


def configure(
    *,
    gie_config: Optional[str] = None,
    gie_configs: Optional[List[str]] = None,
    tracker: Optional[str] = None,
    model_width: Optional[int] = None,
    model_height: Optional[int] = None,
    **overrides,
) -> PipelineSettings:
    """Set global pipeline configuration. Call this before :func:`serve`/:func:`start`.

    Args:
        gie_config:   Path to a single GIE (nvinfer) config file — convenience
                      for the common single-model case.
        gie_configs:  Ordered list of GIE config paths (primary + secondary
                      inference). Takes precedence over ``gie_config``.
        tracker:      Tracker preset (``IOU`` / ``NvSORT`` / ``NvDCF`` /
                      ``NvDeepSORT``), a raw YAML path, or ``off``.
        model_width / model_height: Inference resolution the sources are
                      scaled to before the muxer.
        **overrides:  Any other :class:`PipelineSettings` field
                      (e.g. ``batched_push_timeout``, ``meta_serialization_lib``).

    Returns:
        The resolved :class:`PipelineSettings`.

    Raises:
        RuntimeError: if the pipeline has already been built in this process.
    """
    global _settings, _config_kwargs
    with _lock:
        if _pipeline is not None:
            raise RuntimeError(
                "configure() must be called before serve()/start() / the pipeline is built"
            )

        # nvstreammux's request-pad (new) mode must be enabled before the
        # GStreamer plugins are loaded, so set it now as a safe default.
        os.environ.setdefault("USE_NEW_NVSTREAMMUX", "yes")

        # Remember the raw inputs so a forked server process can re-apply them.
        raw: Dict[str, Any] = dict(overrides)
        if gie_config is not None:
            raw["gie_config"] = gie_config
        if gie_configs is not None:
            raw["gie_configs"] = gie_configs
        if tracker is not None:
            raw["tracker"] = tracker
        if model_width is not None:
            raw["model_width"] = model_width
        if model_height is not None:
            raw["model_height"] = model_height
        _config_kwargs = raw

        kwargs = dict(overrides)
        if tracker is not None:
            kwargs["tracker"] = tracker
        if model_width is not None:
            kwargs["model_width"] = model_width
        if model_height is not None:
            kwargs["model_height"] = model_height

        s = PipelineSettings(**kwargs)

        configs = gie_configs if gie_configs else ([gie_config] if gie_config else None)
        if configs:
            s.with_gie_configs(configs)

        _settings = s
    return _settings


def get_settings() -> PipelineSettings:
    """Return the active settings — programmatic if configured, else env-based."""
    global _settings
    with _lock:
        if _settings is None:
            _settings = _env_settings
        return _settings


def get_pipeline():
    """Return the singleton :class:`DynamicRTSPPipeline`, building it on first use."""
    global _pipeline
    with _lock:
        if _pipeline is None:
            # Lazy import so `import osprey.server` + configure() don't require
            # pyds/GStreamer until the pipeline is actually started.
            from osprey.server.deepstream.pipeline import DynamicRTSPPipeline

            _pipeline = DynamicRTSPPipeline(settings=get_settings())
        return _pipeline


# ---------------------------------------------------------------------------
# In-process (blocking) server — service mode / used by the CLI
# ---------------------------------------------------------------------------
def start(host: str = "0.0.0.0", port: int = 8000, background: bool = False):
    """Run the FastAPI control plane in the **current** process (blocking).

    Its lifespan builds + runs the pipeline. Use :func:`serve` instead for the
    single-file pattern where a client runs alongside the server.

    Args:
        host / port: Bind address for the control-plane API.
        background:  ``False`` blocks. ``True`` runs uvicorn in a daemon thread
                     and returns once the pipeline reaches PLAYING.

    Returns:
        The ``uvicorn.Server`` instance.
    """
    import uvicorn

    from osprey.server.app import app

    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)

    if not background:
        server.run()
        return server

    thread = threading.Thread(target=server.run, daemon=True, name="osprey-server")
    thread.start()

    deadline = time.time() + 45
    while time.time() < deadline:
        if _pipeline is not None:
            break
        time.sleep(0.1)
    if _pipeline is not None:
        _pipeline._ready.wait(timeout=30)
    return server


# ---------------------------------------------------------------------------
# Forked server — the single-file pattern
# ---------------------------------------------------------------------------
class ServerHandle:
    """Handle to a forked Osprey server process (see :func:`serve`)."""

    def __init__(self, process: multiprocessing.Process, url: str):
        self._process = process
        self.url = url

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the server process and wait for it to exit."""
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)
        if self in _servers:
            _servers.remove(self)

    def __enter__(self) -> "ServerHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _server_entry(config_kwargs: Dict[str, Any], host: str, port: int) -> None:
    """Child-process entry point: reconfigure, then run the server blocking."""
    # Fresh process → fresh module state; re-apply the developer's config here.
    if config_kwargs:
        configure(**config_kwargs)
    start(host=host, port=port, background=False)


def _wait_until_ready(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"{url}/api/v1/health/ready", timeout=2)
            if resp.status == 200:
                return
        except Exception as exc:  # noqa: BLE001 — connection refused until it's up
            last_err = exc
        time.sleep(1.0)
    raise RuntimeError(
        f"Osprey server at {url} did not become ready within {timeout:.0f}s"
        + (f" (last error: {last_err})" if last_err else "")
    )


def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    wait_ready: bool = True,
    ready_timeout: float = 180.0,
) -> ServerHandle:
    """Start the server in its **own process** and return once it is healthy.

    This is the single-file entry point: the DeepStream pipeline + control
    plane run in a forked child (isolated GStreamer/CUDA context), leaving the
    parent process free to run a :class:`~osprey.client.DeepStreamClient`.

    Call :func:`configure` first to set the model/tracker.

    Args:
        host / port:   Bind address for the control-plane API.
        wait_ready:    Block until ``/health/ready`` returns 200 (default).
        ready_timeout: Seconds to wait for readiness (first run may build the
                       TensorRT engine, which can take minutes).

    Returns:
        A :class:`ServerHandle`; call ``.stop()`` to shut the server down (it is
        also terminated automatically at interpreter exit).
    """
    # Fork BEFORE the parent initializes CUDA/GStreamer (the client does that
    # later). 'fork' avoids re-importing the developer's __main__, so no
    # ``if __name__ == '__main__'`` guard is needed in a single-file script.
    global _default_url
    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(
        target=_server_entry,
        args=(dict(_config_kwargs), host, port),
        name="osprey-server",
        daemon=True,
    )
    proc.start()

    connect_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    handle = ServerHandle(proc, f"http://{connect_host}:{port}")
    _servers.append(handle)
    _default_url = handle.url

    if wait_ready:
        try:
            _wait_until_ready(handle.url, ready_timeout)
        except Exception:
            handle.stop()
            raise
    return handle


# ---------------------------------------------------------------------------
# REST helpers — manage streams on a running server from the client process
# ---------------------------------------------------------------------------
import json as _json


def _post(path: str, payload: dict, base_url: Optional[str], timeout: float) -> dict:
    url = f"{base_url or _default_url}{path}"
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode())


def add_stream(
    uri: str,
    stream_id: str,
    rtsp_output_width: int = 640,
    rtsp_output_height: int = 640,
    *,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> dict:
    """Add a video source to the running server (POST /api/v1/add).

    Args:
        uri:        ``file:///...`` or ``rtsp://...`` source.
        stream_id:  Unique id; also the RTSP mount suffix (``/ds-test<id>``).
        rtsp_output_width / rtsp_output_height: Output frame size.
        base_url:   Server URL (defaults to the last :func:`serve` target).
        timeout:    Request timeout — the first add may build a TensorRT engine.

    Returns:
        The server response dict. If ``stream_id`` is already active, the add is
        **skipped** (a warning is logged) and ``{"skipped": True, ...}`` is
        returned instead of raising — so a duplicate never crashes the caller.
    """
    # Guard: don't re-add an existing stream_id (a duplicate add would otherwise
    # 400 and, at module scope, crash the app before the client starts).
    try:
        if any(s.get("stream_id") == stream_id for s in list_streams(base_url=base_url)):
            logger.warning(
                "Stream '%s' not added — a stream with this stream_id already exists.",
                stream_id,
            )
            return {"skipped": True, "reason": "duplicate stream_id", "stream_id": stream_id}
    except Exception:
        # If the server can't be reached for the pre-check, fall through — the
        # server itself still guards against duplicates below.
        pass

    try:
        return _post(
            "/api/v1/add",
            {
                "uri": uri,
                "stream_id": stream_id,
                "rtsp_output_width": rtsp_output_width,
                "rtsp_output_height": rtsp_output_height,
            },
            base_url,
            timeout,
        )
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()
        except Exception:
            pass
        if exc.code == 400 and "already exists" in detail:
            logger.warning(
                "Stream '%s' not added — a stream with this stream_id already exists.",
                stream_id,
            )
            return {"skipped": True, "reason": "duplicate stream_id", "stream_id": stream_id}
        # Any other failure is surfaced with the server's message.
        raise RuntimeError(f"add_stream failed ({exc.code}): {detail or exc.reason}") from None


def remove_stream(
    stream_id: str, *, base_url: Optional[str] = None, timeout: float = 10.0
) -> dict:
    """Remove a stream from the running server (DELETE /api/v1/remove/<id>)."""
    url = f"{base_url or _default_url}/api/v1/remove/{stream_id}"
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode())


def list_streams(*, base_url: Optional[str] = None, timeout: float = 10.0) -> list:
    """List active streams on the running server (GET /api/v1/streams)."""
    url = f"{base_url or _default_url}/api/v1/streams"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return _json.loads(resp.read().decode())


def stop() -> None:
    """Stop the in-process pipeline and any forked server(s)."""
    global _pipeline
    with _lock:
        if _pipeline is not None:
            _pipeline.stop()
    for handle in list(_servers):
        handle.stop()


@atexit.register
def _cleanup_servers() -> None:
    for handle in list(_servers):
        try:
            handle.stop()
        except Exception:
            pass
