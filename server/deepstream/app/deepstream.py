"""
Dynamic DeepStream pipeline with runtime source management.

Supports adding and removing video sources at runtime. Each source gets
a dedicated output branch with nvunixfdsink streaming frames to a Unix
domain socket at /run/nvunixfd/<stream_id>.sock.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import cv2
import gi

gi.require_version("Gst", "1.0")

from gi.repository import GLib, Gst

sys.path.append(
    "/opt/nvidia/deepstream/deepstream-8.0/sources/deepstream_python_apps/apps/common"
)

import pyds
from element_factory import DeepStreamElementFactory
from FPS import GETFPS, PERF_DATA
from source_bin_factory import SourceBinFactory
from spotmanager import SpotManager

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Hardware constant — nvstreammux requires a fixed batch-size at creation.
# Sources are hardware-limited; this value covers typical GPU capacity.
# ---------------------------------------------------------------------------
_NVSTREAMMUX_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Per-stream state
# ---------------------------------------------------------------------------
@dataclass
class StreamRecord:
    """All GStreamer resources associated with a single active stream."""

    stream_id: str
    uri: str
    spot: int
    source_bin: object  # Gst.Bin
    branch_elements: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class DynamicRTSPPipeline:
    """DeepStream pipeline supporting runtime add / remove of video sources.

    Each source gets a dedicated ``nvunixfdsink`` output branch reachable
    at ``/run/nvunixfd/<stream_id>.sock``.
    """

    def __init__(
        self,
        settings: Any,
        notification_callback: Optional[Callable] = None,
    ):
        Gst.init(None)

        self._config = settings
        self._notification_callback = notification_callback
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Serializes concurrent add/remove from different threads
        self._lock = threading.Lock()

        # Signals that the pipeline has reached PLAYING state
        self._ready = threading.Event()

        # --- Helpers (must be created before building pipeline elements) ---
        self._spot_manager = SpotManager(_NVSTREAMMUX_BATCH_SIZE)
        self._source_factory = SourceBinFactory()
        self._element_factory = DeepStreamElementFactory()
        self._perf_data = PERF_DATA()
        self._loop = GLib.MainLoop()

        # --- Core GStreamer elements ---
        self._pipeline = Gst.Pipeline()

        self._streammux = self._create_element("nvstreammux", "stream-mux")
        self._streammux.set_property("batch-size", _NVSTREAMMUX_BATCH_SIZE)
        self._streammux.set_property(
            "batched-push-timeout", self._config.batched_push_timeout
        )
        self._pipeline.add(self._streammux)

        self._gies = self._build_inference_chain()
        self._tracker = self._build_tracker()

        self._demux = self._create_element("nvstreamdemux", "stream-demux")
        self._pipeline.add(self._demux)

        # Link: last GIE → tracker → demux
        tail = self._gies[-1] if self._gies else self._streammux
        if self._tracker:
            tail.link(self._tracker)
            self._tracker.link(self._demux)
        else:
            tail.link(self._demux)

        # --- Stream bookkeeping (guarded by _lock) ---
        self._streams: Dict[str, StreamRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_active_streams(self) -> list:
        """Return a list of currently active stream records."""
        with self._lock:
            return [
                {"stream_id": rec.stream_id, "uri": rec.uri, "spot": rec.spot}
                for rec in self._streams.values()
            ]
    def add_source(
        self,
        uri: str,
        rtsp_output_width: int = 640,
        rtsp_output_height: int = 640,
        stream_id: Optional[str] = None,
    ) -> str:
        """Add a video source and wire up its output branch.

        Returns *stream_id*.  Raises ``RuntimeError`` on validation failure,
        capacity limit, or pipeline error.
        """
        self._require_playing()
        self._validate_uri(uri)

        with self._lock:
            if stream_id in self._streams:
                raise RuntimeError(f"Stream {stream_id} already exists")
            spot, _, _ = self._spot_manager.acquire()
            if spot is None:
                raise RuntimeError("No available spots — max source capacity reached")

            try:
                src_bin = self._attach_source(stream_id, uri, spot)
                self._attach_preprocessing(
                    stream_id, src_bin.get_static_pad("src"), spot
                )
                branch = self._build_output_branch(
                    spot, stream_id, rtsp_output_width, rtsp_output_height
                )

                self._streams[stream_id] = StreamRecord(
                    stream_id=stream_id,
                    uri=uri,
                    spot=spot,
                    source_bin=src_bin,
                    branch_elements=branch,
                )

                src_bin.set_state(Gst.State.PAUSED)
                src_bin.get_state(10 * Gst.SECOND)
                src_bin.set_state(Gst.State.PLAYING)
            except Exception:
                self._spot_manager.release(spot)
                raise

        logger.info("Added stream %s on spot %d (uri=%s)", stream_id, spot, uri)
        return stream_id

    def remove_source(self, stream_id: str) -> None:
        """Tear down all GStreamer elements for *stream_id* and free its spot."""
        with self._lock:
            record = self._streams.pop(stream_id, None)
            if record is None:
                logger.warning("remove_source: unknown stream_id %s", stream_id)
                return

            self._teardown_source(record)
            self._teardown_branch(record)
            self._spot_manager.release(record.spot)
            self._perf_data.all_stream_fps.pop(f"stream{record.spot}", None)

        logger.info("Removed stream %s (spot %d)", stream_id, record.spot)

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    def start(self, event_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Block the calling thread running the GLib main loop.

        Pass *event_loop* so that async notification callbacks can be
        scheduled from the GLib thread.
        """
        self._event_loop = event_loop

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, self._loop)
        GLib.timeout_add(self._config.perf_interval_ms, self._on_perf_tick)

        time.sleep(1)
        self._pipeline.set_state(Gst.State.PLAYING)
        self._ready.set()

        try:
            self._perf_data.all_stream_fps.pop("stream0", None)
            self._loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._pipeline.set_state(Gst.State.NULL)

    def stop(self) -> None:
        """Signal the GLib main loop to quit and set the pipeline to NULL.

        Safe to call from any thread. After this returns, the thread running
        :meth:`start` will exit its ``_loop.run()`` block.
        """
        self._loop.quit()
        self._pipeline.set_state(Gst.State.NULL)
        logger.info("Pipeline stopped")

    # ------------------------------------------------------------------
    # Internals — element helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_element(factory: str, name: str) -> Gst.Element:
        elem = Gst.ElementFactory.make(factory, name)
        if elem is None:
            raise RuntimeError(
                f"Failed to create GStreamer element '{factory}' as '{name}'"
            )
        return elem

    def _remove_element(self, name: str) -> None:
        """Set an element to NULL and remove it from the pipeline (no-op if absent)."""
        elem = self._pipeline.get_by_name(name)
        if elem:
            elem.set_state(Gst.State.NULL)
            self._pipeline.remove(elem)

    # ------------------------------------------------------------------
    # Internals — inference chain
    # ------------------------------------------------------------------
    def _build_inference_chain(self) -> list:
        """Create and link nvinfer elements between streammux and demux."""
        gies = []
        prev = self._streammux
        for i, config_path in enumerate(self._config.gie_configs):
            gie = self._create_element("nvinfer", f"gie_{i}")
            gie.set_property("config-file-path", config_path)
            self._pipeline.add(gie)
            prev.link(gie)
            gies.append(gie)
            prev = gie
        return gies

    # ------------------------------------------------------------------
    # Internals — tracker
    # ------------------------------------------------------------------
    def _build_tracker(self) -> Optional[Gst.Element]:
        """Create nvtracker element and add it to the pipeline."""
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
        logger.info(
            "Tracker enabled: DS_TRACKER=%s → %s",
            self._config.tracker,
            self._config.tracker_config,
        )
        return tracker

    # ------------------------------------------------------------------
    # Internals — source attach / detach
    # ------------------------------------------------------------------
    def _attach_source(self, stream_id: str, uri: str, spot: int) -> Gst.Bin:
        src_bin = self._source_factory.create_source_bin(
            stream_id, uri, "nvurisrcbin"
        )
        self._remove_element(f"source-bin-{stream_id}")
        self._pipeline.add(src_bin)
        return src_bin

    def _attach_preprocessing(self, stream_id: str, src_pad, spot: int) -> None:
        """Insert nvvideoconvert + capsfilter between source and streammux."""
        for prefix in ("conv", "capsfilter"):
            self._remove_element(f"{prefix}_{stream_id}")

        conv = self._element_factory.nvvideoconvert(f"conv_{stream_id}")
        caps = self._element_factory.capsfilter(
            f"capsfilter_{stream_id}",
            f"video/x-raw(memory:NVMM), format=NV12, "
            f"width={self._config.model_width}, height={self._config.model_height}",
        )

        self._pipeline.add(conv)
        self._pipeline.add(caps)
        conv.sync_state_with_parent()
        caps.sync_state_with_parent()

        src_pad.link(conv.get_static_pad("sink"))
        conv.link(caps)

        mux_pad = self._streammux.get_static_pad(
            f"sink_{spot}"
        ) or self._streammux.get_request_pad(f"sink_{spot}")
        if not mux_pad:
            raise RuntimeError(f"Cannot obtain streammux pad sink_{spot}")

        caps.get_static_pad("src").link(mux_pad)

    def _teardown_source(self, rec: StreamRecord) -> None:
        rec.source_bin.set_state(Gst.State.NULL)

        mux_pad = self._streammux.get_static_pad(f"sink_{rec.spot}")
        if mux_pad:
            peer = mux_pad.get_peer()
            if peer:
                peer.unlink(mux_pad)

        for prefix in ("conv", "capsfilter"):
            self._remove_element(f"{prefix}_{rec.stream_id}")

        self._pipeline.remove(rec.source_bin)

    # ------------------------------------------------------------------
    # Internals — output branch
    # ------------------------------------------------------------------
    _BRANCH_CHAIN = (
        "q_demux",
        "conv1",
        "caps_rgba",
        "osd",
        "conv2",
        "caps_nv12",
        "q_fd",
        "fdsink",
    )

    def _build_output_branch(
        self,
        spot: int,
        stream_id: str,
        width: int,
        height: int,
    ) -> Dict[str, object]:
        """Wire demux -> queue -> convert(RGBA) -> OSD -> convert(NV12) -> queue -> fdsink."""
        elems = {
            "q_demux": self._element_factory.queue(f"q_demux_{stream_id}"),
            "conv1":   self._element_factory.nvvideoconvert(f"conv1_{stream_id}"),
            "caps_rgba": self._element_factory.capsfilter(
                f"caps_rgba_{stream_id}",
                f"video/x-raw(memory:NVMM),format=RGBA,width={width},height={height}",
            ),
            "osd":     self._create_element("nvdsosd", f"osd_{stream_id}"),
            "conv2":   self._create_element("nvvideoconvert", f"conv2_{stream_id}"),
            "caps_nv12": self._element_factory.capsfilter(
                f"caps_nv12_{stream_id}",
                "video/x-raw(memory:NVMM),format=NV12",
            ),
            "q_fd":    self._element_factory.queue(f"q_fd_{stream_id}"),
            "fdsink":  self._create_element("nvunixfdsink", f"fdsink_{stream_id}"),
        }

        elems["osd"].set_property("display-bbox", 0)
        elems["osd"].set_property("display-mask", 0)
        elems["osd"].set_property("display-text", 0)

        socket_path = f"/run/nvunixfd/{stream_id}.sock"
        fdsink = elems["fdsink"]
        fdsink.set_property("socket-path", socket_path)
        fdsink.set_property("sync", True)
        fdsink.set_property("async", False)
        fdsink.set_property("buffer-timestamp-copy", True)
        fdsink.set_property(
            "meta-serialization-lib", self._config.meta_serialization_lib
        )

        # Purge stale leftovers, then add and sync
        # for elem in elems.values():
        #     self._remove_element(elem.get_name())
        for elem in elems.values():
            self._pipeline.add(elem)
            elem.sync_state_with_parent()

        # Link demux request pad into the chain
        self._demux.get_request_pad(f"src_{spot}").link(
            elems["q_demux"].get_static_pad("sink")
        )
        for a, b in zip(self._BRANCH_CHAIN, self._BRANCH_CHAIN[1:]):
            if not elems[a].link(elems[b]):
                raise RuntimeError(f"Failed to link {a} -> {b}")

        # FPS probe on the OSD sink pad
        elems["osd"].get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, self._fps_probe, 0
        )

        logger.info("Output branch ready — socket %s", socket_path)
        return elems

    def _teardown_branch(self, rec: StreamRecord) -> None:
        q_demux = rec.branch_elements.get("q_demux")
        if q_demux:
            demux_pad = self._demux.get_static_pad(f"src_{rec.spot}")
            if demux_pad:
                demux_pad.unlink(q_demux.get_static_pad("sink"))
                self._demux.release_request_pad(demux_pad)

        for elem in rec.branch_elements.values():
            elem.set_state(Gst.State.NULL)
            self._pipeline.remove(elem)

    # ------------------------------------------------------------------
    # Probes & callbacks
    # ------------------------------------------------------------------
    def _fps_probe(self, pad, info, _user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            key = f"stream{frame_meta.pad_index}"
            if key not in self._perf_data.all_stream_fps:
                self._perf_data.all_stream_fps[key] = GETFPS(key)
            self._perf_data.update_fps(key)
            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    def _on_bus_message(self, bus, message, loop):
        mtype = message.type

        if mtype == Gst.MessageType.ELEMENT:
            self._handle_element_msg(message)
        elif mtype == Gst.MessageType.EOS:
            logger.warning("Global pipeline EOS — all sources ended")
        elif mtype == Gst.MessageType.ERROR:
            self._handle_error_msg(message)

        return True

    def _handle_element_msg(self, message) -> None:
        struct = message.get_structure()
        if not struct:
            return

        name = struct.get_name()
        if name == "attempt-exceeded":
            found, spot = struct.get_uint("stream-id")
            if not found:
                return
            stream_id = self._stream_id_for_spot(spot)
            if stream_id is None:
                logger.warning("attempt-exceeded for unknown spot %d", spot)
                return
            logger.warning("Reconnect attempts exceeded for stream %s", stream_id)
            self._notify({"type": "attempt_exceeded", "stream_id": stream_id})
            self.remove_source(stream_id)

        elif name == "GstNvStreamEos":
            found, spot = struct.get_uint("stream-id")
            if not found:
                return
            stream_id = self._stream_id_for_spot(spot)
            if stream_id is None:
                logger.warning("EOS for unknown spot %d", spot)
                return
            logger.info("EOS for stream %s", stream_id)
            self.remove_source(stream_id)

    def _handle_error_msg(self, message) -> None:
        err, debug = message.parse_error()
        struct = message.get_structure()

        stream_id = None
        if struct:
            found, spot = struct.get_uint("stream-id")
            if found:
                stream_id = self._stream_id_for_spot(spot)

        if stream_id is None:
            logger.error("Pipeline error (no stream): %s (%s)", err.message, debug)
            return

        logger.error("Stream %s error: %s (%s)", stream_id, err.message, debug)
        self._notify(
            {
                "type": "error",
                "stream_id": stream_id,
                "message": str(err.message),
                "debug": debug,
            }
        )
        self.remove_source(stream_id)

    def _on_perf_tick(self) -> bool:
        fps_report = {
            sid: fps.get_fps()
            for sid, fps in self._perf_data.all_stream_fps.items()
        }
        logger.info("FPS data: %s", fps_report)
        self._notify({"type": "performance", "FPS": fps_report})
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _stream_id_for_spot(self, spot: int) -> Optional[str]:
        """Reverse-lookup: translate a mux pad index back to a stream_id."""
        for rec in self._streams.values():
            if rec.spot == spot:
                return rec.stream_id
        return None

    def _notify(self, payload: dict) -> None:
        """Fire the async notification callback if both callback and loop exist."""
        if self._notification_callback and self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._notification_callback(payload),
                self._event_loop,
            )

    def _require_playing(self) -> None:
        if self._pipeline.get_state(1).state != Gst.State.PLAYING:
            raise RuntimeError("Pipeline is not running — call start() first")

    def _validate_uri(self, uri: str) -> None:
        if uri.startswith("file:///"):
            path = uri[7:]
            if not os.path.isfile(path):
                raise RuntimeError(f"File not found: {path}")
        elif uri.startswith("rtsp://"):
            if not self._probe_rtsp(uri):
                raise RuntimeError(f"RTSP stream unreachable: {uri}")
        else:
            raise RuntimeError(f"Unsupported URI scheme: {uri}")

    @staticmethod
    def _probe_rtsp(uri: str) -> bool:
        """Quick reachability check via OpenCV (blocking)."""
        cap = cv2.VideoCapture(uri)
        try:
            return cap.isOpened()
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from backend.app.core.settings import settings
    pipeline = DynamicRTSPPipeline(settings=settings)
    threading.Thread(target=pipeline.start, daemon=True).start()
    time.sleep(3)

    test_uri = "file:///deepstream_app/static/parking-trucks_good.mp4"
    try:
        sid = pipeline.add_source(
            test_uri,
            rtsp_output_width=640,
            rtsp_output_height=640,
            stream_id="test-1",
        )
        logger.info("Added source: %s", sid)
    except RuntimeError as exc:
        logger.error("Failed: %s", exc)


if __name__ == "__main__":
    main()
