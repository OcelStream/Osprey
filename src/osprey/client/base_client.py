#!/usr/bin/env python3

"""
DeepStream IPC Client — base class for building applications on top of
DeepStream inference results.

Handles all GStreamer/pyds boilerplate: socket watching, pipeline
construction, buffer iteration, metadata extraction.  Subclasses receive
clean ``FrameData`` / ``ObjectData`` Python objects — no pyds or
GStreamer knowledge required.

Override points (hooks):
    _process_frame(frame_data)      — called once per frame with clean data
    _draw_object(surface, obj)      — called once per detected object
    _draw_info_text(surface, frame) — draw HUD overlay
    _on_stream_added(record)        — stream lifecycle
    _on_stream_removed(record)      — stream lifecycle
"""

from __future__ import annotations

import logging
import os
import pathlib
import socket
import stat
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")

from gi.repository import GLib, GObject, Gst, GstRtspServer

import pyds

from osprey.paths import default_socket_dir, ensure_socket_dir

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ClientConfig:
    """Client configuration — sourced once from environment and constructor args.

    Subclass this to add application-specific fields, then point
    ``DeepStreamClient._config_class`` at your subclass.
    """

    rtsp_port: str = "8554"
    watch_dir: str = field(default_factory=default_socket_dir)
    watch_interval: float = 1.0
    stale_retry_interval: float = 30.0
    socket_connect_timeout: float = 2.0
    save_frames: bool = False
    drawing_type: str = "native"
    show_info_text: bool = True
    meta_deserialization_lib: str = str(pathlib.Path(__file__).resolve().parent / "lib" / "deserialize_meta.so")
    encoder_bitrate: int = 1_000_000
    udp_base_port: int = 5400

    @classmethod
    def from_env(
        cls,
        rtsp_port: str = "8554",
        watch_dir: Optional[str] = None,
    ) -> "ClientConfig":
        """Build config from environment variables and explicit arguments.

        ``watch_dir=None`` resolves to ``$OSPREY_SOCKET_DIR`` or ``./sockets``.
        """
        return cls(
            rtsp_port=rtsp_port,
            watch_dir=ensure_socket_dir(watch_dir),
            save_frames=os.environ.get("SAVE_FRAMES", "0") == "1",
            drawing_type=os.environ.get("DRAWING_TYPE", "native"),
            show_info_text=bool(os.environ.get("SHOW_INFO_TEXT", True)),
        )


# ---------------------------------------------------------------------------
# Per-stream state
# ---------------------------------------------------------------------------
@dataclass
class StreamRecord:
    """All resources associated with a single active client stream.

    Subclass to carry application-specific per-stream state.
    """

    socket_path: str
    uuid: str
    index: int
    pipeline: object  # Gst.Pipeline


# ---------------------------------------------------------------------------
# Clean per-frame data — what subclasses receive
# ---------------------------------------------------------------------------
@dataclass
class LabelInfo:
    """One label result from a secondary classifier."""

    label_id: int = 0
    result_class_id: int = 0
    result_prob: float = 0.0
    result_label: str = ""


@dataclass
class ClassifierResult:
    """One secondary classifier attached to an object."""

    unique_component_id: int = 0
    labels: List[LabelInfo] = field(default_factory=list)


@dataclass
class ObjectData:
    """One detected object — clean Python data, no pyds types."""

    class_id: int
    confidence: float
    # clipped rect (used for drawing)
    left: int
    top: int
    width: int
    height: int
    # raw detector bbox (may exceed frame bounds)
    detector_left: float = 0.0
    detector_top: float = 0.0
    detector_width: float = 0.0
    detector_height: float = 0.0
    # tracker-smoothed bbox
    tracker_left: float = 0.0
    tracker_top: float = 0.0
    tracker_width: float = 0.0
    tracker_height: float = 0.0
    tracker_confidence: float = 0.0
    object_id: int = 0
    label: str = ""
    unique_component_id: int = 0
    # segmentation mask dimensions (float data via raw_meta.mask_params)
    mask_width: int = 0
    mask_height: int = 0
    mask_threshold: float = 0.0
    # secondary classifier results
    classifiers: List[ClassifierResult] = field(default_factory=list)
    raw_meta: object = field(default=None, repr=False)


@dataclass
class FrameData:
    """One video frame with all its detections — passed to ``_process_frame``.

    ``surface`` is a writable numpy ndarray (RGBA).  Draw on it with
    OpenCV and the result appears in the RTSP output.
    """

    frame_num: int
    source_id: int
    batch_id: int
    pad_index: int
    surface: object  # numpy ndarray (writable)
    buf_pts: int = 0
    ntp_timestamp: int = 0
    objects: List[ObjectData] = field(default_factory=list)
    raw_meta: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class DeepStreamClient:
    """Base class for DeepStream IPC client applications.

    Handles socket watching, pipeline construction, GStreamer buffer
    iteration, and metadata extraction.  Subclasses override hooks to
    implement application logic — no GStreamer or pyds knowledge needed.

    Hook methods (override in subclass):
        ``_process_frame``     — receives ``FrameData`` per frame
        ``_draw_object``       — receives surface + ``ObjectData`` per object
        ``_draw_info_text``    — receives surface + ``FrameData`` for HUD
        ``_on_stream_added``   — called after a stream starts playing
        ``_on_stream_removed`` — called after a stream is torn down
    """

    _config_class = ClientConfig

    _SOURCE_CHAIN = ("source", "identity", "caps_nv", "queue")
    _OUTPUT_CHAIN = (
        "streammux",
        "nvvidconv",
        "caps_rgba",
        "nvosd",
        "nvvidconv2",
        "caps_nv12",
        "enc",
        "parse",
        "pay",
        "sink",
    )

    def __init__(
        self,
        rtsp_port: str = "8554",
        watch_dir: Optional[str] = None,
        config: Optional[ClientConfig] = None,
    ):
        """
        Initialize the DeepStream client.

        Args:
            rtsp_port: Port for the RTSP server
            watch_dir: Directory to watch for socket files. Defaults to
                       ``$OSPREY_SOCKET_DIR`` or ``./sockets`` under the
                       working directory — created if it doesn't exist.
            config: Pre-built config (skips from_env if provided)
        """
        Gst.init(None)

        self._config = config or self._config_class.from_env(
            rtsp_port=rtsp_port, watch_dir=watch_dir
        )
        # A pre-built config may name a directory that doesn't exist yet; the
        # watcher needs it present to scan.
        ensure_socket_dir(self._config.watch_dir)

        # --- RTSP server ---
        self._rtsp_server = GstRtspServer.RTSPServer()
        self._rtsp_server.props.service = self._config.rtsp_port
        self._rtsp_server.attach(None)
        self._mount_points = self._rtsp_server.get_mount_points()

        # --- Stream bookkeeping (guarded by _lock) ---
        self._streams: Dict[str, StreamRecord] = {}
        self._stream_index_counter = 0
        self._lock = threading.Lock()

        # --- Stale socket tracking (guarded by _stale_lock) ---
        self._stale_sockets: Dict[str, float] = {}
        self._stale_lock = threading.Lock()

        # --- Active-check cache (guarded by _check_lock) ---
        # Probe each socket for a live server AT MOST ONCE. The initial scan and
        # the directory watcher would otherwise both connect to a fresh socket,
        # and every throwaway connect makes the server's nvunixfdsink log
        # "Failed to send caps to new client ... Broken pipe". Caching the
        # positive result (and claiming a probe in-flight) removes the duplicate.
        self._checked_active: set = set()   # sockets confirmed to have a live server
        self._checking: set = set()         # sockets currently being probed
        self._check_lock = threading.Lock()

        # --- Watcher ---
        self._watcher_running = False
        self._watcher_thread: Optional[threading.Thread] = None

        # --- Main loop ---
        self._loop: Optional[GObject.MainLoop] = None

        logger.info(
            "Initialized — RTSP port %s, watching %s",
            self._config.rtsp_port,
            self._config.watch_dir,
        )
        logger.info(
            "Stale socket retry interval: %.0fs",
            self._config.stale_retry_interval,
        )

    # ------------------------------------------------------------------
    # Hooks — override these in subclasses
    # ------------------------------------------------------------------
    def _process_frame(self, frame_data: FrameData) -> None:
        """Process a single frame.  Override for custom application logic.

        The base implementation draws info text and bounding boxes.
        A subclass can do anything: analytics, custom overlays, saving
        frames, sending alerts, etc.

        Args:
            frame_data: Clean per-frame data (surface, objects, metadata).
                        ``frame_data.surface`` is a writable numpy array —
                        draw on it with OpenCV.
        """
        if self._config.show_info_text:
            self._draw_info_text(frame_data.surface, frame_data)
        for obj in frame_data.objects:
            self._draw_object(frame_data.surface, obj)

    def _draw_object(self, surface, obj: ObjectData) -> None:
        """Draw a single detected object with bbox, label, tracking ID, and confidence.

        Args:
            surface: Writable numpy ndarray (RGBA frame).
            obj: ObjectData with bbox, class_id, confidence, label.
        """
        color = self._class_color(obj.class_id)
        self._draw_mask(surface, obj)
        self._draw_bounding_box(
            surface, obj.left, obj.top, obj.width, obj.height, color=color
        )
        self._draw_label(surface, obj, color=color)

    def _draw_info_text(self, surface, frame_data: FrameData) -> None:
        """Draw HUD overlay (stream/frame info).  Override for custom HUD.

        Args:
            surface: Writable numpy ndarray (RGBA frame).
            frame_data: FrameData for the current frame.
        """
        info_text = (
            f"Stream: {frame_data.source_id} | "
            f"Frame: {frame_data.frame_num}"
        )
        font_scale = 0.4
        thickness = 1
        border_thickness = 3
        text_color = (0, 0, 0)
        border_color = (255, 255, 255)

        (_, text_height), _ = cv2.getTextSize(
            info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        y_pos = 10 + text_height
        x_pos = 10

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                cv2.putText(
                    surface,
                    info_text,
                    (x_pos + dx, y_pos + dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    border_color,
                    border_thickness,
                    lineType=cv2.LINE_AA,
                )

        cv2.putText(
            surface,
            info_text,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    def _on_stream_added(self, record: StreamRecord) -> None:
        """Called after a new stream is successfully added and playing.

        Override to register cameras, start analytics, etc.

        Args:
            record: The StreamRecord for the new stream.
        """

    def _on_stream_removed(self, record: StreamRecord) -> None:
        """Called after a stream is torn down.

        Override to unregister cameras, flush analytics, etc.

        Args:
            record: The StreamRecord that was just removed.
        """

    # ------------------------------------------------------------------
    # Public API — source management
    # ------------------------------------------------------------------
    def add_stream(self, socket_path: str) -> int:
        """Manually add a stream (for backwards compatibility).

        Args:
            socket_path: Unix domain socket path

        Returns:
            Stream index on success, -1 on failure
        """
        with self._lock:
            if socket_path in self._streams:
                logger.info("Stream already exists for: %s", socket_path)
                return self._streams[socket_path].index

            index = self._stream_index_counter
            self._stream_index_counter += 1

        logger.info("Adding stream %d for socket: %s", index, socket_path)

        pipeline = self.create_pipeline(socket_path, stream_id=index)
        if not pipeline:
            logger.error("Failed to create pipeline for: %s", socket_path)
            return -1

        with self._lock:
            self._streams[socket_path] = StreamRecord(
                socket_path=socket_path,
                uuid=self._parse_uuid(socket_path),
                index=index,
                pipeline=pipeline,
            )

        return index

    def get_active_streams(self) -> dict:
        """Get information about active streams."""
        with self._lock:
            return {
                rec.socket_path: {
                    "stream_id": rec.index,
                    "rtsp_url": (
                        f"rtsp://localhost:{self._rtsp_server.props.service}"
                        f"/ds-test{rec.index}"
                    ),
                }
                for rec in self._streams.values()
            }

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    def start(self, wait_for_sockets: bool = True):
        """Start the client, optionally watching for new sockets.

        Args:
            wait_for_sockets: If True, watch for sockets dynamically.
                              If False, only use manually added streams.
        """
        self._loop = GObject.MainLoop()

        with self._lock:
            for rec in self._streams.values():
                bus = rec.pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect("message", self._on_bus_message, rec.socket_path)

                ret = rec.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    logger.error(
                        "Unable to set pipeline %d to PLAYING", rec.index
                    )
                else:
                    logger.info("Pipeline %d is PLAYING...", rec.index)

        if wait_for_sockets:
            self._start_watcher()
            self._initial_scan()

        logger.info("Client is running!")
        logger.info("Watching for sockets in: %s", self._config.watch_dir)
        logger.info(
            "RTSP streams available at: rtsp://localhost:%s/ds-test<N>",
            self._rtsp_server.props.service,
        )

        try:
            self._loop.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        self.stop()
        return 0

    def stop(self):
        """Stop all pipelines and cleanup."""
        logger.info("Stopping client...")

        self._watcher_running = False
        if self._watcher_thread:
            self._watcher_thread.join(timeout=3.0)

        with self._lock:
            for rec in self._streams.values():
                logger.info("Stopping pipeline %d...", rec.index)
                rec.pipeline.set_state(Gst.State.NULL)
            self._streams.clear()

        logger.info("Client stopped")

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------
    def create_pipeline(
        self,
        socket_path: str,
        stream_id: int = 0,
        index: int = 0,
    ):
        """Create receive pipeline for a socket.

        Args:
            socket_path: Unix domain socket path to receive buffers from
            stream_id: Stream identifier for RTSP mount naming
            index: Numeric index for element naming and UDP port

        Returns:
            GStreamer pipeline or None on failure
        """
        logger.info(
            "Creating pipeline for socket: %s (stream_id=%d)",
            socket_path,
            index,
        )

        pipeline = Gst.Pipeline()
        if not pipeline:
            logger.error("Unable to create Pipeline")
            return None

        elems = self._create_pipeline_elements(index)
        if elems is None:
            return None

        self._configure_source(elems, socket_path)
        self._configure_identity(elems)
        self._configure_mux(elems)
        self._configure_caps(elems)
        self._configure_nvvidconv(elems)
        self._configure_osd(elems)
        self._configure_encoder(elems)
        self._configure_payloader(elems)
        self._configure_sink(elems, index)

        for elem in elems.values():
            pipeline.add(elem)

        if not self._link_pipeline(elems):
            return None

        caps_rgba_pad = elems["caps_rgba"].get_static_pad("src")
        if caps_rgba_pad:
            caps_rgba_pad.add_probe(
                Gst.PadProbeType.BUFFER, self._osd_probe, index
            )
            logger.info("Added probe on caps_rgba src pad (RGBA format)")

        self._setup_rtsp_mount(stream_id, index)
        return pipeline

    # ------------------------------------------------------------------
    # Internals — element helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_element(factory: str, name: str) -> Optional[Gst.Element]:
        elem = Gst.ElementFactory.make(factory, name)
        if elem is None:
            logger.error(
                "Failed to create GStreamer element '%s' as '%s'",
                factory,
                name,
            )
        return elem

    def _create_pipeline_elements(
        self, index: int
    ) -> Optional[Dict[str, Gst.Element]]:
        """Create all elements for one receive pipeline."""
        element_specs = (
            ("nvunixfdsrc", "source", f"nvunixfd-source-{index}"),
            ("identity", "identity", f"identity-{index}"),
            ("capsfilter", "caps_nv", f"caps_nv-{index}"),
            ("queue", "queue", f"queue-{index}"),
            ("nvstreammux", "streammux", f"stream-muxer-{index}"),
            ("nvvideoconvert", "nvvidconv", f"nvvideo-converter-{index}"),
            ("capsfilter", "caps_rgba", f"caps_rgba-{index}"),
            ("nvdsosd", "nvosd", f"nv-onscreendisplay-{index}"),
            ("nvvideoconvert", "nvvidconv2", f"nvvideo-converter2-{index}"),
            ("capsfilter", "caps_nv12", f"caps_nv12-{index}"),
            ("nvv4l2h264enc", "enc", f"enc-{index}"),
            ("h264parse", "parse", f"h264parse-{index}"),
            ("rtph264pay", "pay", f"pay-{index}"),
            ("udpsink", "sink", f"sink-{index}"),
        )

        elems: Dict[str, Gst.Element] = {}
        for factory, key, name in element_specs:
            elem = self._create_element(factory, name)
            if elem is None:
                return None
            elems[key] = elem

        return elems

    # ------------------------------------------------------------------
    # Internals — element configuration
    # ------------------------------------------------------------------
    def _configure_source(self, elems: dict, socket_path: str) -> None:
        src = elems["source"]
        src.set_property("socket-path", socket_path)
        src.set_property("buffer-timestamp-copy", True)
        src.set_property("do-timestamp", True)
        src.set_property(
            "meta-deserialization-lib",
            self._config.meta_deserialization_lib,
        )
        logger.info("Socket configured for IPC with path: %s", socket_path)

    @staticmethod
    def _configure_identity(elems: dict) -> None:
        elems["identity"].set_property("single-segment", True)
        elems["identity"].set_property("silent", True)

    @staticmethod
    def _configure_mux(elems: dict) -> None:
        mux = elems["streammux"]
        mux.set_property("batch-size", 1)
        mux.set_property("batched-push-timeout", 33333)
        mux.set_property("attach-sys-ts", False)

    @staticmethod
    def _configure_caps(elems: dict) -> None:
        elems["caps_nv"].set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"),
        )
        elems["caps_rgba"].set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        elems["caps_nv12"].set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"),
        )

    @staticmethod
    def _configure_nvvidconv(elems: dict) -> None:
        elems["nvvidconv"].set_property(
            "nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        )

    @staticmethod
    def _configure_osd(elems: dict) -> None:
        elems["nvosd"].set_property("display-bbox", 0)
        elems["nvosd"].set_property("display-mask", 0)
        elems["nvosd"].set_property("display-text", 0)

    def _configure_encoder(self, elems: dict) -> None:
        enc = elems["enc"]
        enc.set_property("bitrate", self._config.encoder_bitrate)
        for k, v in (
            ("control-rate", 1),
            ("iframeinterval", 30),
            ("idrinterval", 30),
            ("preset-level", 1),
        ):
            try:
                enc.set_property(k, v)
            except Exception:
                pass

    @staticmethod
    def _configure_payloader(elems: dict) -> None:
        elems["parse"].set_property("config-interval", -1)
        elems["pay"].set_property("pt", 96)
        elems["pay"].set_property("mtu", 1400)

    def _configure_sink(self, elems: dict, index: int) -> None:
        sink = elems["sink"]
        sink.set_property("host", "127.0.0.1")
        sink.set_property("port", self._config.udp_base_port + index)
        sink.set_property("sync", True)
        sink.set_property("async", False)
        logger.info(
            "Configured UDP sink to 127.0.0.1:%d",
            self._config.udp_base_port + index,
        )

    # ------------------------------------------------------------------
    # Internals — pipeline linking
    # ------------------------------------------------------------------
    def _link_pipeline(self, elems: dict) -> bool:
        """Link all pipeline elements. Returns True on success."""
        for a, b in zip(self._SOURCE_CHAIN, self._SOURCE_CHAIN[1:]):
            if not elems[a].link(elems[b]):
                logger.error("Failed to link %s to %s", a, b)
                return False

        sinkpad = elems["streammux"].get_request_pad("sink_0")
        if not sinkpad:
            logger.error("Unable to get streammux request pad")
            return False
        srcpad = elems["queue"].get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            logger.error("Failed to link queue to streammux")
            return False

        for a, b in zip(self._OUTPUT_CHAIN, self._OUTPUT_CHAIN[1:]):
            if not elems[a].link(elems[b]):
                logger.error("Failed to link %s to %s", a, b)
                return False

        return True

    # ------------------------------------------------------------------
    # Internals — RTSP mount
    # ------------------------------------------------------------------
    def _setup_rtsp_mount(self, stream_id, index: int) -> None:
        """Register an RTSP mount point for the stream."""
        self._mount_points = self._rtsp_server.get_mount_points()
        factory = GstRtspServer.RTSPMediaFactory()

        launch = (
            f"( udpsrc port={self._config.udp_base_port + index} "
            f"buffer-size=524288 "
            f'caps="application/x-rtp,media=video,clock-rate=90000,'
            f'encoding-name=H264,payload=96" '
            f"! rtpjitterbuffer latency=200 "
            f"! rtph264depay "
            f"! h264parse config-interval=-1 "
            f"! rtph264pay name=pay0 pt=96 )"
        )
        factory.set_launch(launch)
        factory.set_shared(True)

        mount_path = f"/ds-test{stream_id}"
        self._mount_points.add_factory(mount_path, factory)
        logger.info(
            "Added RTSP mount at rtsp://localhost:%s%s",
            self._rtsp_server.props.service,
            mount_path,
        )

    def _teardown_rtsp_mount(self, stream_id) -> None:
        """Remove an RTSP mount point."""
        mount_path = f"/ds-test{stream_id}"
        try:
            self._mount_points.remove_factory(mount_path)
            logger.info("Removed RTSP mount: %s", mount_path)
        except Exception as exc:
            logger.error("Error removing RTSP mount %s: %s", mount_path, exc)

    # ------------------------------------------------------------------
    # Internals — metadata extraction (pyds boilerplate lives HERE)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_classifiers(om) -> List[ClassifierResult]:
        classifiers: List[ClassifierResult] = []
        l_cls = om.classifier_meta_list
        while l_cls is not None:
            try:
                cm = pyds.NvDsClassifierMeta.cast(l_cls.data)
                labels: List[LabelInfo] = []
                l_lbl = cm.label_info_list
                while l_lbl is not None:
                    try:
                        li = pyds.NvDsLabelInfo.cast(l_lbl.data)
                        labels.append(LabelInfo(
                            label_id=li.label_id,
                            result_class_id=li.result_class_id,
                            result_prob=li.result_prob,
                            result_label=li.result_label if li.result_label else "",
                        ))
                        l_lbl = l_lbl.next
                    except StopIteration:
                        break
                classifiers.append(ClassifierResult(
                    unique_component_id=cm.unique_component_id,
                    labels=labels,
                ))
                l_cls = l_cls.next
            except StopIteration:
                break
        return classifiers

    def _extract_objects(self, frame_meta) -> List[ObjectData]:
        """Convert the pyds object linked-list into a clean Python list."""
        objects: List[ObjectData] = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                om = pyds.NvDsObjectMeta.cast(l_obj.data)
                db = om.detector_bbox_info.org_bbox_coords
                tb = om.tracker_bbox_info.org_bbox_coords
                mp = om.mask_params
                objects.append(
                    ObjectData(
                        class_id=om.class_id,
                        confidence=om.confidence,
                        left=int(om.rect_params.left),
                        top=int(om.rect_params.top),
                        width=int(om.rect_params.width),
                        height=int(om.rect_params.height),
                        detector_left=db.left,
                        detector_top=db.top,
                        detector_width=db.width,
                        detector_height=db.height,
                        tracker_left=tb.left,
                        tracker_top=tb.top,
                        tracker_width=tb.width,
                        tracker_height=tb.height,
                        tracker_confidence=om.tracker_confidence,
                        object_id=om.object_id,
                        label=om.obj_label if om.obj_label else "",
                        unique_component_id=om.unique_component_id,
                        mask_width=int(mp.width),
                        mask_height=int(mp.height),
                        mask_threshold=mp.threshold,
                        classifiers=self._extract_classifiers(om),
                        raw_meta=om,
                    )
                )
                l_obj = l_obj.next
            except StopIteration:
                break
        return objects

    # ------------------------------------------------------------------
    # Internals — probes (all pyds iteration stays in the base class)
    # ------------------------------------------------------------------
    def _osd_probe(self, pad, info, u_data):
        """Probe after caps_rgba — extracts clean data, calls _process_frame."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            surface = pyds.get_nvds_buf_surface(
                hash(gst_buffer), frame_meta.batch_id
            )

            frame_data = FrameData(
                frame_num=frame_meta.frame_num,
                source_id=frame_meta.source_id,
                batch_id=frame_meta.batch_id,
                pad_index=frame_meta.pad_index,
                surface=surface,
                buf_pts=frame_meta.buf_pts,
                ntp_timestamp=frame_meta.ntp_timestamp,
                objects=self._extract_objects(frame_meta),
                raw_meta=frame_meta,
            )

            self._process_frame(frame_data)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _src_probe(self, pad, info, u_data):
        """Probe on nvunixfdsrc — verify metadata arrives from socket."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            logger.debug("No batch metadata found")
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        if not l_frame:
            logger.debug(
                "Batch metadata exists but frame_meta_list is NULL"
            )
            return Gst.PadProbeReturn.OK

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                num_objects = 0
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    num_objects += 1
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break

                if num_objects > 0:
                    logger.info(
                        "Frame#%d: %d objects received from socket",
                        frame_meta.frame_num,
                        num_objects,
                    )
                elif frame_meta.frame_num % 30 == 0:
                    logger.info(
                        "Frame#%d: 0 objects (no metadata)",
                        frame_meta.frame_num,
                    )
            except StopIteration:
                break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    # ------------------------------------------------------------------
    # Internals — drawing utilities (available to subclasses)
    # ------------------------------------------------------------------
    def _draw_bounding_box(self, surface, left, top, width, height, color=(0, 255, 0)):
        """Draw a single bounding box on the frame."""
        cv2.rectangle(
            surface,
            (left, top),
            (left + width, top + height),
            color,
            2,
        )

    def _draw_label(self, surface, obj: ObjectData, color=(0, 255, 0)) -> None:
        """Draw tracking ID, label, and confidence above the bounding box."""
        # Build label text: "ID:42 Helmet 0.93"
        parts = []
        if obj.object_id > 0:
            parts.append(f"ID:{obj.object_id}")
        if obj.label:
            parts.append(obj.label)
        if obj.confidence > 0:
            parts.append(f"{obj.confidence:.2f}")

        label_text = " ".join(parts)
        if not label_text:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )

        # Background rectangle behind the text
        x1 = obj.left
        y1 = max(obj.top - th - baseline - 4, 0)
        x2 = obj.left + tw + 4
        y2 = obj.top

        cv2.rectangle(surface, (x1, y1), (x2, y2), color, cv2.FILLED)

        # Text color: white on dark backgrounds, black on bright
        brightness = color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        cv2.putText(
            surface,
            label_text,
            (x1 + 2, y2 - baseline - 1),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _class_color(class_id: int) -> tuple:
        """Return a consistent color for a given class_id."""
        palette = (
            (0, 255, 0),    # green
            (255, 128, 0),  # orange
            (0, 128, 255),  # blue
            (255, 0, 128),  # pink
            (128, 255, 0),  # lime
            (0, 255, 255),  # cyan
            (255, 255, 0),  # yellow
            (128, 0, 255),  # purple
        )
        return palette[class_id % len(palette)]

    def _draw_mask(self, surface, obj: ObjectData) -> None:
        """Draw segmentation mask overlay.  Requires ``obj.raw_meta``."""
        if obj.raw_meta is None:
            return
        try:
            mp = obj.raw_meta.mask_params
            if int(mp.width) <= 0 or int(mp.height) <= 0:
                return

            mask = mp.get_mask_array()
            if mask is None or mask.size == 0:
                return

            mask = mask.reshape((int(mp.height), int(mp.width)))

            frame_h, frame_w = surface.shape[:2]
            x1 = max(0, obj.left)
            y1 = max(0, obj.top)
            x2 = min(frame_w, obj.left + obj.width)
            y2 = min(frame_h, obj.top + obj.height)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                return

            # Resize to bbox and apply configured threshold
            mask_resized = cv2.resize(
                mask, (bw, bh), interpolation=cv2.INTER_LINEAR
            )
            binary = mask_resized >= mp.threshold  # bool (bh, bw)

            # Blend per-class color into the bbox roi where mask is active
            color = self._class_color(obj.class_id)   # (R, G, B)
            roi = surface[y1:y2, x1:x2]               # view, shape (bh, bw, 4)
            mask_3d = binary[:, :, np.newaxis]         # (bh, bw, 1)

            overlay = np.zeros((bh, bw, 4), dtype=np.uint8)
            overlay[binary] = (color[0], color[1], color[2], 255)

            roi[:] = np.where(
                mask_3d,
                (roi.astype(np.float32) * 0.55 +
                 overlay.astype(np.float32) * 0.45).astype(np.uint8),
                roi,
            )
        except Exception as exc:
            logger.debug("Error drawing mask: %s", exc)

    # ------------------------------------------------------------------
    # Internals — bus messages
    # ------------------------------------------------------------------
    def _on_bus_message(self, bus, message, user_data):
        """Handle GStreamer bus messages."""
        socket_path = user_data
        mtype = message.type

        if mtype == Gst.MessageType.EOS:
            logger.info("End-of-stream for %s", socket_path)
            GLib.idle_add(self._remove_stream, socket_path)
        elif mtype == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("Warning (%s): %s: %s", socket_path, err, debug)
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("Error (%s): %s: %s", socket_path, err, debug)
            GLib.idle_add(self._remove_stream, socket_path)

        return True

    # ------------------------------------------------------------------
    # Internals — stream add / remove (main thread via GLib.idle_add)
    # ------------------------------------------------------------------
    def _add_stream(self, socket_path: str) -> bool:
        """Add a new stream for a socket (called from main thread)."""
        with self._lock:
            if socket_path in self._streams:
                logger.info("Stream already exists for: %s", socket_path)
                return False

            index = self._stream_index_counter
            self._stream_index_counter += 1
            uuid = self._parse_uuid(socket_path)

        logger.info("Adding stream %d for socket: %s", index, socket_path)

        pipeline = self.create_pipeline(
            socket_path, stream_id=uuid, index=index
        )
        if not pipeline:
            logger.error("Failed to create pipeline for: %s", socket_path)
            return False

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, socket_path)

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error(
                "Unable to set pipeline to PLAYING for: %s", socket_path
            )
            pipeline.set_state(Gst.State.NULL)
            return False

        record = StreamRecord(
            socket_path=socket_path,
            uuid=uuid,
            index=index,
            pipeline=pipeline,
        )

        with self._lock:
            self._streams[socket_path] = record

        logger.info("Pipeline %d is PLAYING for: %s", index, socket_path)
        self._on_stream_added(record)
        return False  # Remove from GLib idle queue

    def _remove_stream(self, socket_path: str) -> bool:
        """Remove a stream (called from main thread via GLib.idle_add)."""
        with self._lock:
            record = self._streams.pop(socket_path, None)
            if record is None:
                return False

        logger.info(
            "Removing stream %d for socket: %s",
            record.index,
            socket_path,
        )

        record.pipeline.set_state(Gst.State.NULL)
        self._teardown_rtsp_mount(record.index)
        os.remove(socket_path)

        with self._stale_lock:
            self._stale_sockets.pop(socket_path, None)

        with self._check_lock:
            self._checked_active.discard(socket_path)
            self._checking.discard(socket_path)

        logger.info("Stream %d stopped", record.index)
        self._on_stream_removed(record)
        return False  # Remove from GLib idle queue

    # ------------------------------------------------------------------
    # Internals — socket monitoring
    # ------------------------------------------------------------------
    @staticmethod
    def _is_socket_file(path: str) -> bool:
        """Check if a path is a Unix socket file."""
        try:
            return stat.S_ISSOCK(os.stat(path).st_mode)
        except (OSError, FileNotFoundError):
            return False

    def _is_socket_active(self, socket_path: str) -> bool:
        """Check if a Unix socket has an active server listening."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self._config.socket_connect_timeout)
            result = sock.connect_ex(socket_path)
            sock.close()
            return result == 0
        except socket.timeout:
            logger.info("Socket connection timeout: %s", socket_path)
            return False
        except Exception as exc:
            logger.info(
                "Socket check error for %s: %s", socket_path, exc
            )
            return False

    def _confirm_active(self, socket_path: str) -> str:
        """Probe a socket for a live server at most once, caching the result.

        Returns one of:
            ``"active"``   — a server is (or was already confirmed) listening;
            ``"inactive"`` — the probe connected to nothing (stale);
            ``"pending"``  — another scanner is probing it right now, skip.

        The single-probe guarantee is what stops the initial scan and the
        watcher from both connecting to a fresh socket (each throwaway connect
        triggers a "Broken pipe" on the server's nvunixfdsink).
        """
        with self._check_lock:
            if socket_path in self._checked_active:
                return "active"
            if socket_path in self._checking:
                return "pending"
            self._checking.add(socket_path)
        active = False
        try:
            active = self._is_socket_active(socket_path)
        finally:
            with self._check_lock:
                self._checking.discard(socket_path)
                if active:
                    self._checked_active.add(socket_path)
        return "active" if active else "inactive"

    def _should_check_socket(self, socket_path: str) -> bool:
        """Check if we should attempt to connect (not in cooldown)."""
        with self._stale_lock:
            if socket_path not in self._stale_sockets:
                return True

            elapsed = time.time() - self._stale_sockets[socket_path]
            if elapsed >= self._config.stale_retry_interval:
                del self._stale_sockets[socket_path]
                return True

            return False

    def _mark_socket_stale(self, socket_path: str) -> None:
        """Mark a socket as stale (failed connection check)."""
        with self._stale_lock:
            self._stale_sockets[socket_path] = time.time()
            logger.info(
                "Marked socket as stale (will retry in %.0fs): %s",
                self._config.stale_retry_interval,
                socket_path,
            )

    def _clear_stale_mark(self, socket_path: str) -> None:
        """Clear stale mark for a socket that became active."""
        with self._stale_lock:
            self._stale_sockets.pop(socket_path, None)

    def _scan_directory(self) -> list:
        """Scan the watch directory for socket files."""
        sockets = []
        try:
            if not os.path.exists(self._config.watch_dir):
                return sockets

            for entry in os.listdir(self._config.watch_dir):
                full_path = os.path.join(self._config.watch_dir, entry)
                if self._is_socket_file(full_path):
                    sockets.append(full_path)
        except Exception as exc:
            logger.error(
                "Error scanning directory %s: %s",
                self._config.watch_dir,
                exc,
            )

        return sorted(sockets)

    # ------------------------------------------------------------------
    # Internals — directory watcher
    # ------------------------------------------------------------------
    def _start_watcher(self) -> None:
        """Launch the background directory watcher thread."""
        self._watcher_running = True
        self._watcher_thread = threading.Thread(
            target=self._watcher_loop, daemon=True
        )
        self._watcher_thread.start()

    def _watcher_loop(self) -> None:
        """Background thread that polls for new/removed sockets."""
        logger.info(
            "Directory watcher started for: %s", self._config.watch_dir
        )

        while self._watcher_running:
            try:
                current_sockets = set(self._scan_directory())

                with self._lock:
                    known_sockets = set(self._streams.keys())

                for socket_path in current_sockets - known_sockets:
                    if not self._should_check_socket(socket_path):
                        continue

                    status = self._confirm_active(socket_path)
                    if status == "pending":
                        continue  # another scanner is probing it — handle next round
                    if status == "active":
                        logger.info("Socket is ACTIVE: %s", socket_path)
                        self._clear_stale_mark(socket_path)
                        GLib.idle_add(self._add_stream, socket_path)
                    else:
                        logger.info(
                            "Socket is STALE (no server listening): %s",
                            socket_path,
                        )
                        self._mark_socket_stale(socket_path)

                for socket_path in known_sockets - current_sockets:
                    logger.info("Socket removed: %s", socket_path)
                    GLib.idle_add(self._remove_stream, socket_path)
                    self._clear_stale_mark(socket_path)

            except Exception as exc:
                logger.error("Watcher error: %s", exc)

            time.sleep(self._config.watch_interval)

        logger.info("Directory watcher stopped")

    def _initial_scan(self) -> None:
        """Perform the first scan and connect to active sockets."""
        logger.info(
            "Performing initial scan of %s...", self._config.watch_dir
        )
        initial_sockets = self._scan_directory()

        if not initial_sockets:
            logger.info(
                "No sockets found yet, waiting for new connections..."
            )
            return

        logger.info(
            "Found %d socket file(s), checking which are active...",
            len(initial_sockets),
        )
        active_count = 0

        for socket_path in initial_sockets:
            logger.info("Checking socket: %s", socket_path)
            status = self._confirm_active(socket_path)
            if status == "pending":
                continue  # watcher is already probing it — it will be added there
            if status == "active":
                logger.info("Socket is ACTIVE: %s", socket_path)
                self._add_stream(socket_path)
                active_count += 1
            else:
                logger.info("Socket is STALE (skipping): %s", socket_path)
                self._mark_socket_stale(socket_path)

        if active_count == 0:
            logger.info(
                "No active sockets found, waiting for new connections..."
            )
        else:
            logger.info("Connected to %d active socket(s)", active_count)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_uuid(socket_path: str) -> str:
        """Extract UUID from socket filename (expects <uuid>.sock)."""
        return os.path.basename(socket_path).split(".")[0]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main():
    """Main entry point for standalone client usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = DeepStreamClient(rtsp_port="8554")
    client._config.watch_interval = 2.0
    return client.start(wait_for_sockets=True)


if __name__ == "__main__":
    sys.exit(main())
