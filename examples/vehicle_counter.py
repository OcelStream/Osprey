#!/usr/bin/env python3
"""Single-file Osprey app: configure + start the server, then run analytics.

The server (DeepStream inference + FastAPI control plane) runs in its own
forked process; this process runs the client, draws on each frame, and serves
RTSP. Run it, then open  rtsp://<host>:8554/ds-testcam1  in any RTSP player.

    python3 examples/vehicle_counter.py
"""

import osprey.server as osprey

# 1. Configure the model/tracker, then start the server (own process).
osprey.configure(
    gie_config="/run/model/gie.txt",   # your nvinfer config (model + parser)
    tracker="NvSORT",
    model_width=640,
    model_height=640,
)
osprey.serve()                          # returns once the server is healthy

# 2. Add a source over the control plane (a file here; rtsp:// also works).
osprey.add_stream(
    "file:///assets/static/parking-2.mp4",
    stream_id="cam1",
    rtsp_output_width=640,
    rtsp_output_height=640,
)
osprey.add_stream(
    "file:///assets/static/parking-2.mp4",
    stream_id="cam1",
    rtsp_output_width=640,
    rtsp_output_height=640,
)

# 3. Write your analytics with the client — no GStreamer/pyds knowledge needed.
from osprey.client import DeepStreamClient, FrameData


class VehicleCounter(DeepStreamClient):
    def _process_frame(self, frame: FrameData) -> None:
        for obj in frame.objects:
            self._draw_object(frame.surface, obj)   # box + tracking label


if __name__ == "__main__":
    VehicleCounter().start()            # serves RTSP for every discovered stream
