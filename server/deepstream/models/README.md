# Models

This directory holds the TensorRT-ready ONNX models the inference pipeline loads.
On first run, `build_engines.py` reads each active `GIE_*_CONFIG` and builds a
TensorRT engine (`*.engine`) next to the ONNX. Engines are GPU/TensorRT-specific
and are **not** committed — they are generated locally.

## Expected files

Each PGIE config points at a specific ONNX filename. Drop the matching model here:

| Task | Config (`server/deepstream/config/`) | Expected ONNX | Parser | Labels |
|------|--------------------------------------|---------------|--------|--------|
| YOLO detection | `config_pgie_yolo_detct.txt` | `yolo11l_bbox_v8-trt.onnx` | `nvdsinfer_yolo_det.so` | `labels_det.txt` |
| YOLO segmentation | `config_pgie_yolo_seg.txt` | `conv_seg_v6-trt.onnx` | `nvdsinfer_yolo_seg.so` | `yolo11l-seg_labels.txt` |
| RT-DETR | `config_pgie_rtdetr_l.txt` | `rtdetr-l-det-template.onnx` | `nvdsinfer_rtdetr.so` | `labels_det.txt` |

The ONNX must expose output layers that match the parser it is paired with. If you
change the filename, update the `onnx-file` / `model-engine-file` lines in the
corresponding config.

## Re-ID model (NvDeepSORT only)

`DS_TRACKER=NvDeepSORT` additionally needs `resnet50_market1501.etlt`, a TAO-encoded
Re-ID model published by NVIDIA. It is **not bundled**; download it from NGC as
documented in [`docs/guides/tracking-implementation.md`](../../../docs/guides/tracking-implementation.md).
The default tracker (`NvDCF`) does not require it.

## Licensing

The sample models distributed with Osprey carry their own upstream licenses, separate
from Osprey's Apache-2.0 code. See [`NOTICE`](../../../NOTICE) before redistributing or
using them commercially.
