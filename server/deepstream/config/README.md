DeepStream Configs
===================

This folder contains configuration and label files for the DeepStream primary inference engines.

Files:

- `config_pgie_yolo_detct.txt`  
  - YOLO-based **detection** (bounding box) primary inference configuration.  
  - Points to the detection ONNX/TensorRT engine and uses `labels_det.txt` for class names.  
  - Key settings include GPU ID, batch size, number of detected classes, and NMS thresholds.

- `config_pgie_yolo_seg.txt`  
  - YOLO-based **instance segmentation** primary inference configuration.  
  - Points to the segmentation ONNX/TensorRT engine and uses `labels_seg.txt` for class names.  
  - Key settings include GPU ID, batch size, input dimensions, number of detected classes, and segmentation threshold.

- `labels_det.txt`  
  - Label file for the detection model (`config_pgie_yolo_detct.txt`).

- `labels_seg.txt`  
  - Label file for the segmentation model (`config_pgie_yolo_seg.txt`).

---

Parameter reference
-------------------

### Common `[property]` section parameters

- `gpu-id`  
  - Index of the GPU on which this inference engine will run.

- `net-scale-factor`  
  - Value used to scale raw pixel intensities before feeding them to the network (e.g. `1/255` ≈ `0.0039215`).

- `model-color-format`  
  - Input color format expected by the model.  
  - `0` usually means BGR, `1` usually means RGB (depending on plugin implementation).

- `onnx-file`  
  - Path to the ONNX model file used to build or load the TensorRT engine.

- `model-engine-file`  
  - Path to the serialized TensorRT engine file. If it exists, DeepStream loads it directly; otherwise it can be generated from the ONNX.

- `labelfile-path`  
  - Path to the label file mapping class IDs to human‑readable class names.

- `batch-size`  
  - Number of frames/images processed together in a single inference batch.

- `network-mode`  
  - Precision mode for inference.  
  - Typical values: `0` = FP32, `1` = INT8, `2` = FP16.

- `num-detected-classes`  
  - Number of classes the model is trained to predict (must match the labels file and model).

- `interval`  
  - Frame-skip interval. `0` = run inference on every frame; `1` = every other frame; `N` = every (N+1)-th frame.

- `gie-unique-id`  
  - Unique ID for this inference engine within the DeepStream pipeline (used for linking metadata between elements).

- `process-mode`  
  - Operating mode of the GIE element.  
  - `1` commonly means primary inference engine (PGIE).

- `network-type`  
  - Type of task the network performs. Typical mapping:  
  - `0` = Detector, `1` = Classifier, `2` = Segmentation, `3` = Instance Segmentation.

- `cluster-mode`  
  - Post‑processing clustering algorithm for bounding boxes or instances. Typical mapping:  
  - `0` = Group Rectangle, `1` = DBSCAN, `2` = NMS, `3` = DBSCAN+NMS, `4` = None.

- `maintain-aspect-ratio`  
  - If `1`, adds padding instead of stretching the image so the original aspect ratio is preserved.

- `symmetric-padding`  
  - If `1`, applies padding symmetrically on both sides when maintaining aspect ratio.

### Detection config specific (`config_pgie_yolo_detct.txt`)

- `parse-bbox-func-name`  
  - Name of the custom YOLO bounding box parsing function used by the DeepStream inference plugin.

- `custom-lib-path`  
  - Path to the shared library providing the custom YOLO parsing and engine creation functions.

- `engine-create-func-name`  
  - Name of the function inside the custom library that builds or retrieves the TensorRT engine.

#### `[class-attrs-all]` section (detection)

- `nms-iou-threshold`  
  - IoU threshold used in Non‑Maximum Suppression (NMS) to remove overlapping boxes.

- `pre-cluster-threshold`  
  - Confidence score threshold before clustering/NMS; boxes below this threshold are discarded.

- `topk`  
  - Maximum number of boxes kept after NMS per frame.

### Segmentation config specific (`config_pgie_yolo_seg.txt`)

- `force-implicit-batch-dim`  
  - Controls whether TensorRT uses implicit batch dimension. `0` usually means explicit batch (recommended for newer TensorRT versions).

- `infer-dims`  
  - Network input tensor dimensions in the form `C;H;W` (channels, height, width).

- `parse-bbox-instance-mask-func-name`  
  - Name of the custom function that parses both bounding boxes and instance masks from the YOLO segmentation outputs.

- `output-instance-mask`  
  - If `1`, instructs the plugin to output instance masks along with bounding boxes.

- `segmentation-threshold`  
  - Confidence threshold used to keep or discard segmentation masks.

- `pre-cluster-threshold` (in `[class-attrs-all]`)  
  - Confidence threshold for candidate instances before clustering/post‑processing for the segmentation model.

