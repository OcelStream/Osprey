#!/usr/bin/env python3
"""Generate a ready-to-run GIE (nvinfer) config with the shipped parser resolved.

The parser `.so` ships inside the installed ospreyai package, so its absolute
path differs per machine. This script fills it in for you.

    python3 make_gie_config.py \
        --onnx   /assets/models/yolo26m-trt.onnx \
        --labels /assets/config/yolo26m-trt.txt \
        --out    /run/model/gie.txt

    # then:  osprey.configure(gie_config="/run/model/gie.txt", ...)

Use --task seg or --task rtdetr for the other shipped parsers.
"""
import argparse
import os

import osprey

_PARSERS = {
    "det":    ("nvdsinfer_yolo_det.so", "NvDsInferParseCustomYoloDet"),
    "seg":    ("nvdsinfer_yolo_seg.so", "NvDsInferParseCustomYoloSeg"),
    "rtdetr": ("nvdsinfer_rtdetr.so",   "NvDsInferParseCustomRTDetr"),
}


def parser_path(so_name: str) -> str:
    p = os.path.join(
        os.path.dirname(osprey.__file__), "server", "deepstream", "lib", so_name
    )
    if not os.path.exists(p):
        raise SystemExit(f"shipped parser not found: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="path to the .onnx model")
    ap.add_argument("--engine", help="prebuilt .engine (defaults to <onnx>.engine)")
    ap.add_argument("--labels", required=True, help="path to the labels .txt")
    ap.add_argument("--task", choices=_PARSERS, default="det")
    ap.add_argument("--classes", type=int, default=80)
    ap.add_argument("--size", type=int, default=640, help="square infer size")
    ap.add_argument("--out", default="/run/model/gie.txt")
    args = ap.parse_args()

    so_name, func = _PARSERS[args.task]
    lib = parser_path(so_name)
    engine = args.engine or (args.onnx + ".engine")

    cfg = f"""[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file={args.onnx}
model-engine-file={engine}
labelfile-path={args.labels}
batch-size=1
network-mode=2
num-detected-classes={args.classes}
infer-dims=3;{args.size};{args.size}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=4
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name={func}
custom-lib-path={lib}
[class-attrs-all]
threshold=0.25
topk=100
"""

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(cfg)
    print(f"wrote {args.out}")
    print(f"  parser : {lib}")
    print(f"  onnx   : {args.onnx}")
    print(f"  engine : {engine}")
    print(f"  labels : {args.labels}")


if __name__ == "__main__":
    main()
