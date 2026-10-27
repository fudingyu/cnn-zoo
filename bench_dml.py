"""iGPU(780M, DirectML) 对照：FP32 模型在 DML EP 上的延迟。"""
import os, sys, json, time, ctypes
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import paths as P

ROOT = Path(__file__).resolve().parent
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve(p):
    """Repo-relative paths stay relative; SDK/third-party ones arrive absolute."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def make_input(kind, shape):
    rnd = np.random.RandomState(7)
    nchw = shape[-1] != 3
    if nchw:
        c, h, w = shape[1], shape[2], shape[3]
    else:
        h, w, c = shape[1], shape[2], shape[3]
    x = rnd.rand(h, w, c).astype(np.float32) * 0.6 + 0.2
    if kind == "seg":
        x = (x - MEAN) / STD
    else:
        x = x * 2.0 - 1.0
    if nchw:
        x = x.transpose(2, 0, 1)
    return x[None].astype(np.float32)


_SW = P.sdk("CNN-examples")
JOBS = [
    dict(key="lraspp_seg", fp32="fp32/lraspp_mobilenet_v3_large_512.onnx", in_name="input", shape=[1, 3, 512, 512], kind="seg"),
    dict(key="deeplabv3_seg", fp32="fp32/deeplabv3_mobilenet_v3_large_512.onnx", in_name="input", shape=[1, 3, 512, 512], kind="seg"),
    dict(key="modnet_matting", fp32="fp32/modnet_fp32_static.onnx", in_name="input", shape=[1, 3, 512, 512], kind="unit"),
    dict(key="animeganv2", fp32="fp32/animeganv2_fp32_static.onnx", in_name="input_image", shape=[1, 3, 512, 512], kind="unit"),
    dict(key="realesrgan", fp32=P.ext("realsrgan256", "onnx-models", "realesrgan_nchw_256x256_fp32.onnx"),
         in_name="input", shape=[1, 3, 256, 256], kind="unit"),
    dict(key="resnet50", fp32=os.path.join(_SW, "torchvision_inference", "models", "resnet50.onnx"),
         in_name="input", shape=[1, 3, 224, 224], kind="seg"),
    dict(key="yolov8m", fp32=os.path.join(_SW, "object_detection", "yolov8m", "models", "yolov8m.onnx"),
         in_name="images", shape=[1, 3, 640, 640], kind="seg"),
]

results = []
for j in JOBS:
    p = resolve(j["fp32"])
    if not p.exists():
        continue
    feed = make_input(j["kind"], j["shape"])
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        s = ort.InferenceSession(str(p), sess_options=so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        for _ in range(3):
            s.run(None, {j["in_name"]: feed})
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            s.run(None, {j["in_name"]: feed})
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        rec = dict(j, dml_ms=round(float(ts.mean()), 2))
        print("[%s] dml_fp32=%.2f ms" % (j["key"], rec["dml_ms"]))
        results.append(rec)
    except Exception as e:
        print("[%s] DML ERR %s: %s" % (j["key"], type(e).__name__, str(e)[:150]))
        results.append(dict(j, dml_err=str(e)[:150]))

(ROOT / "result" / "bench_dml.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("saved")
