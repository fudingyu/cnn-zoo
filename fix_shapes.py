"""打印关键模型的输入/输出 shape 与 dtype。

排查「NPU 不认动态维度」「输入名对不上」这类问题时第一个跑它。

    python fix_shapes.py
"""
import os

import onnx

import paths as P

TARGETS = [
    P.sdk("CNN-examples", "torchvision_inference", "models", "resnet50_XINT8.onnx"),
    P.sdk("CNN-examples", "object_detection", "yolov8m", "models", "yolov8m_XINT8_skip.onnx"),
    P.ext("realsrgan256", "onnx-models", "realesrgan_nchw_256x256_u8s8.onnx"),
]


def show(tag, value_info):
    for v in value_info:
        dims = [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
        print("   %-4s %-40s %s  dtype=%d" % (tag, v.name, dims, v.type.tensor_type.elem_type))


for p in TARGETS:
    name = os.path.relpath(p, str(P.EXT))
    if not os.path.exists(p):
        print("--- %s  [missing]" % name)
        continue
    try:
        m = onnx.load(p, load_external_data=False)
    except Exception as e:
        print("--- %s  ERR %s" % (name, str(e)[:80]))
        continue
    print("--- %s" % name)
    show("IN", m.graph.input)
    show("OUT", m.graph.output)
