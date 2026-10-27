"""minMSE 重新标定 DeepLabV3 与 MODNet 的对比脚本（后台跑）。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from quantize_all import ListReader, pre_seg, ROOT
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

src = ROOT / "fp32" / "deeplabv3_mobilenet_v3_large_512.onnx"
dst = ROOT / "int8" / "deeplabv3_seg_XINT8_MSE.onnx"
reader = ListReader("input", pre_seg(512, 16))
qc = get_default_config("XINT8")
t0 = time.time()
ModelQuantizer(Config(global_quant_config=qc)).quantize_model(str(src), str(dst), reader)
print("done", dst, round(dst.stat().st_size / 1e6, 2), "MB", round(time.time() - t0, 1), "s")
