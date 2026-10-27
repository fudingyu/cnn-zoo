"""MODNet 用 MinMSE 重新标定（更准但慢），输出 int8/modnet_matting_XINT8_MSE.onnx"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pathlib import Path
from quantize_all import ListReader, pre_unit, ROOT
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

src = ROOT / "fp32" / "modnet_fp32_static.onnx"
dst = ROOT / "int8" / "modnet_matting_XINT8_MSE.onnx"
samples = pre_unit(512, 16)
reader = ListReader("input", samples)
qc = get_default_config("XINT8")
config = Config(global_quant_config=qc)
t0 = time.time()
ModelQuantizer(config).quantize_model(str(src), str(dst), reader)
print("done", dst, dst.stat().st_size / 1e6, "MB", round(time.time() - t0, 1), "s")
