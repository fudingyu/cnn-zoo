"""用 amd-quark 把候选 CNN 模型量化成 XINT8（可上 NPU 的格式）。"""
import os, sys, time, random, json
import numpy as np
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import paths as P

ROOT = Path(__file__).resolve().parent
FP32 = ROOT / "fp32"
INT8 = ROOT / "int8"
INT8.mkdir(exist_ok=True)

# ---------- 素材 ----------
# Bundled samples first; any extra calibration dirs you drop under the SDK
# root are picked up automatically (see paths.py for where that root is).
SRC_DIRS = [
    ROOT / "testimg",
    P.EXT / "yolo_official" / "calib_images",
    P.EXT / "realsrgan256" / "assets",
    P.EXT / "winml_resnet" / "images",
]
IMAGES = []
for d in SRC_DIRS:
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.stat().st_size > 5000:
                IMAGES.append(p)
print(f"[calib] source images: {len(IMAGES)}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_variants(size, n):
    """从本地素材生成 n 个 (3,size,size) float32 RGB 变体（随机裁切/翻转/缩放）。"""
    rnd = random.Random(1234)
    out = []
    if not IMAGES:
        raise RuntimeError("no calibration images")
    for i in range(n):
        p = IMAGES[i % len(IMAGES)]
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        # 随机 crop 一个方形区域再 resize，模拟不同尺度
        s = int(min(w, h) * rnd.uniform(0.6, 1.0))
        x = rnd.randint(0, max(0, w - s))
        y = rnd.randint(0, max(0, h - s))
        im = im.crop((x, y, x + s, y + s)).resize((size, size), Image.BILINEAR)
        if rnd.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        a = np.asarray(im, dtype=np.float32) / 255.0
        out.append(a)
    return out


def pre_seg(size, n):
    xs = _load_variants(size, n)
    return [((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32) for x in xs]


def pre_unit(size, n):
    """输入归一化到 [-1,1]（AnimeGAN/MODNet 风格）"""
    xs = _load_variants(size, n)
    return [(x * 2.0 - 1.0).transpose(2, 0, 1)[None].astype(np.float32) for x in xs]


class ListReader:
    def __init__(self, name, samples):
        self.name, self.samples, self.i = name, samples, 0

    def get_next(self):
        if self.i >= len(self.samples):
            return None
        v = self.samples[self.i]
        self.i += 1
        return {self.name: v}

    def rewind(self):
        self.i = 0


MODELS = [
    # (nick, fp32 rel path, input_name, size, preprocess, 校准样本数)
    ("lraspp_seg", "lraspp_mobilenet_v3_large_512.onnx", "input", 512, pre_seg, 16),
    ("deeplabv3_seg", "deeplabv3_mobilenet_v3_large_512.onnx", "input", 512, pre_seg, 16),
    ("modnet_matting", "modnet_fp32_static.onnx", "input", 512, pre_unit, 16),
    ("animeganv2", "animeganv2_fp32_static.onnx", "input_image", 512, pre_unit, 16),
]

only = sys.argv[1:] if len(sys.argv) > 1 else None

for nick, fname, iname, size, pre_fn, n in MODELS:
    if only and nick not in only:
        continue
    src = FP32 / fname
    dst = INT8 / f"{nick}_XINT8.onnx"
    if not src.exists():
        print(f"[skip] {nick}: {src} missing")
        continue
    print(f"\n===== {nick} =====  src={src.name}")
    t0 = time.time()
    try:
        samples = pre_fn(size, n)
        reader = ListReader(iname, samples)
        from quark.onnx import ModelQuantizer
        from quark.onnx.quantization.config import Config, get_default_config

        quant_config = get_default_config("XINT8")
        # 默认 MinMSE 阈值搜索在小数据上极慢（>1h），改用 NonOverflow（power-of-two）快速标定
        try:
            from quark.onnx.calibration.methods import PowerOfTwoMethod
            quant_config.calibrate_method = PowerOfTwoMethod.NonOverflow
        except Exception:
            quant_config.calibrate_method = 0
        try:
            quant_config.enable_npu_cnn = True
        except Exception:
            pass
        config = Config(global_quant_config=quant_config)
        quantizer = ModelQuantizer(config)
        quantizer.quantize_model(str(src), str(dst), reader)
        dt = time.time() - t0
        ok = dst.exists() and dst.stat().st_size > 1000
        print(f"[done] {nick} -> {dst.name}  {dst.stat().st_size/1e6:.1f}MB  {dt:.1f}s  ok={ok}")
    except Exception as e:
        import traceback
        print(f"[FAIL] {nick}: {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc(limit=3)
print("\nALL DONE")
