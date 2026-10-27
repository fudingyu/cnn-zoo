"""对候选模型做真实图片推理，输出可视化结果 + INT8 量化精度影响。
结果写入 result/ 目录（png + metrics.json）
"""
import os, sys, io, json, time, ctypes
from pathlib import Path

try:
    ctypes.windll.kernel32.SetThreadUILanguage(0x0409)
except Exception:
    pass
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import onnxruntime as ort

import paths as P
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "testimg"
RES = ROOT / "result"
RES.mkdir(exist_ok=True)

INSTALL = r"C:\Program Files\RyzenAI\1.7.1"
VOE = os.path.join(INSTALL, "voe-4.0-win_amd64")
XCLBIN = os.path.join(VOE, "xclbins", "phoenix", "4x4.xclbin")
CACHE_DIR = str(ROOT / ".vitis_cache")
os.environ["RYZEN_AI_INSTALLATION_PATH"] = INSTALL
os.makedirs(CACHE_DIR, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
       "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
       "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def palette(n=256):
    def bitget(b, i):
        return (b >> i) & 1
    p = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        for j in range(8):
            r = r | (bitget(i, 0) << (7 - j))
            g = g | (bitget(i, 1) << (7 - j))
            b = b | (bitget(i, 2) << (7 - j))
            r = (r << 0) | 0
        p[i] = [bitget(i, 0) * 128 + bitget(i, 3) * 64 + bitget(i, 6) * 32,
                bitget(i, 1) * 128 + bitget(i, 4) * 64 + bitget(i, 7) * 32,
                bitget(i, 2) * 128 + bitget(i, 5) * 64 + bitget(i, 8) * 32]
    return p


PAL = palette()


def load_img(path, size=512):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0, im


def prep(x, kind):
    if kind == "seg":
        v = (x - MEAN) / STD
    else:
        v = x * 2.0 - 1.0
    return v.transpose(2, 0, 1)[None].astype(np.float32)


def sess_npu(path, key):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 1
    opts = {"target": "X1", "xlnx_enable_py3_round": 0, "xclbin": XCLBIN,
            "cache_dir": CACHE_DIR, "cache_key": "zoo_%s_4x4" % key,
            "enable_cache_file_io_in_mem": "0"}
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=[("VitisAIExecutionProvider", opts), "CPUExecutionProvider"])


def sess_cpu(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def save(arr_u8, name):
    Image.fromarray(arr_u8).save(str(RES / name))


def chessboard(h, w, tile=32):
    y, x = np.mgrid[0:h, 0:w]
    c = ((x // tile) + (y // tile)) % 2
    a = (c * 40 + 190).astype(np.uint8)
    b = (c * -40 + 235).astype(np.uint8)
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = a, b, b
    return img


TESTS = [
    dict(key="lraspp_seg", kind="seg", in_name="input",
         fp32="fp32/lraspp_mobilenet_v3_large_512.onnx", int8="int8/lraspp_seg_XINT8.onnx",
         img="testimg/city.jpg"),
    dict(key="deeplabv3_seg", kind="seg", in_name="input",
         fp32="fp32/deeplabv3_mobilenet_v3_large_512.onnx", int8="int8/deeplabv3_seg_XINT8.onnx",
         img="testimg/city.jpg"),
    dict(key="modnet_matting", kind="unit", in_name="input",
         fp32="fp32/modnet_fp32_static.onnx", int8="int8/modnet_matting_XINT8.onnx",
         img=str(BASE / "face_test.jpg")),
    dict(key="animeganv2", kind="unit", in_name="input_image",
         fp32="fp32/animeganv2_fp32_static.onnx", int8="int8/animeganv2_XINT8.onnx",
         img=str(BASE / "face_test.jpg")),
]

only = sys.argv[1:] or None
metrics = {}
for t in TESTS:
    if only and t["key"] not in only:
        continue
    f32, i8 = ROOT / t["fp32"], ROOT / t["int8"]
    if not i8.exists() or i8.stat().st_size < 1000:
        print("[skip]", t["key"])
        continue
    x, im = load_img(t["img"], 512)
    feed = prep(x, t["kind"])
    save((x * 255).clip(0, 255).astype(np.uint8), "%s_input.png" % t["key"])

    rec = {}
    # fp32 CPU 参考
    try:
        s = sess_cpu(f32)
        o32 = s.run(None, {t["in_name"]: feed})[0]
        rec["fp32_out_shape"] = list(o32.shape)
    except Exception as e:
        print("fp32 fail", t["key"], str(e)[:120]); o32 = None
    # int8 NPU
    try:
        s2 = sess_npu(i8, t["key"])
        for _ in range(2):
            o8 = s2.run(None, {t["in_name"]: feed})[0]
        rec["int8_out_shape"] = list(o8.shape)
    except Exception as e:
        print("npu fail", t["key"], str(e)[:120]); o8 = None

    if o32 is None or o8 is None:
        metrics[t["key"]] = rec
        continue

    if t["kind"] == "seg":
        m32 = o32[0].argmax(0)
        m8 = o8[0].argmax(0)
        agree = float((m32 == m8).mean())
        rec["pixel_agreement_vs_fp32"] = round(agree, 4)
        rec["classes_fp32"] = sorted(set(m32.reshape(-1).tolist()))[:12]
        rec["classes_int8"] = sorted(set(m8.reshape(-1).tolist()))[:12]
        rec["top_classes_int8"] = [VOC[c] for c in np.bincount(m8.reshape(-1)).argsort()[::-1][:5]]
        save(PAL[m32], "%s_fp32_mask.png" % t["key"])
        save(PAL[m8], "%s_npu_mask.png" % t["key"])
        orig = (x * 255).astype(np.uint8)
        blend = (orig * 0.45 + PAL[m8] * 0.55).astype(np.uint8)
        save(blend, "%s_npu_overlay.png" % t["key"])
        diffmask = np.where(m32 == m8, 255, 0).astype(np.uint8)
        save(np.stack([diffmask, np.zeros_like(diffmask), np.where(m32 == m8, 0, 255).astype(np.uint8)], -1),
             "%s_diff.png" % t["key"])
    else:
        # unit: 输出 [-1,1] 或 [0,1] -> 归一化到 0..255
        def to_u8(a):
            a = np.squeeze(a)
            if a.ndim == 3 and a.shape[0] in (1, 3):
                a = a.transpose(1, 2, 0)
            lo, hi = float(a.min()), float(a.max())
            if lo < -0.05:      # tanh 输出
                a = (a + 1.0) * 0.5
            return (np.clip(a, 0, 1) * 255).astype(np.uint8)
        u32, u8 = to_u8(o32), to_u8(o8)
        rec["fp32_range"] = [float(o32.min()), float(o32.max())]
        rec["int8_range"] = [float(o8.min()), float(o8.max())]
        if u32.shape == u8.shape:
            d = np.abs(u32.astype(np.float32) - u8.astype(np.float32))
            rec["mae_vs_fp32"] = round(float(d.mean()), 2)
            mse = float((d ** 2).mean())
            rec["psnr_vs_fp32"] = round(10 * np.log10(255 * 255 / max(mse, 1e-8)), 2)
        save(u32, "%s_fp32_out.png" % t["key"])
        save(u8, "%s_npu_out.png" % t["key"])
        if t["key"] == "modnet_matting":
            a = np.squeeze(o8)
            a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
            bg = chessboard(512, 512)
            fg = (x * 255).astype(np.float32)
            comp = fg * a[..., None] + bg * (1 - a[..., None])
            save(comp.astype(np.uint8), "%s_npu_compose.png" % t["key"])
            save((a * 255).astype(np.uint8), "%s_npu_alpha.png" % t["key"])
    metrics[t["key"]] = rec
    print("[ok]", t["key"], json.dumps(rec, ensure_ascii=False)[:300])

(RES / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved metrics")
