"""NPU(VitisAI EP) vs CPU 基准：候选 CNN 模型。
用法:
    python bench.py                 # 静默计时，结果写 result/bench.json
    LOG=1 python bench.py           # verbose，把 ORT 日志打出来（用于抓 No. of Operators）
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
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import paths as P

ROOT = Path(__file__).resolve().parent
INSTALL = str(P.INSTALL)
VOE = str(P.VOE)
XCLBIN = P.xclbin("4x4")
CACHE_DIR = str(ROOT / ".vitis_cache")
os.environ["RYZEN_AI_INSTALLATION_PATH"] = INSTALL
os.makedirs(CACHE_DIR, exist_ok=True)


def resolve(p):
    """Repo-relative paths stay relative; SDK/third-party ones arrive absolute."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

VERBOSE = os.environ.get("LOG", "0") == "1"
WARMUP = int(os.environ.get("WARMUP", "5"))
RUNS = int(os.environ.get("RUNS", "20"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_input(kind, shape=None, size=512):
    """按模型声明的输入布局生成一批像真实图像的随机数（避免全零/全一的退化分布）。"""
    if shape is None:
        shape = [1, 3, size, size]
    rnd = np.random.RandomState(7)
    nchw = shape[-1] != 3                # 末维不是 3 → NCHW；否则 NHWC
    if nchw:
        c, h, w = shape[1], shape[2], shape[3]
    else:
        h, w, c = shape[1], shape[2], shape[3]
    x = rnd.rand(h, w, c).astype(np.float32) * 0.6 + 0.2
    if kind == "seg":
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    else:                                    # unit / [-1,1]
        x = x * 2.0 - 1.0
    if nchw:
        x = x.transpose(2, 0, 1)
    return x[None].astype(np.float32)


def build(path, npu, key):
    so = ort.SessionOptions()
    so.log_severity_level = 0 if VERBOSE else 3
    if npu:
        # NPU 场景下 CPU 只跑兜底算子，单线程即可
        so.intra_op_num_threads = 1
    else:
        # 纯 CPU 对照：放开线程，避免低估 CPU 能力
        so.intra_op_num_threads = min(8, os.cpu_count() or 8)
    if npu:
        opts = {"target": "X1",
                "xlnx_enable_py3_round": 0,
                "xclbin": XCLBIN,
                "cache_dir": CACHE_DIR,
                "cache_key": "zoo_%s_4x4" % key,
                "enable_cache_file_io_in_mem": "0"}
        prov = [("VitisAIExecutionProvider", opts), "CPUExecutionProvider"]
    else:
        prov = ["CPUExecutionProvider"]
    t0 = time.time()
    sess = ort.InferenceSession(str(path), sess_options=so, providers=prov)
    return sess, time.time() - t0


def timeit(sess, feed, in_name, runs=RUNS, warmup=WARMUP):
    out_name = None
    for _ in range(warmup):
        sess.run(None, {in_name: feed})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {in_name: feed})
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return float(ts.mean()), float(ts.min()), float(ts.max())


JOBS = [
    dict(key="lraspp_seg",     path="int8/lraspp_seg_XINT8.onnx",     in_name="input",       kind="seg",  size=512, title="LRASPP-MobileNetV3 语义分割",
         fp32="fp32/lraspp_mobilenet_v3_large_512.onnx"),
    dict(key="deeplabv3_seg",  path="int8/deeplabv3_seg_XINT8.onnx",  in_name="input",       kind="seg",  size=512, title="DeepLabV3-MobileNetV3 语义分割",
         fp32="fp32/deeplabv3_mobilenet_v3_large_512.onnx"),
    dict(key="modnet_matting", path="int8/modnet_matting_XINT8.onnx", in_name="input",       kind="unit", size=512, title="MODNet 人像抠图",
         fp32="fp32/modnet_fp32_static.onnx"),
    dict(key="animeganv2",     path="int8/animeganv2_XINT8.onnx",     in_name="input_image", kind="unit", size=512, title="AnimeGANv2 动漫风格化",
         fp32="fp32/animeganv2_fp32_static.onnx"),
    dict(key="realesrgan",     path=P.ext("realsrgan256", "onnx-models", "realesrgan_nchw_256x256_u8s8.onnx"),
         in_name="input", kind="unit", size=256, title="Real-ESRGAN x4 超分",
         fp32=P.ext("realsrgan256", "onnx-models", "realesrgan_nchw_256x256_fp32.onnx")),
]

# ---- 第二批：此前跑过的模型，用完全相同的口径重测，便于横向对比 ----
_SW = P.sdk("CNN-examples")
JOBS_HISTORY = [
    dict(key="yolov8m", path=os.path.join(_SW, "object_detection", "yolov8m", "models", "yolov8m_XINT8_skip.onnx"),
         in_name="images", shape=[1, 3, 640, 640], kind="seg", title="YOLOv8m 目标检测"),
    dict(key="resnet50", path=os.path.join(_SW, "torchvision_inference", "models", "resnet50_XINT8.onnx"),
         in_name="input", shape=[1, 3, 224, 224], kind="seg", title="ResNet50 分类"),
    dict(key="mobilenetv2", path=P.ext("hf_models", "mobilenetv2_int8.onnx"),
         in_name="EfficientNet::input_0", shape=[1, 224, 224, 3], kind="seg", title="MobileNetV2 分类"),
    dict(key="retinaface", path=P.ext("hf_models", "RetinaFace_int.onnx"),
         in_name="RetinaFace::input_0", shape=[1, 608, 640, 3], kind="unit", title="RetinaFace 人脸检测"),
    dict(key="rcan_sr", path=P.ext("models_src", "RCAN_int8_NHWC.onnx"),
         in_name="Model::input_0", shape=[1, 56, 56, 3], kind="unit", title="RCAN 超分(2×)"),
]

# ---- 输出文件名随批次走 ----
OUTNAME = "bench_hist.json" if os.environ.get("MODE") == "hist" else "bench.json"
if os.environ.get("MODE") == "hist":
    JOBS = JOBS_HISTORY

only = sys.argv[1:] or None
results = []
for j in JOBS:
    if only and j["key"] not in only:
        continue
    p = resolve(j["path"])
    if not p.exists() or p.stat().st_size < 1000:
        print("[skip] %s missing" % j["key"])
        continue
    feed = make_input(j["kind"], j.get("shape"), j.get("size", 512))
    rec = dict(j)
    rec["model_size_mb"] = round(p.stat().st_size / 1e6, 2)
    rec["input_shape"] = j.get("shape") or [1, 3, j.get("size", 512), j.get("size", 512)]
    rec["cpu_threads"] = min(8, os.cpu_count() or 8)
    # --- NPU ---
    try:
        s, load = build(p, True, j["key"])
        rec["npu_load_s"] = round(load, 2)
        m, mn, mx = timeit(s, feed, j["in_name"])
        rec["npu_ms"], rec["npu_min_ms"], rec["npu_max_ms"] = round(m, 2), round(mn, 2), round(mx, 2)
        rec["npu_ok"] = True
    except Exception as e:
        rec["npu_ok"] = False
        rec["npu_err"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    # --- CPU ---
    try:
        s2, load2 = build(p, False, j["key"])
        rec["cpu_load_s"] = round(load2, 2)
        m2, mn2, mx2 = timeit(s2, feed, j["in_name"])
        rec["cpu_ms"], rec["cpu_min_ms"], rec["cpu_max_ms"] = round(m2, 2), round(mn2, 2), round(mx2, 2)
        rec["cpu_ok"] = True
    except Exception as e:
        rec["cpu_ok"] = False
        rec["cpu_err"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    # --- FP32 CPU（现实基线：不量化就得跑 fp32）---
    fp = j.get("fp32")
    if fp:
        pf = resolve(fp)
        if pf.exists() and pf.stat().st_size > 1000:
            try:
                s3, _ = build(pf, False, j["key"] + "_fp32")
                m3, _, _ = timeit(s3, feed, j["in_name"])
                rec["fp32_cpu_ms"] = round(m3, 2)
            except Exception as e:
                rec["fp32_err"] = "%s: %s" % (type(e).__name__, str(e)[:120])
    if rec.get("npu_ok") and rec.get("cpu_ok"):
        rec["speedup"] = round(rec["cpu_ms"] / rec["npu_ms"], 2)
    if rec.get("npu_ok") and rec.get("fp32_cpu_ms"):
        rec["speedup_vs_fp32"] = round(rec["fp32_cpu_ms"] / rec["npu_ms"], 2)
    print("[%s] npu=%s cpu_int8=%s fp32_cpu=%s vs_int8=%s vs_fp32=%s" %
          (j["key"], rec.get("npu_ms"), rec.get("cpu_ms"), rec.get("fp32_cpu_ms"),
           rec.get("speedup"), rec.get("speedup_vs_fp32")))
    results.append(rec)

out = ROOT / "result"
out.mkdir(exist_ok=True)
(out / OUTNAME).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved", out / OUTNAME)
