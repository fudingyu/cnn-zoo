# CNN Zoo Studio - inference engine
#
# 10 models (5 new + 5 legacy) x 3 providers:
#   NPU  = VitisAIExecutionProvider (XDNA1, INT8)
#   CPU  = CPUExecutionProvider      (INT8, 8 threads)
#   iGPU = DmlExecutionProvider      (FP32, RDNA3 780M)
#
# Nothing here touches HTTP. Reuses the already-compiled VitisAI caches from
# npu_studio (keys "studio_*") and cnn_zoo (keys "zoo_*"), so warm-up is fast.
import os
import io
import sys
import json
import time
import ctypes
import threading
import collections

try:
    ctypes.windll.kernel32.SetThreadUILanguage(0x0409)
except Exception:
    pass
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))          # cnn_zoo/studio
ZOO = os.path.dirname(HERE)                                # cnn_zoo (repo root)
sys.path.insert(0, ZOO)
import paths as P

BASE = str(P.EXT)                                          # holds the AMD SDK snapshot

import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# vendored pre/post primitives: COCO names, anchors, letterbox, NMS, tiled SR
import legacy as LEG

INSTALL = str(P.INSTALL)
VOE = str(P.VOE)
XCLBIN = P.xclbin("4x4")
os.environ["RYZEN_AI_INSTALLATION_PATH"] = INSTALL

CACHE_NPU_NEW = os.path.join(ZOO, ".vitis_cache")                  # keys zoo_*
CACHE_NPU_OLD = os.path.join(BASE, ".vitis171_cache")              # keys studio_*
for d in (CACHE_NPU_NEW, CACHE_NPU_OLD):
    os.makedirs(d, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
       "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
       "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

_PROVIDERS = ("npu", "cpu", "igpu")

# XDNA1 的 XRT context 是稀缺资源：一个进程里同时活着的 NPU 会话太多会
# "Failed to create context" 甚至卡死在 native 重试里（实测 11 个全开会挂）。
# 所以 NPU 会话做 LRU 淘汰，上限 NPU_CAP 个；CPU/iGPU 会话不占 NPU context，不限。
NPU_CAP = 5
_NPU_LRU = collections.OrderedDict()      # key -> last use timestamp
_SESS_LOCK = threading.Lock()


def _touch_lru(key):
    with _SESS_LOCK:
        _NPU_LRU[key] = time.time()
        _NPU_LRU.move_to_end(key)


def _release_all_npu():
    """Drop every live NPU session and force the native XRT contexts to be freed."""
    import gc
    with _SESS_LOCK:
        for m in MODELS.values():
            m.sess.pop("npu", None)
        _NPU_LRU.clear()
        gc.collect()


def _enforce_npu_cap(incoming_key):
    """Keep at most NPU_CAP live NPU sessions; evict least-recently-used."""
    import gc
    with _SESS_LOCK:
        alive = [k for k in _NPU_LRU if k in MODELS and "npu" in MODELS[k].sess]
        while len(alive) >= NPU_CAP:
            victim = None
            for k in alive:
                if k != incoming_key:
                    victim = k
                    break
            if victim is None:
                return
            m = MODELS[victim]
            try:
                del m.sess["npu"]
            except Exception:
                pass
            _NPU_LRU.pop(victim, None)
            gc.collect()                    # free the native XRT context
            alive = [k for k in _NPU_LRU if k in MODELS and "npu" in MODELS[k].sess]
        _NPU_LRU[incoming_key] = time.time()


# --------------------------------------------------------------------------- #
# palette (VOC 21 colors, fixed assignment for stable legend)
# --------------------------------------------------------------------------- #
def voc_palette():
    p = np.zeros((21, 3), np.uint8)
    p[0] = (0, 0, 0)
    fixed = {
        1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0), 4: (0, 0, 128),
        5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (64, 0, 0),
        9: (192, 0, 0), 10: (64, 128, 0), 11: (192, 128, 0), 12: (64, 0, 128),
        13: (192, 0, 128), 14: (64, 128, 128), 15: (192, 128, 128), 16: (0, 64, 0),
        17: (128, 64, 0), 18: (0, 192, 0), 19: (128, 192, 0), 20: (0, 64, 128),
    }
    for k, v in fixed.items():
        p[k] = v
    return p


PAL21 = voc_palette()


# --------------------------------------------------------------------------- #
# pre / post for the 5 new models (all verified in cnn_zoo/*.py)
# --------------------------------------------------------------------------- #
def _lb_geom(w, h, size):
    """letterbox 几何：等比缩放进 size×size，居中，返回 (scale, dx, dy, nw, nh)。"""
    s = min(size / float(w), size / float(h))
    nw, nh = int(round(w * s)), int(round(h * s))
    nw, nh = min(nw, size), min(nh, size)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    return s, dx, dy, nw, nh


def letterbox(img, size):
    """等比缩放 + 反射填充到 size×size（替代直接 resize 成方形的拉伸做法）。"""
    h, w = img.shape[:2]
    _s, dx, dy, nw, nh = _lb_geom(w, h, size)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(r, dy, size - nh - dy, dx, size - nw - dx,
                              cv2.BORDER_REFLECT_101)


def unletterbox(img, w, h, size, scale=1):
    """从 letterbox 过的模型输出里裁出有效区域，还原到 (w*scale, h*scale)。"""
    _s, dx, dy, nw, nh = _lb_geom(w, h, size)
    x0, y0 = int(round(dx * scale)), int(round(dy * scale))
    x1, y1 = int(round((dx + nw) * scale)), int(round((dy + nh) * scale))
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_LINEAR)


def prep_seg(img, size=512):
    """NCHW RGB + ImageNet mean/std (torchvision segmentation pipeline)."""
    r = letterbox(img, size)
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def _strip_batch(out):
    """onnx outputs keep the batch dim: (1,C,H,W) -> (C,H,W)."""
    a = np.asarray(out[0])
    if a.ndim == 4:
        a = a[0]
    return a


def post_seg(out, orig_bgr):
    """(1,21,H,W) logits -> argmax mask + blended overlay + class stats."""
    a = _strip_batch(out)
    m = a.argmax(0).astype(np.uint8)          # model resolution (512,512)
    h, w = orig_bgr.shape[:2]
    mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    color = PAL21[mask]                        # (h,w,3)
    blend = (orig_bgr.astype(np.float32) * 0.45 + color.astype(np.float32) * 0.55)
    overlay = blend.astype(np.uint8)
    counts = np.bincount(mask.reshape(-1), minlength=21)
    total = max(int(counts.sum()), 1)
    top = [(VOC[i], round(float(counts[i]) / total * 100, 1))
           for i in counts.argsort()[::-1][:4] if counts[i] > total * 0.01]
    return {"kind": "seg", "image": overlay, "mask": color, "top": top}


def prep_unit(img, size=512):
    """NCHW RGB in [-1,1] (MODNet / AnimeGANv2 pipeline), letterboxed."""
    p = letterbox(img, size)
    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = rgb * 2.0 - 1.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def post_anime(out, orig_bgr):
    a = _strip_batch(out).transpose(1, 2, 0)   # HWC in [-1,1]
    img = np.clip((a + 1.0) * 0.5, 0, 1)
    bgr = (img * 255).astype(np.uint8)[:, :, ::-1]
    h, w = orig_bgr.shape[:2]
    return {"kind": "unit", "image": unletterbox(bgr, w, h, 512)}


def _chessboard(h, w, tile=32):
    y, x = np.mgrid[0:h, 0:w]
    c = ((x // tile) + (y // tile)) % 2
    a = (c * 40 + 190).astype(np.uint8)
    b = (c * -40 + 235).astype(np.uint8)
    out = np.zeros((h, w, 3), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = a, b, b
    return out


def post_modnet(out, orig_bgr):
    """MODNet alpha (H,W) in [0,1] -> matte + chessboard composite."""
    a = np.squeeze(np.asarray(out[0]))
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[:, :, 0]
    if a.min() < -0.05:
        a = (a + 1.0) * 0.5
    a = np.clip(a, 0, 1)
    h, w = orig_bgr.shape[:2]
    # alpha 也是在 letterbox 输入上算的，先裁掉填充边再缩回原尺寸
    a8 = unletterbox((a * 255).astype(np.uint8), w, h, 512)
    a = a8.astype(np.float32) / 255.0
    fg = orig_bgr.astype(np.float32)
    bg = _chessboard(h, w).astype(np.float32)
    comp = (fg * a[..., None] + bg * (1 - a[..., None])).astype(np.uint8)
    matte = (a * 255).astype(np.uint8)
    cov = float((a > 0.5).mean()) * 100.0
    return {"kind": "matte", "image": comp, "mask": matte,
            "note": "前景覆盖率 %.1f%%" % cov}


def prep_sr256(img):
    p = letterbox(img, 256)
    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def post_realesrgan(out, orig_bgr):
    """(1,3,1024,1024) [0,1] -> uint8 BGR 4x，letterbox 有效区裁回原始宽高比。"""
    a = _strip_batch(out).transpose(1, 2, 0)
    img = np.clip(a, 0, 1)
    bgr = (img * 255).astype(np.uint8)[:, :, ::-1]
    h, w = orig_bgr.shape[:2]
    return {"kind": "unit", "image": unletterbox(bgr, w, h, 256, scale=4),
            "note": "4× 超分 · 等比输出 %d×%d" % (w * 4, h * 4)}


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #
class ZooModel(object):
    def __init__(self, key, title, desc, group, kind, int8=None, fp32=None,
                 in_name="input", size=512, prep=None, post=None,
                 cache_key=None, cache_dir=None, bench_key=None, warn=None):
        self.key = key
        self.title = title
        self.desc = desc
        self.group = group            # "new" | "old"
        self.kind = kind              # seg | unit | matte | detect | face | classify | sr
        self.int8 = int8              # NPU + CPU share this file
        self.fp32 = fp32              # iGPU (DML)
        self.in_name = in_name
        self.size = size
        self.prep = prep
        self.post = post
        self.cache_key = cache_key or ("zoo_%s_4x4" % key)
        self.cache_dir = cache_dir or CACHE_NPU_NEW
        self.bench_key = bench_key or key
        self.warn = warn or {}        # provider -> honesty note
        self.lock = threading.Lock()
        self.sess = {}                # provider -> session
        self.times = collections.defaultdict(collections.deque)
        for p in _PROVIDERS:
            self.times[p] = collections.deque(maxlen=40)
        self.load_sec = collections.defaultdict(float)

    # -- availability ------------------------------------------------------- #
    def has(self, provider):
        if provider in ("npu", "cpu"):
            return bool(self.int8) and os.path.exists(self.int8)
        return bool(self.fp32) and os.path.exists(self.fp32)

    def path(self, provider):
        return self.int8 if provider in ("npu", "cpu") else self.fp32

    # -- sessions ----------------------------------------------------------- #
    def _make(self, provider):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if provider == "npu":
            so.intra_op_num_threads = 1
            opts = {"target": "X1",
                    "xlnx_enable_py3_round": 0,
                    "xclbin": XCLBIN,
                    "cache_dir": self.cache_dir,
                    "cache_key": self.cache_key,
                    "enable_cache_file_io_in_mem": "0"}
            prov = [("VitisAIExecutionProvider", opts), "CPUExecutionProvider"]
        elif provider == "cpu":
            so.intra_op_num_threads = min(8, os.cpu_count() or 8)
            prov = ["CPUExecutionProvider"]
        else:
            so.intra_op_num_threads = 1
            prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ort.InferenceSession(self.path(provider), sess_options=so, providers=prov)

    def load(self, provider):
        if not self.has(provider):
            return "provider 不可用（缺少 %s 模型文件）" % provider
        if provider in self.sess:
            if provider == "npu":
                _touch_lru(self.key)
            return None
        try:
            with self.lock:
                if provider not in self.sess:
                    if provider == "npu":
                        _enforce_npu_cap(self.key)
                    t0 = time.time()
                    try:
                        self.sess[provider] = self._make(provider)
                    except Exception as make_err:
                        # XRT context 释放是异步的：连续创建会话时后一个可能失败。
                        # 全量释放 NPU 会话、稍等再试一次。
                        if provider == "npu":
                            _release_all_npu()
                            time.sleep(1.5)
                            self.sess[provider] = self._make(provider)
                        else:
                            raise
                    self.load_sec[provider] = round(time.time() - t0, 2)
                    if provider == "npu":
                        _touch_lru(self.key)
        except Exception as e:
            return "%s: %s" % (type(e).__name__, str(e)[:200])
        return None

    # -- inference ---------------------------------------------------------- #
    def run(self, frame_bgr, provider):
        """Returns (result_dict, ms). Raises on failure."""
        err = self.load(provider)
        if err:
            raise RuntimeError(err)
        x = self.prep(frame_bgr)
        sess = self.sess[provider]
        name = sess.get_inputs()[0].name
        with self.lock:
            t0 = time.perf_counter()
            out = sess.run(None, {name: x})
            ms = (time.perf_counter() - t0) * 1000.0
        self.times[provider].append(ms)
        res = self.post(out, frame_bgr)
        res["ms"] = round(ms, 2)
        return res


MODELS = collections.OrderedDict()

#: declared in this file but skipped because their weights are not on disk
MISSING = []


def _p(*rel):
    return os.path.join(BASE, *rel)


def reg(m):
    """Register a model only if its weights are actually present.

    Four entries (Real-ESRGAN, MobileNetV2, RetinaFace, RCAN) rely on
    third-party ONNX files this repo cannot redistribute. Rather than
    advertising cards that fail on click, they are skipped and listed in
    `MISSING`; drop the files in place and they reappear automatically.
    """
    have_int8 = bool(m.int8) and os.path.exists(str(m.int8))
    have_fp32 = bool(m.fp32) and os.path.exists(str(m.fp32))
    if not (have_int8 or have_fp32):
        MISSING.append(m.key)
        return None
    MODELS[m.key] = m
    return m


# ---- 5 new models -------------------------------------------------------- #
reg(ZooModel("lraspp_seg", "LRASPP 语义分割", "PASCAL VOC 21 类 · 整图 1 个 NPU 子图 · 本轮唯一跑赢 CPU 的大模型",
             "new", "seg",
             int8=os.path.join(ZOO, "int8", "lraspp_seg_XINT8.onnx"),
             fp32=os.path.join(ZOO, "fp32", "lraspp_mobilenet_v3_large_512.onnx"),
             prep=prep_seg, post=post_seg))

reg(ZooModel("deeplabv3_seg", "DeepLabV3 语义分割", "VOC 21 类 · ASPP 空洞卷积 · 2 个 NPU 子图仍慢于 CPU",
             "new", "seg",
             int8=os.path.join(ZOO, "int8", "deeplabv3_seg_XINT8_MSE.onnx"),
             fp32=os.path.join(ZOO, "fp32", "deeplabv3_mobilenet_v3_large_512.onnx"),
             prep=prep_seg, post=post_seg))

reg(ZooModel("modnet_matting", "MODNet 人像抠图", "软 alpha 抠图 · EP 放弃编译（0 子图全回退）· INT8 会崩需 MinMSE 标定",
             "new", "matte",
             int8=os.path.join(ZOO, "int8", "modnet_matting_XINT8_MSE.onnx"),
             fp32=os.path.join(ZOO, "fp32", "modnet_fp32_static.onnx"),
             prep=prep_unit, post=post_modnet,
             warn={"npu": "INT8 标定不稳定：EP 实为 0 子图全走 CPU 兜底，NPU 结果仅作示意，以 iGPU 为准"}))

reg(ZooModel("animeganv2", "AnimeGANv2 动漫风格化", "日系动漫风 · 40 个碎片子图 · NPU 3.7s/帧被搬运开销拖垮",
             "new", "unit",
             int8=os.path.join(ZOO, "int8", "animeganv2_XINT8.onnx"),
             fp32=os.path.join(ZOO, "fp32", "animeganv2_fp32_static.onnx"),
             in_name="input_image", prep=prep_unit, post=post_anime))

reg(ZooModel("realesrgan", "Real-ESRGAN ×4 超分", "RRDB 4 倍重建 · 81 个碎片子图 · 输入 256 输出 1024",
             "new", "unit",
             int8=_p("realsrgan256", "onnx-models", "realesrgan_nchw_256x256_u8s8.onnx"),
             fp32=_p("realsrgan256", "onnx-models", "realesrgan_nchw_256x256_fp32.onnx"),
             size=256, prep=prep_sr256, post=post_realesrgan))

# ---- 5 legacy models (same pre/post as npu_studio, reuses its NPU cache) -- #
_SW = os.path.join(BASE, "RyzenAI-SW-1.7.1", "CNN-examples")


def _wrap_legacy(key, title, desc, kind, int8, fp32=None, in_name="input",
                 prep=None, post=None, cache_key=None):
    """Legacy model whose pre/post already live in npu_studio.engine.
    cache_key matches npu_studio's so the already-compiled VitisAI caches hit."""
    m = ZooModel(key, title, desc, "old", kind, int8=int8, fp32=fp32,
                 in_name=in_name, prep=prep, post=post,
                 cache_key=cache_key or ("studio_%s_4x4" % key),
                 cache_dir=CACHE_NPU_OLD,
                 bench_key=key)
    return m


# yolo: letterbox 640 + NMS (from npu_studio engine)
def prep_yolo640(img):
    x, meta = LEG.prep_yolo(img, 640)
    return x


def post_yolo_det(out, frame):
    # meta is recomputed by caller; npu_studio stores it differently, so we
    # letterbox again here (cheap, ~0.2ms) to stay self-contained
    lb, meta = LEG.letterbox(frame, (640, 640))
    dets = LEG.post_yolo(out, meta, conf=0.35)
    return {"kind": "detect", "dets": dets}


reg(_wrap_legacy("yolov8m", "YOLOv8m 目标检测", "COCO 80 类 · 640×640 · NPU 4.5× 真加速",
                 "detect",
                 os.path.join(_SW, "object_detection", "yolov8m", "models",
                              "yolov8m_XINT8_skip.onnx"),
                 fp32=os.path.join(_SW, "object_detection", "yolov8m", "models",
                                   "yolov8m.onnx"),
                 in_name="images", prep=prep_yolo640, post=post_yolo_det,
                 cache_key="studio_yolo_4x4"))

# resnet50: 224 center crop, ImageNet norm
def prep_resnet(img):
    x, _ = LEG.prep_cls224_nchw(img)
    return x


def post_resnet(out, frame):
    return {"kind": "classify", "top": LEG.post_cls(out, topk=5)}


reg(_wrap_legacy("resnet50", "ResNet50 分类", "ImageNet 1000 类 · NPU 6.9× 真加速",
                 "classify",
                 os.path.join(_SW, "torchvision_inference", "models",
                              "resnet50_XINT8.onnx"),
                 fp32=os.path.join(_SW, "torchvision_inference", "models",
                                   "resnet50.onnx"),
                 prep=prep_resnet, post=post_resnet,
                 cache_key="studio_resnet50_4x4"))

# mobilenetv2: NHWC 224, plain 0..1
def prep_mnv2(img):
    x, _ = LEG.prep_mnv2(img)
    return x


reg(_wrap_legacy("mobilenetv2", "MobileNetV2 分类", "ImageNet 1000 类 · NHWC 布局 · NPU 4.2×",
                 "classify",
                 _p("hf_models", "mobilenetv2_int8.onnx"),
                 in_name="EfficientNet::input_0",
                 prep=prep_mnv2, post=post_resnet,
                 cache_key="studio_mobilenet_4x4"))

# retinaface: letterbox 640x608 NHWC, BGR mean subtraction
def prep_retina(img):
    x, _ = LEG.prep_retina(img)
    return x


def post_retina(out, frame):
    lb, meta = LEG.letterbox(frame, (LEG.RF_H, LEG.RF_W))
    faces = LEG.post_retina(out, meta, conf=0.5)
    return {"kind": "face", "faces": faces}


reg(_wrap_legacy("retinaface", "RetinaFace 人脸检测", "人脸框 + 五点关键点 · NHWC 608×640 · NPU 3.1×",
                 "face",
                 _p("hf_models", "RetinaFace_int.onnx"),
                 in_name="RetinaFace::input_0",
                 prep=prep_retina, post=post_retina,
                 cache_key="studio_retina_4x4"))

# rcan: tiled 2x SR (same as npu_studio live path)
def prep_rcan(img):
    return img          # tiled path takes the raw frame


def post_rcan(_out, frame):
    return {"kind": "unit", "note": "tiled"}


class _RcanProxy(object):
    pass


# RCAN needs the tiled loop, so handle it as a special case in run_rcan()
reg(_wrap_legacy("rcan_sr", "RCAN 超分辨率", "2× 分块重建 · 56×56 固定输入 · NPU 4.9×",
                 "sr",
                 _p("models_src", "RCAN_int8_NHWC.onnx"),
                 in_name="Model::input_0",
                 prep=prep_rcan, post=post_rcan,
                 cache_key="studio_rcan_4x4"))


def run_rcan(m, frame, provider):
    """RCAN fixed 56x56 input -> tiled 2x whole-image SR (npu_studio path).
    注意：不能在这里持 m.lock——sr_tiled 内部会自己拿 proxy.lock（= m.lock），
    非重入锁会造成自死锁。"""
    err = m.load(provider)
    if err:
        raise RuntimeError(err)
    proxy = _RcanProxy()
    proxy.sess = m.sess[provider]
    proxy.sess_cpu = m.sess.get("cpu")
    proxy.lock = m.lock
    t0 = time.perf_counter()
    img, _ms = LEG.sr_tiled(proxy, frame, npu=(provider == "npu"))
    ms = (time.perf_counter() - t0) * 1000.0
    m.times[provider].append(ms)
    return {"kind": "unit", "image": img, "ms": round(ms, 2),
            "note": "2× 分块超分"}


# --------------------------------------------------------------------------- #
# drawing results onto frames (server-side, keeps frontend dumb)
# --------------------------------------------------------------------------- #
def draw_dets(frame, dets, color=(90, 220, 255)):
    for d in dets:
        b = d["box"]
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
        txt = "%s %.2f" % (d.get("label", "?"), d.get("score", 0))
        y = max(int(b[1]) - 6, 20)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (int(b[0]), y - th - 6), (int(b[0]) + tw, y), (16, 20, 28), -1)
        cv2.putText(frame, txt, (int(b[0]), y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for p in (d.get("landmarks") or []):
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, color, -1)


def render(m, frame_bgr, provider):
    """Run model and return a displayable BGR image + metadata."""
    if m.key == "rcan_sr":
        res = run_rcan(m, frame_bgr, provider)
    else:
        res = m.run(frame_bgr, provider)
    kind = res.get("kind")
    if kind == "detect":
        vis = frame_bgr.copy()
        draw_dets(vis, res.get("dets") or [])
        res["image"] = vis
    elif kind == "face":
        vis = frame_bgr.copy()
        draw_dets(vis, res.get("faces") or [])
        res["image"] = vis
    elif kind == "classify":
        vis = frame_bgr.copy()
        top = res.get("top") or []
        lines = ["%d. %-18s %.3f" % (i + 1, d["label"][:18], d["score"])
                 for i, d in enumerate(top)]
        _panel(vis, lines, title=m.title.split()[0])
        res["image"] = vis
    return res


def _panel(img, lines, title=""):
    """Dark info panel, bottom-left."""
    h, w = img.shape[:2]
    fs = max(0.45, min(0.7, w / 900.0))
    th = 22
    pad = 10
    box_h = pad * 2 + th * (len(lines) + (1 if title else 0))
    y0 = max(h - box_h - 12, 8)
    cv2.rectangle(img, (8, y0), (8 + int(w * 0.34), y0 + box_h), (16, 20, 28), -1)
    cv2.rectangle(img, (8, y0), (8 + int(w * 0.34), y0 + box_h), (60, 72, 90), 1)
    y = y0 + pad + th - 6
    if title:
        cv2.putText(img, title, (18, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (120, 230, 205), 1, cv2.LINE_AA)
        y += th
    for ln in lines:
        cv2.putText(img, ln, (18, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (225, 232, 240), 1, cv2.LINE_AA)
        y += th


# --------------------------------------------------------------------------- #
# static benchmark numbers (from the cnn_zoo report run) for the catalogue
# --------------------------------------------------------------------------- #
def load_static_bench():
    out = {}
    for fname, keys in (("bench.json", None), ("bench_hist.json", None)):
        p = os.path.join(ZOO, "result", fname)
        if not os.path.exists(p):
            continue
        try:
            for r in json.load(open(p, encoding="utf-8")):
                out[r.get("bench_key") or r["key"]] = r
        except Exception:
            pass
    p = os.path.join(ZOO, "result", "bench_dml.json")
    if os.path.exists(p):
        try:
            for r in json.load(open(p, encoding="utf-8")):
                k = r["key"]
                if k in out:
                    out[k]["dml_ms"] = r.get("dml_ms")
                else:
                    r2 = dict(r)
                    r2["dml_ms"] = r.get("dml_ms")
                    out[k] = r2
        except Exception:
            pass
    p = os.path.join(ZOO, "result", "npu_subgraph.json")
    if os.path.exists(p):
        try:
            for k, v in json.load(open(p, encoding="utf-8")).items():
                kk = k.replace("zoo_", "").replace("_4x4", "")
                if kk in out:
                    out[kk]["subgraphs"] = v.get("subgraphs")
                    out[kk]["npu_nodes"] = v.get("npu_nodes")
        except Exception:
            pass
    return out


STATIC_BENCH = load_static_bench()


def avg(provider, key):
    d = MODELS[key].times[provider]
    return round(sum(d) / len(d), 2) if d else None


def catalogue():
    items = []
    for m in MODELS.values():
        b = STATIC_BENCH.get(m.bench_key, {})
        items.append({
            "key": m.key, "title": m.title, "desc": m.desc, "group": m.group,
            "kind": m.kind,
            "providers": [p for p in _PROVIDERS if m.has(p)],
            "bench": {
                "npu": b.get("npu_ms"), "cpu": b.get("cpu_ms"),
                "dml": b.get("dml_ms"), "fp32_cpu": b.get("fp32_cpu_ms"),
                "subgraphs": b.get("subgraphs"), "npu_nodes": b.get("npu_nodes"),
                "size_mb": b.get("model_size_mb"),
            },
            "live": {p: avg(p, m.key) for p in _PROVIDERS},
        })
    return items


# --------------------------------------------------------------------------- #
# warm-up: build every NPU session + one dummy run (compiled caches are reused)
# --------------------------------------------------------------------------- #
WARM = {"total": 0, "done": 0, "current": "", "ready": False, "sec": 0.0,
        "failed": []}


def warmup_all():
    """Pre-build NPU sessions for the 5 new models (within the context cap).
    Legacy models load on demand and evict LRU sessions when needed."""
    t0 = time.time()
    keys = [k for k, m in MODELS.items() if m.group == "new"]
    WARM["total"] = len(keys)
    WARM["done"] = 0
    WARM["failed"] = []
    dummy = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(dummy, (140, 120), (460, 360), (70, 120, 200), -1)
    for k in keys:
        m = MODELS[k]
        WARM["current"] = m.title
        try:
            if m.load("npu") is None:
                render(m, dummy, "npu")
        except Exception:
            WARM["failed"].append(k)
        # 立即释放 NPU 会话：预热只验证"能编译能跑"，不常驻，
        # 避免 10 个模型连续创建会话时 XRT 上下文释放跟不上而失败/挂死
        m.sess.pop("npu", None)
        WARM["done"] += 1
    import gc as _gc
    _gc.collect()
    WARM["current"] = ""
    WARM["sec"] = round(time.time() - t0, 1)
    WARM["ready"] = True
