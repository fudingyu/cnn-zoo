# Pre/post-processing primitives for the legacy (official AMD example) models:
# YOLOv8m letterbox + NMS, 224 centre-crop classification, MobileNetV2 NHWC,
# RetinaFace anchors, RCAN tiled 2x SR, plus the VitisAI session plumbing.
#
# Vendored from npu_studio/engine.py so that cnn_zoo is a self-contained repo.
# zoo_engine.py only exercises the helpers it reaches through `LEG.*`; the
# camera/live-path leftovers are harmless and kept for reproduction fidelity.
import os
import io
import sys
import json
import time
import ctypes
import threading
import collections

# Chinese Windows turns pybind exceptions into UnicodeDecodeError. Force en-US.
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
import cv2
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))          # cnn_zoo/studio
ROOT = os.path.dirname(HERE)                               # cnn_zoo
sys.path.insert(0, ROOT)
import paths as P

INSTALL = str(P.INSTALL)
VOE = str(P.VOE)
XCLBINS = {
    "1x4": P.xclbin("1x4"),
    "4x4": P.xclbin("4x4"),
}
# Shares the compile cache with npu_studio-era runs (keys "studio_*"), so the
# first launch after a clone is warm instead of recompiling every model.
CACHE_DIR = str(P.EXT / ".vitis171_cache")

os.environ["RYZEN_AI_INSTALLATION_PATH"] = INSTALL
os.makedirs(CACHE_DIR, exist_ok=True)

COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
              "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
              "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
              "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
              "hair drier", "toothbrush"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load_imagenet_labels():
    p = os.path.join(HERE, "assets", "imagenet_classes.txt")
    if not os.path.exists(p):
        return ["class_%d" % i for i in range(1000)]
    with open(p, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    # cdn file has 999 lines ("73" insect class missing) -> repair by index insert
    if len(names) == 999:
        names.insert(72, "insect")
    while len(names) < 1000:
        names.append("class_%d" % len(names))
    return names[:1000]


LABELS = _load_imagenet_labels()


# --------------------------------------------------------------------------- #
# box utilities
# --------------------------------------------------------------------------- #
def nms(boxes, scores, thr=0.45):
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, np.float32)
    s = np.asarray(scores, np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= thr]
    return keep


def letterbox(img, size, color=(0, 0, 0)):
    """resize keeping aspect ratio, pad to (w,h). returns img, (scale, padx, pady)"""
    h, w = size if len(size) == 2 else (size[0], size[1])
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padx, pady = (w - nw) / 2.0, (h - nh) / 2.0
    top, bot = int(pady), int(h - nh - pady + 0.5)
    left, right = int(padx), int(w - nw - padx + 0.5)
    out = cv2.copyMakeBorder(r, top, bot, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, (s, left, top)


# --------------------------------------------------------------------------- #
# model wrapper
# --------------------------------------------------------------------------- #
class Wrap(object):
    def __init__(self, key, title, desc, path, kind, xclbin="4x4", needs=(0,), pad=None):
        self.key = key
        self.title = title
        self.desc = desc
        self.path = path
        self.kind = kind              # detect | face | classify | sr
        self.xclbin = xclbin
        self.lock = threading.Lock()
        self.sess = None              # NPU session (lazy)
        self.sess_cpu = None          # CPU session (lazy)
        self.error = None if os.path.exists(path) else "model file missing"
        self.load_sec = 0.0
        self.calls = 0
        self.times = collections.deque(maxlen=60)
        self.cpu_times = collections.deque(maxlen=60)

    # -- session ----------------------------------------------------------- #
    def _make(self, npu):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 1
        if npu:
            opts = {"target": "X1",
                    "xlnx_enable_py3_round": 0,
                    "xclbin": XCLBINS[self.xclbin],
                    "cache_dir": CACHE_DIR,
                    "cache_key": "studio_%s_%s" % (self.key, self.xclbin),
                    "enable_cache_file_io_in_mem": "0"}
            prov = [("VitisAIExecutionProvider", opts), "CPUExecutionProvider"]
        else:
            prov = ["CPUExecutionProvider"]
        return ort.InferenceSession(self.path, sess_options=so, providers=prov)

    def load(self, npu=True):
        """Create session on demand. Returns error string or None."""
        if self.error:
            return self.error
        target_attr = "sess" if npu else "sess_cpu"
        if getattr(self, target_attr) is not None:
            return None
        try:
            with self.lock:
                if getattr(self, target_attr) is None:
                    t0 = time.time()
                    setattr(self, target_attr, self._make(npu))
                    self.load_sec = time.time() - t0
        except Exception as e:
            self.error = str(e)[:300]
        return self.error

    def input_name(self, sess=None):
        sess = sess or self.sess
        return sess.get_inputs()[0].name

    def run_raw(self, feed, npu=True):
        sess = self.sess if npu else self.sess_cpu
        return sess.run(None, feed)

    def note(self, ms, npu):
        (self.times if npu else self.cpu_times).append(ms)
        self.calls += 1


# --------------------------------------------------------------------------- #
# per-kind pre / post
# --------------------------------------------------------------------------- #
def prep_yolo(img, w=640):
    """Letterbox to 640x640 then NCHW RGB/255 (AMD official pipeline)."""
    lb, meta = letterbox(img, (w, w))
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])
    return x, meta


def _unlb(boxes, meta, shape):
    """letterbox coords -> original frame coords"""
    s, padx, pady = meta
    b = np.asarray(boxes, np.float32).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - padx) / s
    b[:, [1, 3]] = (b[:, [1, 3]] - pady) / s
    return b


def post_yolo(out, meta, conf=0.35, iou=0.45):
    p = out[0][0]                       # (84, 8400)
    cls_scores = p[4:]
    scores = cls_scores.max(axis=0)
    cls = cls_scores.argmax(axis=0)
    m = scores > conf
    if not m.any():
        return []
    cx, cy, bw, bh = p[0][m], p[1][m], p[2][m], p[3][m]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = nms(boxes, scores[m], iou)
    boxes = _unlb(boxes[keep], meta, None)
    return [{"label": COCO_NAMES[cls[m][k]], "score": float(scores[m][k]),
             "box": [float(v) for v in boxes[i]]} for i, k in enumerate(keep)]


def prep_cls224_nchw(img):
    h, w = img.shape[:2]
    side = 256 if min(h, w) > 256 else min(h, w)
    r = cv2.resize(img, (int(w * side / h), side) if h < w else (side, int(h * side / w)),
                   interpolation=cv2.INTER_AREA)
    ih, iw = r.shape[:2]
    top = (ih - 224) // 2
    left = (iw - 224) // 2
    c = r[max(top, 0):max(top, 0) + 224, max(left, 0):max(left, 0) + 224]
    if c.shape[:2] != (224, 224):
        c = cv2.resize(c, (224, 224))
    rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None]), None


def post_cls(out, topk=3):
    v = np.asarray(out[0])[0]
    v = np.squeeze(v)
    if v.ndim > 1:
        v = v[0]
    e = np.exp(v - v.max())
    pr = e / e.sum()
    idx = pr.argsort()[::-1][:topk]
    return [{"label": LABELS[int(i)], "score": float(pr[i])} for i in idx]


# --- RetinaFace --------------------------------------------------------- #
# amd/retinaface = MobileNet0.25 RetinaFace from biubug6/Pytorch_Retinaface,
# widened to 640x608 by AMD. Three facts that cost real debugging time:
#   1. mean subtraction is BGR order and NOT scaled (x - (104,117,123))
#   2. prior order is stride 8 -> 16 -> 32 (biubug6 PriorBox), NOT insightface's 32/16/8
#   3. the head emits raw logits -> threshold on logit > 0, never on "probability"
RF_W, RF_H = 640, 608
RF_CFG = [(8, (16, 32)), (16, (64, 128)), (32, (256, 512))]
RF_MEAN = np.array([104.0, 117.0, 123.0], np.float32)


def _rf_anchors():
    out = []
    for stride, sizes in RF_CFG:
        fh = int(np.ceil(RF_H / stride))
        fw = int(np.ceil(RF_W / stride))
        for y in range(fh):
            for x in range(fw):
                for s in sizes:
                    out.append(((x + 0.5) * stride, (y + 0.5) * stride, s, s))
    return np.asarray(out, np.float32)


RF_ANCHORS = _rf_anchors()


def prep_retina(img):
    lb, meta = letterbox(img, (RF_H, RF_W))
    x = np.ascontiguousarray((lb.astype(np.float32) - RF_MEAN)[None])
    return x, meta


def post_retina(out, meta, conf=0.5):
    """conf is a *logit* threshold (0.5 -> sigmoid 0.62)."""
    a = RF_ANCHORS
    b = np.asarray(out[0])[0]
    lg = np.asarray(out[1])[0][:, 1]
    lms = np.asarray(out[2])[0] if len(out) > 2 else None
    m = lg > conf
    if not m.any():
        return []
    b, lg, a = b[m], lg[m], a[m]
    cx = b[:, 0] * 0.1 * a[:, 2] + a[:, 0]
    cy = b[:, 1] * 0.1 * a[:, 3] + a[:, 1]
    bw = np.exp(b[:, 2] * 0.2) * a[:, 2]
    bh = np.exp(b[:, 3] * 0.2) * a[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    pts = None
    if lms is not None:
        lms = lms[m]
        px = np.stack([lms[:, i] * 0.1 * a[:, 2] + a[:, 0] for i in (0, 2, 4, 6, 8)], 1)
        py = np.stack([lms[:, i] * 0.1 * a[:, 3] + a[:, 1] for i in (1, 3, 5, 7, 9)], 1)
        pts = np.stack([px, py], 2)
    keep = nms(boxes, lg, 0.35)
    boxes = _unlb(boxes[keep], meta, None)
    res = []
    for i, k in enumerate(keep):
        item = {"label": "face", "score": float(lg[k]),
                "box": [float(v) for v in boxes[i]]}
        if pts is not None:
            p = pts[k].copy()
            s, padx, pady = meta
            p[:, 0] = (p[:, 0] - padx) / s
            p[:, 1] = (p[:, 1] - pady) / s
            item["landmarks"] = [[float(u), float(v)] for u, v in p]
        res.append(item)
    return res


def prep_sr(img):
    """RCAN expects NHWC 56x56 -> produces 112x112 (x2). Square centre crop.
    仅用于小图/缩略路径；整图超分走 sr_tiled()。"""
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    c = img[max(top, 0):max(top, 0) + s, max(left, 0):max(left, 0) + s]
    c = cv2.resize(c, (56, 56), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.ascontiguousarray(rgb[None]), (top, left, s, h, w)


SR_TILE = 56               # RCAN 输入约束：单块固定 56x56 -> 112x112
SR_MAX_SIDE = 1120         # 整图超分前最长边上限（块数 = (H/56)*(W/56)，每块 ~0.65ms）


def sr_tiled(w, frame, npu=True):
    """整图 2x 分块超分：pad 到 56 倍数 -> 逐块 NPU 推理 -> 裁回 pad。
    输出恒为输入 2 倍尺寸，不再有"缩到 56x56 再放大"的模糊小图问题。"""
    h, wd = frame.shape[:2]
    s = SR_MAX_SIDE / float(max(h, wd))
    if s < 1.0:
        frame = cv2.resize(frame, (max(1, int(wd * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        h, wd = frame.shape[:2]
    ph, pw = (-h) % SR_TILE, (-wd) % SR_TILE
    pad = cv2.copyMakeBorder(frame, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    rows, cols = pad.shape[0] // SR_TILE, pad.shape[1] // SR_TILE
    out = np.empty((pad.shape[0] * 2, pad.shape[1] * 2, 3), np.uint8)
    sess = w.sess if npu else w.sess_cpu
    name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    with w.lock:
        for i in range(rows):
            for j in range(cols):
                tile = pad[i * SR_TILE:(i + 1) * SR_TILE,
                           j * SR_TILE:(j + 1) * SR_TILE]
                x = np.ascontiguousarray(
                    (cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32))[None])
                a = sess.run(None, {name: x})[0][0]
                out[i * SR_TILE * 2:(i + 1) * SR_TILE * 2,
                    j * SR_TILE * 2:(j + 1) * SR_TILE * 2] = \
                    cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ms = (time.perf_counter() - t0) * 1000.0
    return out[:h * 2, :wd * 2], ms


def post_sr(out, meta):
    a = np.asarray(out[0])[0]           # (112,112,3)
    a = np.clip(a, 0, 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


PREP = {"detect": prep_yolo, "classify": prep_cls224_nchw,
        "face": prep_retina, "sr": prep_sr}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def _p(*rel):
    return os.path.join(ROOT, *rel)


MODELS = collections.OrderedDict()


def reg(w):
    MODELS[w.key] = w
    return w


reg(Wrap("yolo", "YOLOv8m 目标检测", "COCO 80 类 · 640×640 · 实时框选",
         _p("RyzenAI-SW-1.7.1", "CNN-examples", "object_detection", "yolov8m",
            "models", "yolov8m_XINT8_skip.onnx"), "detect"))
reg(Wrap("resnet50", "ResNet-50 图像分类", "ImageNet 1000 类 · top-3 预测",
         _p("RyzenAI-SW-1.7.1", "CNN-examples", "torchvision_inference",
            "models", "resnet50_XINT8.onnx"), "classify"))
reg(Wrap("retina", "RetinaFace 人脸检测", "人脸框 + 五点关键点",
         _p("hf_models", "RetinaFace_int.onnx"), "face"))
reg(Wrap("mobilenet", "MobileNetV2 轻量分类", "ImageNet 1000 类 · 更快更小",
         _p("hf_models", "mobilenetv2_int8.onnx"), "classify"))
reg(Wrap("rcan", "RCAN 超分辨率", "2× 超分重建",
         _p("models_src", "RCAN_int8_NHWC.onnx"), "sr"))


# MobileNetV2 from HF is NHWC 224 and wants plain 0..1 RGB, patch its prep
def prep_mnv2(img):
    h, w = img.shape[:2]
    side = 256 if min(h, w) > 256 else min(h, w)
    r = cv2.resize(img, (int(w * side / h), side) if h < w else (side, int(h * side / w)),
                   interpolation=cv2.INTER_AREA)
    ih, iw = r.shape[:2]
    top, left = max((ih - 224) // 2, 0), max((iw - 224) // 2, 0)
    c = r[top:top + 224, left:left + 224]
    if c.shape[:2] != (224, 224):
        c = cv2.resize(c, (224, 224))
    rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb[None]), None


_model = MODELS["mobilenet"]
_model.path = _p("hf_models", "mobilenetv2_int8.onnx")


def get(key):
    return MODELS.get(key)


def infer(w, frame, npu=True, conf=0.35):
    """Run one model on a BGR frame. Returns dict with result payload."""
    err = w.load(npu)
    if err:
        return {"ok": False, "error": err}
    # RCAN 整图超分走分块专用路径：输出 2x 大图，而非 112x112 模糊小图
    if w.kind == "sr":
        try:
            res, ms = sr_tiled(w, frame, npu)
            w.note(ms, npu)
            return {"ok": True, "ms": ms, "result": res}
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
    try:
        x, meta = _prep_for(w, frame)
        name = (w.sess if npu else w.sess_cpu).get_inputs()[0].name
        feed = {name: x}
        with w.lock:
            t0 = time.perf_counter()
            out = (w.sess if npu else w.sess_cpu).run(None, feed)
            ms = (time.perf_counter() - t0) * 1000.0
        w.note(ms, npu)
        res = _post_for(w, out, frame, meta, conf)
        return {"ok": True, "ms": ms, "result": res}
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:200])}


def _prep_for(w, frame):
    if w.key == "mobilenet":
        return prep_mnv2(frame)
    if w.key == "yolo":
        return prep_yolo(frame, 640)
    if w.kind == "classify":
        return prep_cls224_nchw(frame)
    if w.kind == "face":
        return prep_retina(frame)
    if w.kind == "sr":
        return prep_sr(frame)
    raise ValueError(w.key)


def _post_for(w, out, frame, meta, conf):
    if w.kind == "detect":
        return post_yolo(out, meta, conf)
    if w.kind == "face":
        return post_retina(out, meta, conf)
    if w.kind == "classify":
        return post_cls(out)
    if w.kind == "sr":
        return post_sr(out, meta)
    return None


def stats(w):
    def avg(d):
        return round(sum(d) / len(d), 2) if d else None
    return {
        "key": w.key, "title": w.title, "kind": w.kind,
        "npu_ms": avg(w.times), "cpu_ms": avg(w.cpu_times),
        "speedup": round(avg(w.cpu_times) / avg(w.times), 2)
        if w.times and w.cpu_times else None,
        "load_sec": round(w.load_sec, 1), "calls": w.calls, "error": w.error,
    }


# --------------------------------------------------------------------------- #
# warm-up: build every NPU session + run one dummy frame at boot, so the first
# real user click never pays the ~40s one-off compile cost
# --------------------------------------------------------------------------- #
WARM = {"total": 0, "done": 0, "current": "", "ready": False, "sec": 0.0,
        "failed": []}


def warmup_all(keys=None):
    t0 = time.time()
    keys = keys or list(MODELS.keys())
    WARM["total"] = len(keys)
    WARM["done"] = 0
    WARM["ready"] = False
    WARM["failed"] = []
    dummy = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(dummy, (140, 120), (460, 360), (70, 120, 200), -1)
    for k in keys:
        w = MODELS[k]
        WARM["current"] = w.title
        try:
            if not w.load(True):
                infer(w, dummy, npu=True, conf=0.7)
        except Exception:
            WARM["failed"].append(k)
        WARM["done"] += 1
    WARM["current"] = ""
    WARM["sec"] = round(time.time() - t0, 1)
    WARM["ready"] = True
