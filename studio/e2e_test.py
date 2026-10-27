"""End-to-end HTTP test: POST /api/infer with a real image for every model
on every provider, verify JSON payload shape + latency sanity."""
import sys, os, json, time, base64
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.dirname(HERE)

BASE = "http://127.0.0.1:8780"
OUT = open(os.path.join(HERE, "e2e.txt"), "w", encoding="utf-8", buffering=1)

def w(*a):
    OUT.write(" ".join(str(x) for x in a) + "\n")
    OUT.flush()

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def post(path, obj, timeout=300):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

# wait for warm-up
for _ in range(60):
    m = get("/api/models")
    if m.get("warm", {}).get("ready"):
        break
    time.sleep(2)
w("warm:", json.dumps(m["warm"], ensure_ascii=False))
w("models with bench:")
for x in m["models"]:
    b = x["bench"]
    w("  %-16s npu=%-8s cpu=%-8s dml=%-8s sub=%s prov=%s" % (
        x["key"], b["npu"], b["cpu"], b["dml"], b["subgraphs"], x["providers"]))

# build a small test image (city scene) as base64 jpeg
import cv2
import numpy as np
img = cv2.imread(os.path.join(ZOO, "testimg", "city.jpg"))
ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
w("test image:", img.shape, len(b64), "b64 chars")

for key in ["lraspp_seg", "modnet_matting", "animeganv2", "yolov8m", "rcan_sr"]:
    w("--- infer %s (all providers) ---" % key)
    t0 = time.time()
    r = post("/api/infer", {"image": b64, "model": key, "providers": ["npu", "cpu", "igpu"]})
    for p, item in (r.get("results") or {}).items():
        if item.get("ok"):
            w("    %-5s ok ms=%-9s img=%dB vs_best=%s note=%s" % (
                p, item["ms"], len(item.get("image", "")), item.get("vs_best"), item.get("note", "")))
        else:
            w("    %-5s FAIL %s" % (p, str(item.get("error"))[:120]))
    w("    wall=%.1fs orig=%dB" % (time.time() - t0, len(r.get("orig", ""))))

w("== bench endpoint ==")
for key in ["lraspp_seg", "yolov8m"]:
    r = get("/api/bench?model=" + key)
    w("  %s -> %s" % (key, json.dumps(r.get("results", {}), ensure_ascii=False)))
    w("     speedups: %s %s %s" % (r.get("speedup_npu_vs_cpu"),
                                   r.get("speedup_npu_vs_igpu"),
                                   r.get("speedup_igpu_vs_cpu")))

w("== done ==")
OUT.close()
