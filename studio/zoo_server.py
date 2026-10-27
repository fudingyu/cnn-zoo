# CNN Zoo Studio - local HTTP backend (standard library only)
#
#   python zoo_server.py [--port 8780]
#
# Routes
#   GET  /                     UI
#   GET  /api/models           catalogue + warm-up status
#   GET  /api/samples          sample image list
#   GET  /api/img?name=..      serve a sample image
#   POST /api/infer            {image: b64, model: key, providers: [..]}
#   GET  /api/bench?model=key  live NPU/CPU/iGPU head-to-head (2 warmup + 8 runs)
import os
import io
import sys
import json
import time
import base64
import threading
import traceback
import http.server
import socketserver
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import cv2

import zoo_engine as Z

LOG = os.path.join(HERE, "server.log")
_lf = open(LOG, "a", encoding="utf-8", buffering=1)


def log(*a):
    msg = "[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    _lf.write(msg + "\n")
    try:
        print(msg, flush=True)
    except Exception:
        pass


SAMPLES = [
    {"name": "city", "path": os.path.join(Z.ZOO, "testimg", "city.jpg"),
     "desc": "城市街景 · 适合分割"},
    {"name": "portrait", "path": os.path.join(Z.ZOO, "testimg", "portrait.jpg"),
     "desc": "人像 · 适合抠图/动漫化"},
    {"name": "face", "path": os.path.join(Z.ZOO, "testimg", "face_test.jpg"),
     "desc": "人脸特写 · 适合抠图/人脸"},
    {"name": "retina", "path": os.path.join(Z.ZOO, "testimg", "retina_demo.jpg"),
     "desc": "多人脸 · 适合检测"},
    {"name": "rcan", "path": os.path.join(Z.ZOO, "testimg", "fx_rcan.jpg"),
     "desc": "细节图 · 适合超分"},
]
SAMPLES = [s for s in SAMPLES if os.path.exists(s["path"])]


def _json(obj):
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _b64_png(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CNNZooStudio/1.0"

    def log_message(self, fmt, *args):
        pass

    # -- helpers --------------------------------------------------------- #
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # -- routes ---------------------------------------------------------- #
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/models":
                return self._send(200, _json({
                    "models": Z.catalogue(),
                    "missing": Z.MISSING,
                    "warm": dict(Z.WARM),
                    "ort": __import__("onnxruntime").__version__,
                    "providers": __import__("onnxruntime").get_available_providers(),
                }))
            if u.path == "/api/samples":
                return self._send(200, _json({"samples": [
                    {"name": s["name"], "desc": s["desc"],
                     "url": "/api/img?name=%s" % s["name"]} for s in SAMPLES]}))
            if u.path == "/api/img":
                name = (q.get("name") or [""])[0]
                for s in SAMPLES:
                    if s["name"] == name:
                        with open(s["path"], "rb") as f:
                            return self._send(200, f.read(), "image/jpeg")
                return self._send(404, _json({"error": "no such sample"}))
            if u.path == "/api/bench":
                key = (q.get("model") or [""])[0]
                return self._send(200, _json(bench(key)))
            return self._send(404, _json({"error": "not found"}))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:
            log("GET error:", traceback.format_exc()[-500:])
            try:
                return self._send(500, _json({"error": traceback.format_exc()[-300:]}))
            except Exception:
                return

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path == "/api/infer":
                return self.infer()
            return self._send(404, _json({"error": "not found"}))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:
            log("POST error:", traceback.format_exc()[-500:])
            try:
                return self._send(500, _json({"error": traceback.format_exc()[-300:]}))
            except Exception:
                return

    # -- implementations ------------------------------------------------- #
    def infer(self):
        body = self._read_json()
        key = body.get("model") or ""
        m = Z.MODELS.get(key)
        if m is None:
            return self._send(400, _json({"error": "未知模型 %r，请刷新页面重新选择" % key}))
        b64 = body.get("image") or ""
        if "," in b64[:64]:
            b64 = b64.split(",", 1)[1]
        try:
            data = base64.b64decode(b64, validate=False)
        except Exception as e:
            return self._send(400, _json({"error": "图片 base64 解码失败：%s" % e}))
        if not data or len(data) < 64:
            return self._send(400, _json({"error": "图片数据为空或过短（%d 字节），请重新选择图片" % len(data)}))
        arr = np.frombuffer(data, np.uint8)
        try:
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except cv2.error as e:
            return self._send(400, _json({"error": "图片解码失败：%s" % str(e)[:120]}))
        if frame is None:
            return self._send(400, _json({"error": "无法解码图片（格式不支持？）"}))
        want = [p for p in body.get("providers", []) if p in Z._PROVIDERS]
        if not want:
            want = [p for p in Z._PROVIDERS if m.has(p)]
        results = {}
        for p in want:
            try:
                res = Z.render(m, frame, p)
                item = {"ok": True, "ms": res.get("ms"), "image": _b64_png(res["image"]),
                        "note": res.get("note", "")}
                if m.warn.get(p):
                    item["warn"] = m.warn[p]
                if res.get("top"):
                    item["top"] = res["top"]
                if res.get("dets") is not None:
                    item["count"] = len(res["dets"])
                if res.get("faces") is not None:
                    item["count"] = len(res["faces"])
                results[p] = item
            except Exception as e:
                results[p] = {"ok": False,
                              "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
        best = None
        oks = [r["ms"] for r in results.values() if r.get("ok")]
        if len(oks) > 1:
            best = min(oks)
        for r in results.values():
            if r.get("ok") and best:
                r["vs_best"] = round(best / r["ms"], 2)
        return self._send(200, _json({"model": key, "results": results,
                                      "orig": _b64_png(frame)}))


def bench(key, warmup=2, runs=8):
    m = Z.MODELS.get(key)
    if m is None:
        return {"error": "unknown model"}
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(frame, (100, 100), (400, 380), (60, 120, 200), -1)
    cv2.circle(frame, (320, 240), 80, (120, 200, 255), -1)
    out = {"model": key, "title": m.title, "results": {}}
    for p in Z._PROVIDERS:
        if not m.has(p):
            continue
        try:
            # DML 首次调用要 JIT 编译 kernel，多给几次热手
            nw = 5 if p == "igpu" else warmup
            for _ in range(nw):
                Z.render(m, frame, p)
            ts = []
            for _ in range(runs):
                ts.append(Z.render(m, frame, p)["ms"])
            out["results"][p] = {"ms": round(sum(ts) / len(ts), 2),
                                 "min": round(min(ts), 2), "n": len(ts)}
        except Exception as e:
            out["results"][p] = {"error": str(e)[:160]}
    r = out["results"]
    if r.get("npu", {}).get("ms") and r.get("cpu", {}).get("ms"):
        out["speedup_npu_vs_cpu"] = round(r["cpu"]["ms"] / r["npu"]["ms"], 2)
    if r.get("npu", {}).get("ms") and r.get("igpu", {}).get("ms"):
        out["speedup_npu_vs_igpu"] = round(r["igpu"]["ms"] / r["npu"]["ms"], 2)
    if r.get("igpu", {}).get("ms") and r.get("cpu", {}).get("ms"):
        out["speedup_igpu_vs_cpu"] = round(r["cpu"]["ms"] / r["igpu"]["ms"], 2)
    return out


# --------------------------------------------------------------------------- #
class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _warm_bg():
    try:
        log("warming up NPU sessions ...")
        Z.warmup_all()
        log("warm-up done in %.1fs%s" % (Z.WARM["sec"],
            ("  failed=%s" % Z.WARM["failed"]) if Z.WARM["failed"] else ""))
    except Exception:
        log("warm-up error:", traceback.format_exc()[-300:])


def _port_in_use(port):
    """True if something is already serving on 127.0.0.1:port."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def main():
    port = 8780
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
    # 防双实例：Windows 下 SO_REUSEADDR 会让两个进程同时"监听"同一端口，
    # 连接在两者间漂移，造成间歇性怪错误。已有实例在跑就直接退出。
    if _port_in_use(port):
        log("=" * 62)
        log("端口 %d 已有实例在运行，本次启动取消。" % port)
        log("直接浏览器打开 http://127.0.0.1:%d 即可；如需重启请先关掉旧窗口。" % port)
        log("=" * 62)
        try:
            import webbrowser
            webbrowser.open("http://127.0.0.1:%d" % port)
        except Exception:
            pass
        return
    threading.Thread(target=_warm_bg, daemon=True).start()
    srv = Server(("127.0.0.1", port), Handler)
    log("=" * 62)
    log("CNN Zoo Studio  ->  http://127.0.0.1:%d" % port)
    log("ORT %s" % __import__("onnxruntime").__version__)
    log("=" * 62)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        log("server stopped")


if __name__ == "__main__":
    main()
