"""验证 MinMSE 版 INT8 模型的视觉质量（MODNet / DeepLabV3），与 FP32 对比。"""
import os, sys, json, ctypes
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent
RES = ROOT / "result"
BASE = ROOT / "testimg"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAL = np.zeros((256, 3), dtype=np.uint8)
for i in range(256):
    PAL[i] = [(i >> 0 & 1) * 128 + (i >> 3 & 1) * 64 + (i >> 6 & 1) * 32,
              (i >> 1 & 1) * 128 + (i >> 4 & 1) * 64 + (i >> 7 & 1) * 32,
              (i >> 2 & 1) * 128 + (i >> 5 & 1) * 64 + (i >> 8 & 1) * 32]

out = {}
# ---------- MODNet ----------
img = Image.open(BASE / "face_test.jpg").convert("RGB").resize((512, 512), Image.BILINEAR)
x = np.asarray(img, np.float32) / 255.0
feed = ((x * 2 - 1).transpose(2, 0, 1)[None]).astype(np.float32)
s32 = ort.InferenceSession(str(ROOT / "fp32" / "modnet_fp32_static.onnx"), providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession(str(ROOT / "int8" / "modnet_matting_XINT8_MSE.onnx"), providers=["CPUExecutionProvider"])
o32 = np.squeeze(s32.run(None, {"input": feed})[0])
o8 = np.squeeze(s8.run(None, {"input": feed})[0])
d = np.abs(o32 - o8)
mse = float((d ** 2).mean())
rec = dict(mae=round(float(d.mean()), 3), psnr=round(10 * np.log10(1 / max(mse, 1e-9)), 2),
           range8=[float(o8.min()), float(o8.max())])
out["modnet_matting_mse"] = rec
print("modnet MSE:", rec)
u8 = (np.clip(o8, 0, 1) * 255).astype(np.uint8)
bg = np.zeros((512, 512, 3), np.uint8); bg[::64, ::32] = 220; bg[::64, 16::32] = 160
comp = (x * 255)[..., ::-1] * o8[..., None] + bg * (1 - o8[..., None])
Image.fromarray(comp.astype(np.uint8)[..., ::-1]).save(str(RES / "modnet_matting_npu_compose.png"))
Image.fromarray(u8).save(str(RES / "modnet_matting_npu_alpha.png"))

# ---------- DeepLabV3 ----------
x2 = np.asarray(img, np.float32) / 255.0
feed2 = (((x2 - MEAN) / STD).transpose(2, 0, 1)[None]).astype(np.float32)
s32 = ort.InferenceSession(str(ROOT / "fp32" / "deeplabv3_mobilenet_v3_large_512.onnx"), providers=["CPUExecutionProvider"])
s8 = ort.InferenceSession(str(ROOT / "int8" / "deeplabv3_seg_XINT8_MSE.onnx"), providers=["CPUExecutionProvider"])
m32 = s32.run(None, {"input": feed2})[0][0].argmax(0)
m8 = s8.run(None, {"input": feed2})[0][0].argmax(0)
agree = float((m32 == m8).mean())
rec2 = dict(agree=round(agree, 4),
            cls_fp32=sorted(set(m32.reshape(-1).tolist())), cls_int8=sorted(set(m8.reshape(-1).tolist())))
out["deeplabv3_mse"] = rec2
print("deeplabv3 MSE:", rec2)
orig = (x2 * 255).astype(np.uint8)
Image.fromarray(PAL[m8]).save(str(RES / "deeplabv3_seg_npu_mask.png"))
Image.fromarray((orig * 0.45 + PAL[m8] * 0.55).astype(np.uint8)).save(str(RES / "deeplabv3_seg_npu_overlay.png"))

(RES / "mse_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("saved mse_check.json")
