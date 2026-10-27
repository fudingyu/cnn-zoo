#!/usr/bin/env python
"""Fetch / rebuild the FP32 weights this zoo needs.

Weights (~161 MB) are not committed to git. This script rebuilds the FP32 side
from public sources; the NPU runs INT8, so run quantize_all.py afterwards.

    python tools/fetch_models.py --check    # report what is on disk
    python tools/fetch_models.py            # fetch whatever is missing

What this script can obtain
---------------------------
  LRASPP, DeepLabV3    exported locally from torchvision (needs torch)
  MODNet               HuggingFace (Xenova/modnet, fallback DavG25/...)
  AnimeGANv2           HuggingFace (akhaliq/AnimeGANv2-ONNX)

What it cannot
--------------
  YOLOv8m, ResNet-50   ship with the AMD RyzenAI-SW SDK snapshot, not here
  Real-ESRGAN, RCAN, MobileNetV2, RetinaFace
                       third-party redistributions — see README, drop them in
                       the paths listed there and the studio picks them up

HuggingFace is often unreachable from mainland China; set HF_ENDPOINT or use a
proxy if the downloads time out.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FP32 = ROOT / "fp32"
INT8 = ROOT / "int8"

HF = "https://huggingface.co/%s/resolve/main/%s"

# (dest name, [ (repo, file), ... ] in preference order)
HF_JOBS = [
    ("modnet_fp32.onnx", [
        ("Xenova/modnet", "onnx/model.onnx"),
        ("DavG25/modnet-pretrained-models",
         "models/modnet_photographic_portrait_matting.onnx"),
    ]),
    ("animeganv2_fp32.onnx", [
        ("akhaliq/AnimeGANv2-ONNX", "face_paint_512_v2_0.onnx"),
    ]),
]

# torchvision re-exports
TV_JOBS = [
    ("lraspp_mobilenet_v3_large_512.onnx", "lraspp_mobilenet_v3_large", 512),
    ("deeplabv3_mobilenet_v3_large_512.onnx", "deeplabv3_mobilenet_v3_large", 512),
]

# dynamic-batch source -> static [1,C,512,512] derivative
STATIC_JOBS = [
    ("modnet_fp32.onnx", "modnet_fp32_static.onnx", "input"),
    ("animeganv2_fp32.onnx", "animeganv2_fp32_static.onnx", "input_image"),
]

# models the zoo can run, for the --check report
EXPECTED_INT8 = [
    "lraspp_seg_XINT8.onnx",
    "deeplabv3_seg_XINT8.onnx",
    "modnet_matting_XINT8.onnx",
    "animeganv2_XINT8.onnx",
]


def size(p):
    try:
        return p.stat().st_size
    except OSError:
        return 0


def ok(p):
    """A usable ONNX file: exists and is not a 0-byte / HTML-error download."""
    return size(p) > 100_000


def report():
    print("repo     : %s" % ROOT)
    print("fp32/    :")
    for name in [j[0] for j in TV_JOBS] + [j[0] for j in HF_JOBS] + [j[1] for j in STATIC_JOBS]:
        p = FP32 / name
        print("   %-42s %s" % (name, "%8.1f MB" % (size(p) / 1e6) if ok(p) else "MISSING"))
    print("int8/    :")
    for name in EXPECTED_INT8:
        p = INT8 / name
        print("   %-42s %s" % (name, "%8.1f MB" % (size(p) / 1e6) if ok(p) else "MISSING"))
    if not all(ok(INT8 / n) for n in EXPECTED_INT8):
        print("\nint8 weights missing — run:  python quantize_all.py")


def download(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "cnn_zoo/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    # HuggingFace sometimes answers 200 with an error page or an LFS pointer.
    if len(data) < 100_000:
        raise RuntimeError("only %d bytes — probably an LFS pointer or error page" % len(data))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def fetch_hf():
    for dst_name, sources in HF_JOBS:
        dst = FP32 / dst_name
        if ok(dst):
            print("[skip] %s" % dst_name)
            continue
        for repo, path in sources:
            url = HF % (repo, path)
            try:
                print("[get ] %s" % url)
                download(url, dst)
                print("[ ok ] %s  %.1f MB" % (dst_name, size(dst) / 1e6))
                break
            except Exception as e:
                print("[fail] %s: %s: %s" % (repo, type(e).__name__, str(e)[:120]))
        else:
            print("[MISS] %s — download it by hand (see README)" % dst_name)


def export_tv():
    for dst_name, arch, dim in TV_JOBS:
        dst = FP32 / dst_name
        if ok(dst):
            print("[skip] %s" % dst_name)
            continue
        try:
            import torch
            import torchvision
        except ImportError:
            print("[MISS] %s — torch/torchvision not installed" % dst_name)
            continue
        print("[exp ] %s" % dst_name)
        fn = getattr(torchvision.models.segmentation, arch)
        model = fn(weights="DEFAULT").eval()
        x = torch.randn(1, 3, dim, dim)
        with torch.no_grad():
            torch.onnx.export(model, x, str(dst), input_names=["input"], output_names=["out"],
                              opset_version=13, dynamo=False, dynamic_axes=None)
        print("[ ok ] %s  %.1f MB" % (dst_name, size(dst) / 1e6))


def make_static():
    for src_name, dst_name, in_name in STATIC_JOBS:
        src, dst = FP32 / src_name, FP32 / dst_name
        if ok(dst):
            print("[skip] %s" % dst_name)
            continue
        if not ok(src):
            print("[MISS] %s — need %s first" % (dst_name, src_name))
            continue
        try:
            import onnx
            from onnxsim import simplify
        except ImportError:
            print("[MISS] %s — pip install onnx onnxsim" % dst_name)
            continue
        print("[sim ] %s" % dst_name)
        model = onnx.load(str(src))
        simplified, check = simplify(
            model, overwrite_input_shapes={in_name: [1, 3, 512, 512]})
        if not check:
            print("[warn] simplifier produced an invalid graph, keeping the original")
            continue
        onnx.save(simplified, str(dst))
        print("[ ok ] %s  %.1f MB" % (dst_name, size(dst) / 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report what is on disk")
    args = ap.parse_args()

    FP32.mkdir(exist_ok=True)
    INT8.mkdir(exist_ok=True)

    if args.check:
        report()
        return

    export_tv()
    fetch_hf()
    make_static()
    print()
    report()


if __name__ == "__main__":
    main()
