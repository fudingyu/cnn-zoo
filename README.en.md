# cnn_zoo — Fun CNNs benchmarked on an AMD XDNA1 NPU

**NPU (VitisAI EP) vs CPU (INT8, 8 threads) vs iGPU (780M, DirectML), measured end to end — plus an interactive web studio.**

Measured on a Ryzen 7 8845HS (Zen4 + RDNA3 780M + XDNA1 "Phoenix" 16 TOPS NPU),
RyzenAI-SW 1.7.1 / ONNX Runtime 1.23.3.dev / VitisAIExecutionProvider.

The repo answers one narrow question: **what kind of CNN is the NPU actually good at?**
Not a leaderboard — a dozen visually interesting models (segmentation, matting,
stylisation, super-resolution), each quantised, pushed onto the hardware, and timed,
to see which input shapes win, which lose, and why.

[中文](README.md)

---

## TL;DR

> Models with inputs ≤ 256 in a single subgraph run **3–7× faster** on the NPU than
> on 8-thread INT8 CPU. 512×512 vision models run **1.8–5.6× slower**, and the
> fastest engine on this machine is the iGPU. The cause is not compute — the EP
> **shatters the graph**: one subgraph of 682 nodes takes 27 ms, the same convolution
> stack split into 40 subgraphs takes 3.6 seconds.

---

## Results

Single-frame latency in milliseconds, mean of 20 runs after warm-up. Lower is better.
**Bold** = fastest in that row.

### New in this round: heavy vision models

| Model | Input | NPU | CPU INT8×8 | FP32 CPU | iGPU 780M | Subgraphs/nodes | Verdict |
|---|---|---|---|---|---|---|---|
| LRASPP segmentation | 512 | 27.7 | 35.8 | 59.4 | **22.0** | 1 / 682 | the only large model fully on NPU |
| DeepLabV3 segmentation | 512 | 151.1 | 83.6 | 85.4 | **33.4** | 2 / 697 | 1.8× slower than CPU (dilated convs) |
| MODNet portrait matting | 512 | 453.9 | 149.9 | 249.3 | **30.1** | 0 / 0 | full fallback, NPU only copies memory |
| AnimeGANv2 stylisation | 512 | 3658.4 | 656.4 | 945.3 | **177.7** | 40 / 481 | 5.6× slower, textbook fragmentation |
| Real-ESRGAN ×4 upscale | 256→1024 | 4770.9 | 1114.5 | 2905.2 | **169.2** | 81 / 707 | 4.3× slower, 81 subgraphs |

### Baseline: small models that already worked

| Model | Input | NPU | CPU INT8×8 | iGPU 780M | Speed-up |
|---|---|---|---|---|---|
| ResNet-50 classify | 224 | **5.2** | 35.9 | 10.6 | **6.9×** |
| YOLOv8m detect | 640 | 48.7 | 220.6 | **44.1** | 4.5× |
| MobileNetV2 classify | 224 | **2.8** | 12.0 | — | **4.2×** |
| RetinaFace faces | 608×640 | **4.1** | 12.7 | — | 3.1× |
| RCAN ×2 upscale | 56→112 | **0.65** | 3.2 | — | **4.9×** |

Raw data: [`result/bench.json`](result/bench.json), [`result/bench_hist.json`](result/bench_hist.json),
[`result/bench_dml.json`](result/bench_dml.json). Visual report: [`result/report.html`](result/report.html).

---

## Subgraph fragmentation is the real bottleneck

After compilation the VitisAI EP slices the ONNX graph into subgraphs for the NPU;
anything it cannot claim falls back to CPU. **Every subgraph boundary is a
NPU↔CPU memory round-trip**, and that transfer costs more than the arithmetic itself.

Subgraph counts come straight out of the EP's compile cache (`parse_context.py`):

| Model | Subgraphs | Largest subgraph | NPU latency | vs iGPU |
|---|---|---|---|---|
| LRASPP | **1** | 682 | 27.7 ms | 1.3× slower |
| DeepLabV3 | 2 | 652 | 151.1 ms | 4.5× slower |
| MODNet | **0** (all CPU) | — | 453.9 ms | 15× slower |
| AnimeGANv2 | **40** | 22 | 3658 ms | 21× slower |
| Real-ESRGAN | **81** | 39 | 4771 ms | 28× slower |

The node counts are comparable (481–707); the latencies differ by 170×.
**This is a scheduling problem, not a compute problem.**

---

## Quick start

### Requirements

- Windows 11 + Ryzen AI 1.7.1 installed at `C:\Program Files\RyzenAI\1.7.1`
  (override with `RYZEN_AI_INSTALL`)
- A Phoenix-class NPU (8845HS / 7840HS and similar XDNA1 parts, PCI `VEN_1022&DEV_1502`)
- A Python env with **AMD's** onnxruntime — not the PyPI one, see below

### Three steps

```bash
# 1. dependencies (onnxruntime must come from the AMD installer)
pip install -r requirements.txt

# 2. fetch weights, ~161 MB, not in git
python tools/fetch_models.py

# 3. start the studio
cd studio && python zoo_server.py     # open http://127.0.0.1:8780
```

The studio does three things: pick an image → pick a model → compare NPU / CPU / iGPU
side by side with speed-ups. The iGPU and CPU columns still work on machines without
an NPU.

### About onnxruntime

AMD's VitisAI EP ships only inside the RyzenAI installer; the PyPI `onnxruntime`
has no `VitisAIExecutionProvider`. Install from the wheel bundled with RyzenAI, or
use its conda env. Check with:

```bash
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**`numpy` must stay on 1.26.x** — newer releases break the ABI of AMD's ORT build.

---

## Layout

```
cnn_zoo/
├── paths.py            path resolution (AMD_NPU_HOME / RYZEN_AI_INSTALL)
├── bench.py            NPU vs CPU(INT8×8T) vs FP32 latency harness
├── bench_dml.py        iGPU 780M (DirectML) comparison
├── quantize_all.py     amd-quark XINT8 pipeline
├── requant_deeplab.py  MinMSE re-calibration for DeepLabV3 (fixes all-background)
├── requant_modnet.py   same for MODNet
├── verify_mse.py       MAE / PSNR before vs after quantization
├── infer_visual.py     real-image inference → result/*.png
├── make_report.py      aggregate → result/report.html
├── parse_context.py    subgraph / node counts from the EP compile cache
├── export_tv.py        torchvision → ONNX (opset 13, static shapes)
├── fix_shapes.py       dump ONNX input/output shapes
├── testimg/            5 bundled sample images
├── fp32/  int8/        weights (.gitignore'd, fetch with tools/fetch_models.py)
├── result/             measured JSON + effect images + report.html
├── studio/             web studio
│   ├── zoo_server.py   stdlib-only HTTP, no web framework
│   ├── zoo_engine.py   model registry + three-provider session management
│   ├── legacy.py       vendored pre/post primitives
│   ├── index.html      single-page UI
│   └── e2e_test.py     HTTP end-to-end regression
└── tools/fetch_models.py
```

### Where the external pieces go

`paths.py` resolves them automatically: `$AMD_NPU_HOME` first, then a
`RyzenAI-SW-1.7.1/` **beside** the repo, then one **inside** it. Default layout:

```
<parent>/
  RyzenAI-SW-1.7.1/      AMD SDK (official yolov8m / resnet50 models)
  hf_models/             MobileNetV2, RetinaFace
  realsrgan256/          Real-ESRGAN
  models_src/            RCAN
  cnn_zoo/               ← this repo
```

---

## Four models need weights you supply

The four models below depend on third-party ONNX files this repo cannot redistribute.
When the files are absent the studio **skips them and lists them** instead of showing
dead cards; drop them in and they reappear automatically.

| Model | Expected path | Source |
|---|---|---|
| MobileNetV2 | `hf_models/mobilenetv2_int8.onnx` | any INT8 MobileNetV2 on HuggingFace |
| RetinaFace | `hf_models/RetinaFace_int.onnx` | insightface / RetinaFace ONNX export, input `[1,608,640,3]` (**first dim is H**) |
| RCAN ×2 | `models_src/RCAN_int8_NHWC.onnx` | RCAN official repo export, fixed 56×56→112×112 |
| Real-ESRGAN ×4 | `realsrgan256/onnx-models/realesrgan_nchw_256x256_{u8s8,fp32}.onnx` | Real-ESRGAN official ONNX |

`yolov8m` and `resnet50` come from the AMD SDK's `CNN-examples/`; clone RyzenAI-SW yourself.

---

## Reproducing

```bash
python export_tv.py        # 1. export FP32 from torchvision (skip if fp32/ is present)
python quantize_all.py     # 2. XINT8 quantization → int8/
python requant_deeplab.py  # 3. MinMSE re-cal for DeepLabV3 (otherwise all-background)
python bench.py            # 4. NPU vs CPU latency → result/bench.json
MODE=hist python bench.py  #    baseline models
python bench_dml.py        # 5. iGPU comparison → result/bench_dml.json
python infer_visual.py     # 6. effect images → result/*.png
python parse_context.py    # 7. subgraph stats → result/npu_subgraph.json
python make_report.py      # 8. aggregate → result/report.html
```

### Quantization pitfalls we hit

`amd-quark`'s default `PowerOfTwoMethod.MinMSE` gives the best accuracy but is
brutally slow (~22 s per tensor-step on 512² images, an hour per model). Use
`NonOverflow` or `MinMax` day to day — **except** for these two, which need MinMSE:

- **DeepLabV3** — fast calibration collapses the output to a single "background"
  class everywhere; MinMSE restores 99.1% agreement with FP32
- **MODNet** — InstanceNorm plus soft upsampling; two rounds of calibration both
  failed (near-zero output) and MinMSE does not rescue it either.
  **Trust the iGPU result for this one.**

```python
from quark.onnx.calibration import PowerOfTwoMethod
quant_config.calibrate_method = PowerOfTwoMethod.NonOverflow   # fast, good enough
```

`calibrate_method` accepts only that enum — passing the string `"MinMax"` raises.

---

## Known limits

- **Windows only.** The EP and xclbin paths are Windows; RyzenAI's Linux support matrix differs.
- **XDNA1 (Phoenix) only.** XDNA2 (Strix) has an 8-column AIE2P NPU and different
  subgraph behaviour from the 4x4 xclbin used here.
- **Zero Transformers make it onto the NPU** — measured 0% (dynamic quantization
  1.9%; static QDQ 44.9% but 13× slower). No Transformer models included.
- **No FP32, no BF16.** The NPU only takes INT8 (XINT8 layout).
- **No NPU performance counters on this machine**, so latencies are end-to-end with
  no per-operator breakdown.
- 1x4 vs 4x4 xclbin: on the 1.7.1 stack, **try 4x4 first** (MobileNetV2 is 40%
  faster than 1x4, the opposite of the 1.2.0-era rule for small models).

---

## License

Code is MIT (see [LICENSE](LICENSE)). Model weights carry their own licenses and are
not distributed here. AMD, Ryzen and XDNA are trademarks of Advanced Micro Devices,
Inc.; this project is not affiliated with or endorsed by AMD.
