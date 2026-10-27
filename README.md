# cnn_zoo — 在 AMD XDNA1 NPU 上实测趣味 CNN

**NPU (VitisAI EP) vs CPU (INT8, 8 线程) vs iGPU (780M, DirectML) 三方延迟实测，外加一个可交互的 Web 演示台。**

跑在 Ryzen 7 8845HS（Zen4 + RDNA3 780M + XDNA1 "Phoenix" 16 TOPS NPU）上，
RyzenAI-SW 1.7.1 / ONNX Runtime 1.23.3.dev / VitisAIExecutionProvider。

这个仓库想回答一个具体问题：**NPU 到底适合跑什么样的 CNN？** 不是跑分，是拿十来个
有视觉效果的模型（分割、抠图、抠像、动漫化、超分）逐个量化、上板、量延迟，然后看
它们在哪种形状下赢、哪种形状下输，以及为什么。

---

## 一句话结论

> 输入 ≤ 256 的单子图小模型，NPU 比 8 线程 INT8 CPU 快 **3~7 倍**；
> 512×512 的大视觉模型，NPU 反而慢 **1.8~5.6 倍**，本机最快的是 iGPU。
> 根因不是算力，是 **EP 把图切成了碎片**——1 个子图 682 个节点跑 27 ms，
> 同一张图切成 40 个子图，同样的卷积要跑 3.6 秒。

---

## 实测结果

延迟为单帧推理毫秒数（warm 后取 20 次均值），越低越好。加粗 = 该行最快。

### 本轮新测：视觉重负载模型

| 模型 | 输入 | NPU | CPU INT8×8 | FP32 CPU | iGPU 780M | 子图/节点 | 结论 |
|---|---|---|---|---|---|---|---|
| LRASPP 语义分割 | 512 | 27.7 | 35.8 | 59.4 | **22.0** | 1 / 682 | 唯一整图上 NPU 的大模型 |
| DeepLabV3 语义分割 | 512 | 151.1 | 83.6 | 85.4 | **33.4** | 2 / 697 | NPU 慢 1.8×（空洞卷积低效） |
| MODNet 人像抠图 | 512 | 453.9 | 149.9 | 249.3 | **30.1** | 0 / 0 | EP 全回退，NPU 只做搬运 |
| AnimeGANv2 动漫化 | 512 | 3658.4 | 656.4 | 945.3 | **177.7** | 40 / 481 | 慢 5.6×，碎片化典型 |
| Real-ESRGAN ×4 超分 | 256→1024 | 4770.9 | 1114.5 | 2905.2 | **169.2** | 81 / 707 | 慢 4.3×，81 个子图 |

### 对照组：此前已跑通的小模型

| 模型 | 输入 | NPU | CPU INT8×8 | iGPU 780M | 加速比 |
|---|---|---|---|---|---|
| ResNet-50 分类 | 224 | **5.2** | 35.9 | 10.6 | **6.9×** |
| YOLOv8m 检测 | 640 | 48.7 | 220.6 | **44.1** | 4.5× |
| MobileNetV2 分类 | 224 | **2.8** | 12.0 | — | **4.2×** |
| RetinaFace 人脸 | 608×640 | **4.1** | 12.7 | — | 3.1× |
| RCAN ×2 超分 | 56→112 | **0.65** | 3.2 | — | **4.9×** |

完整数据见 [`result/bench.json`](result/bench.json)、[`result/bench_hist.json`](result/bench_hist.json)、
[`result/bench_dml.json`](result/bench_dml.json)，可视化报告见 [`result/report.html`](result/report.html)。

---

## 子图碎片化：真正的性能杀手

VitisAI EP 编译后会把 ONNX 图切成若干「子图」下发给 NPU，切不进去的节点回退 CPU。
**每个子图的边界都是一次 NPU↔CPU 内存搬运**，而搬运开销远大于计算本身。

子图数从 EP 的编译缓存 `context.json` 里数出来（`parse_context.py`）：

| 模型 | 子图数 | 最大子图节点 | NPU 延迟 | 相对 iGPU |
|---|---|---|---|---|
| LRASPP | **1** | 682 | 27.7 ms | 1.3× 慢 |
| DeepLabV3 | 2 | 652 | 151.1 ms | 4.5× 慢 |
| MODNet | **0**（全回退） | — | 453.9 ms | 15× 慢 |
| AnimeGANv2 | **40** | 22 | 3658 ms | 21× 慢 |
| Real-ESRGAN | **81** | 39 | 4771 ms | 28× 慢 |

节点数差不多（481~707），延迟差 170 倍。**这是调度问题，不是算力问题。**

---

## 快速开始

### 前提

- Windows 11 + Ryzen AI 1.7.1 安装包（`C:\Program Files\RyzenAI\1.7.1`，可用 `RYZEN_AI_INSTALL` 覆盖）
- Phoenix 系 NPU（8845HS / 7840HS 等 XDNA1，PCI `VEN_1022&DEV_1502`）
- conda 环境带 AMD 的 onnxruntime（**不是 PyPI 的 `onnxruntime`**，见下方说明）

### 三步跑起来

```bash
# 1. 装依赖（onnxruntime 必须来自 AMD 安装包，见下）
pip install -r requirements.txt

# 2. 拉模型权重（约 161 MB，走 Release，不入 git）
python tools/fetch_models.py

# 3. 起 Web 演示台
cd studio && python zoo_server.py     # 浏览器打开 http://127.0.0.1:8780
```

`studio/index.html` 里能做三件事：选图 → 选模型 → 同时看 NPU / CPU / iGPU 三方结果与加速比。
没有 NPU 的机器上，iGPU 和 CPU 两列仍然可用。

### 关于 onnxruntime

AMD 的 VitisAI EP 只随 RyzenAI 安装包分发，PyPI 上的 `onnxruntime` 不含
`VitisAIExecutionProvider`。装好 RyzenAI 后从它的 wheel 目录安装，或用它自带的
conda 环境；`python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
里必须能看到 `VitisAIExecutionProvider`。

**`numpy` 必须锁 1.26.x** —— 更新版本会和 AMD 的 ORT 构建 ABI 冲突。

---

## 目录结构

```
cnn_zoo/
├── paths.py            统一路径解析（AMD_NPU_HOME / RYZEN_AI_INSTALL）
├── bench.py            NPU vs CPU(INT8×8T) vs FP32 延迟基准
├── bench_dml.py        iGPU 780M (DirectML) 对照
├── quantize_all.py     amd-quark XINT8 量化流水线
├── requant_deeplab.py  MinMSE 重标定 DeepLabV3（修复"全背景"）
├── requant_modnet.py   同上，MODNet
├── verify_mse.py       量化前后 MAE / PSNR 对比
├── infer_visual.py     真实图片推理 → result/*.png
├── make_report.py      汇总 → result/report.html
├── parse_context.py    从 EP 编译缓存统计子图 / 节点数
├── export_tv.py        torchvision → ONNX（opset 13，静态 shape）
├── fix_shapes.py       打印 ONNX 输入输出 shape
├── testimg/            内置样图 5 张
├── fp32/  int8/        权重（.gitignore，用 fetch_models.py 拉）
├── result/             实测 JSON + 效果图 + report.html
├── studio/             Web 演示台
│   ├── zoo_server.py   纯标准库 HTTP，无 web 框架依赖
│   ├── zoo_engine.py   模型注册 + 三 provider 会话管理
│   ├── legacy.py       从 npu_studio 内置的前后处理原语
│   ├── index.html      单页前端
│   └── e2e_test.py     HTTP 端到端回归
└── tools/fetch_models.py
```

### 外部依赖放在哪

`paths.py` 会自己找：优先 `$AMD_NPU_HOME`，其次仓库**同级**的 `RyzenAI-SW-1.7.1/`，
再次仓库**内部**的同名目录。默认布局：

```
<parent>/
  RyzenAI-SW-1.7.1/      AMD SDK（官方模型 yolov8m / resnet50）
  hf_models/             MobileNetV2、RetinaFace
  realsrgan256/          Real-ESRGAN
  models_src/            RCAN
  cnn_zoo/               ← 本仓库
```

---

## 四个模型需要你自备权重

下面四个模型依赖不便再分发的第三方 ONNX，仓库里**不含**它们。缺文件时
`studio` 会自动跳过并列出，补上就自动出现：

| 模型 | 期望路径 | 来源 |
|---|---|---|
| MobileNetV2 | `hf_models/mobilenetv2_int8.onnx` | HuggingFace 上任意 INT8 MobileNetV2 |
| RetinaFace | `hf_models/RetinaFace_int.onnx` | insightface / RetinaFace ONNX 导出，输入 `[1,608,640,3]`（**第一维是 H**） |
| RCAN ×2 | `models_src/RCAN_int8_NHWC.onnx` | RCAN 官方仓库导出，固定 56×56→112×112 |
| Real-ESRGAN ×4 | `realsrgan256/onnx-models/realesrgan_nchw_256x256_{u8s8,fp32}.onnx` | Real-ESRGAN 官方 ONNX |

`yolov8m` / `resnet50` 来自 AMD 官方 SDK 的 `CNN-examples/`，自行 `git clone` RyzenAI-SW 即可。

---

## 复现流程

```bash
python export_tv.py        # 1. torchvision 导出 FP32（若已有 fp32/ 可跳过）
python quantize_all.py     # 2. XINT8 量化 → int8/
python requant_deeplab.py  # 3. DeepLabV3 用 MinMSE 重标定（否则输出全背景）
python bench.py            # 4. NPU vs CPU 延迟 → result/bench.json
MODE=hist python bench.py  #    对照组模型
python bench_dml.py        # 5. iGPU 对照 → result/bench_dml.json
python infer_visual.py     # 6. 效果图 → result/*.png
python parse_context.py    # 7. 子图统计 → result/npu_subgraph.json
python make_report.py      # 8. 汇总 → result/report.html
```

### 量化踩过的坑

`amd-quark` 默认的 `PowerOfTwoMethod.MinMSE` 精度最好但慢得离谱（512 图约 22 s/张量步，
整流程一小时起步）。日常用 `NonOverflow` 或 `MinMax`，**只有这两个模型必须 MinMSE**：

- **DeepLabV3** — 用快速标定会整张图输出「背景」类，换 MinMSE 后与原图 99.1% 一致
- **MODNet** — InstanceNorm + 软上采样，两轮标定都崩（输出近全零），MinMSE 也救不回来，
  **建议直接用 iGPU 版结果**

改法：

```python
from quark.onnx.calibration import PowerOfTwoMethod
quant_config.calibrate_method = PowerOfTwoMethod.NonOverflow   # 快，够用
```

注意 MinMax 只认这个枚举，传字符串 `"MinMax"` 会报错。

---

## 已知限制

- **仅 Windows**。EP 与 xclbin 路径是 Windows 的；Linux 上 RyzenAI 支持矩阵不同。
- **只测过 XDNA1 (Phoenix)**。XDNA2 (Strix) 的 NPU 是 8 列、AIE2P，子图行为和 4x4 xclbin 都不一样。
- **Transformer 一个都上不去**，实测 0%（动态量化 1.9% / 静态 QDQ 44.9% 但慢 13 倍），
  本仓库不含 Transformer 模型。
- **FP32 和 BF16 不支持**。NPU 只吃 INT8（XINT8 布局）。
- **NPU 性能计数器在本机不可用**，所以只有端到端延迟，没有算子级拆分。
- 1x4 与 4x4 xclbin 的取舍：本机 1.7.1 栈**默认先试 4x4**（MobileNetV2 上 4x4 比 1x4 快 40%）。

---

## 许可

代码 MIT（见 [LICENSE](LICENSE)）。模型权重各有其原始许可证，未包含在本仓库中。
AMD、Ryzen、XDNA 是 Advanced Micro Devices, Inc. 的商标，本项目与其无隶属关系。
