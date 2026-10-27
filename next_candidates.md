# 下一批 CNN 候选模型清单（供审阅）

生成时间：2026-10-15
硬件：Ryzen 7 8845HS · XDNA1 Phoenix NPU · RyzenAI-SW 1.7.1
状态：**仅列候选，未跑配置，未下载，未量化**

---

## 筛选原则（沿用前一批结论）

- ✅ 纯 CNN（无 Transformer / Attention / GAN 大循环）
- ✅ 输入 ≤ 320×320（大视觉模型 NPU 反而慢）
- ✅ INT8 量化不掉太多精度（避开 MODNet 类软激活 + InstanceNorm 陷阱）
- ✅ 有现成 ONNX 或 torchvision / PyTorch 轻量可导出
- ✅ 视觉输出**有戏**——不是纯数字分类那种"看得到结果但看不见效果"的
- ❌ 不重复已测：YOLOv8m / ResNet50 / MobileNetV2 / RetinaFace / RCAN / LRASPP / DeepLabV3 / MODNet / AnimeGANv2 / RealESRGAN
- ❌ 不碰：Transformer、GAN 大模型（AnimeGANv2 类已证实 NPU 慢）、BF16、动态量化

---

## A 组 · 视觉奇观（视觉冲击最强，主打好玩）

### A1. CNN 风格迁移（Johnson 2016 / Fast Style Transfer）
- **输入**：256×256 content + 256×256 style → 256×256 风格化输出
- **参数**：~10M（15 层 conv + BN）
- **效果**：一张街景 → 梵高星空 / 莫奈睡莲 / 毕加索立体派
- **NPU 预判**：✅ 纯卷积 + 256 输入，INT8 化良好，应快于 8 线程 CPU
- **量化难度**：中（BN 融合后 INT8 可行）
- **来源**：PyTorch `torchvision.models` 无；Keras-Applications `fast_neural_style`；或 PyTorch Lightning 的 `fast-neural-style`
- **有趣度**：⭐⭐⭐⭐⭐

### A2. 老照片上色 / 黑白转彩色 CNN（Palette Tiny / DeOldify Tiny）
- **输入**：256×256 灰度 → 256×256 RGB
- **参数**：~3M
- **效果**：爷爷奶奶的黑白全家福一键上色
- **NPU 预判**：✅ 小型 ResNet-ish 编码器 + 解码器
- **量化难度**：低
- **来源**：HuggingFace `adobe/DeOldify`（Tiny 版 ONNX 有）、GitHub `adobe/DeOldify`
- **有趣度**：⭐⭐⭐⭐⭐

### A3. 人脸微笑强度回归 CNN
- **输入**：224×224 人脸 → 单值 0.0~1.0 微笑度
- **参数**：~3M（ResNet18 骨干 + 1 输出头）
- **效果**：照片扫脸弹"微笑 78%"分数条，可配 RetinaFace 做联调 demo
- **NPU 预判**：✅ 极快（预测单值）
- **量化难度**：低（回归头需 careful calib）
- **来源**：Kaggle FER+ smile 子集训练脚本；或 `deepface` 库 Smile 模型
- **有趣度**：⭐⭐⭐⭐（配 RetinaFace 联调可做成"表情直播"）

### A4. 人脸 68/106 关键点回归 CNN
- **输入**：200×200 人脸 → 68×2 或 106×2 坐标
- **参数**：~1M（轻量 HRNet 或 ResNet18）
- **效果**：AR 妆容、虚拟形象驱动、表情动画
- **NPU 预判**：✅ 极快（回归任务）
- **量化难度**：低
- **来源**：`dlib` 68 点模型 ONNX 化；GitHub `1adrianb/face-alignment` ONNX 版
- **有趣度**：⭐⭐⭐⭐（与 RetinaFace 组成完整人脸分析流水线）

### A5. 图像旋转/方向分类 CNN（EXIF 修复器）
- **输入**：224×224 → 4 类（0°/90°/180°/270°）
- **参数**：~0.5M（MobileNet-v2 或 ResNet18）
- **效果**：手机横拍歪斜的照片一键扶正
- **NPU 预判**：✅ 极快
- **量化难度**：低
- **来源**：GitHub `julesdv/dataset-image-rotation` 训练脚本；或 ImageNet 旋转子集
- **有趣度**：⭐⭐⭐（实用小工具）

### A6. Deepfake 检测 CNN
- **输入**：224×224 人脸 → 真人/伪造概率
- **参数**：~13M（ResNet50 微调）
- **效果**：照片扫脸判"AI 生成 92%"
- **NPU 预判**：✅ 纯 ResNet50 分类器
- **量化难度**：中（对细粒度特征敏感）
- **来源**：HuggingFace `deepfake-detection` 类仓库；Kaggle `deepfake-detection-challenge` baseline ONNX
- **有趣度**：⭐⭐⭐⭐（话题性强）

---

## B 组 · 人脸/生物识别（能与已有 RetinaFace 组成联调）

### B1. ArcFace 人脸比对 CNN ⭐推荐
- **输入**：112×112 人脸 → 512 维嵌入向量
- **参数**：~13M（ResNet50 骨干）
- **效果**：两张脸相似度打分（余弦），做人脸识别锁屏、双胞胎识别 demo
- **NPU 预判**：✅ 极快（112 输入 + INT8 分类骨干，NPU 秒杀）
- **量化难度**：中（ArcFace 对量化敏感，推荐 Quark 静态 INT8 + cosine metric 校准）
- **来源**：InsightFace 官方 `w600k_r50.onnx`（现成 ONNX！）
- **有趣度**：⭐⭐⭐⭐⭐（配 RetinaFace 组成完整人脸识别，可写一个"人脸锁屏"demo）

### B2. 年龄/性别回归 CNN
- **输入**：112×112 人脸 → 年龄（0~100）+ 性别概率
- **参数**：~5M
- **效果**：照片扫脸报"男/女，约 32 岁"
- **NPU 预判**：✅ 极快
- **量化难度**：低
- **来源**：GitHub `GMASS`；或 InsightFace 附带年龄性别模型
- **有趣度**：⭐⭐⭐

### B3. 表情分类 CNN（FER+ 7 类）
- **输入**：96×96 人脸 → 7 类（快乐/悲伤/愤怒/恐惧/惊讶/厌恶/中性）
- **参数**：~2M
- **效果**：情绪温度计，输出条形图
- **NPU 预判**：✅ 极快（小输入小模型）
- **量化难度**：低
- **来源**：Kaggle FER+ 数据集 + ResNet18 训练脚本
- **有趣度**：⭐⭐⭐

### B4. ArcFace + RetinaFace 人脸识别全套（组合件）
- **说明**：A/B/C/D 组合成一个"人脸录入 → 匹配 → 显示相似度"小工具
- **NPU 预判**：✅ 全 NPU 跑
- **有趣度**：⭐⭐⭐⭐⭐（可作为 NPU 能力的旗舰 demo）

---

## C 组 · OCR / 字符识别（NPU 极快，可视化清晰）

### C1. MNIST 手写数字 CNN
- **输入**：28×28 → 10 类
- **参数**：~100K
- **效果**：键盘画数字 → 实时识别
- **NPU 预判**：✅ 极小极快
- **量化难度**：极低
- **来源**：PyTorch `torch.utils.data.MNIST` + 3 层 conv 训练脚本
- **有趣度**：⭐⭐⭐（经典玩具，NPU 上"0.1ms 推理"很炫）

### C2. EMNIST 手写字符 CNN（62 类）
- **输入**：28×28 → 0-9 + a-z + A-Z
- **参数**：~500K
- **效果**：手写英文识别
- **NPU 预判**：✅ 极快
- **量化难度**：低
- **来源**：Keras-EMNIST / PyTorch EMNIST
- **有趣度**：⭐⭐⭐

### C3. 车牌检测 + 号码识别 CNN
- **输入**：640×640 → 车牌框 + 号码字符
- **参数**：YOLOv8n 微调 ~3M + CRNN 号码 ~10M
- **效果**：车流量统计、车牌抓拍
- **NPU 预判**：✅ YOLOv8n INT8 有基础（已测 v8m）
- **量化难度**：中
- **来源**：GitHub `KahingChow/LPNR`；`sifraos/plate-recognition`
- **有趣度**：⭐⭐⭐⭐（实用性 + NPU 检测能力展示）

---

## D 组 · 图像质量/修复（视觉对照性强）

### D1. 图像去雾 CNN（DehazeNet 轻量）
- **输入**：256×256 → 256×256 去雾
- **参数**：~5M（ResNet-ish 编解码）
- **效果**：雾霾照片一键透视
- **NPU 预判**：✅ 256 输入 + 小型编解码
- **量化难度**：低
- **来源**：GitHub `yuan1203/dehaze`（PyTorch 官方）
- **有趣度**：⭐⭐⭐⭐（前后对比极明显）

### D2. 图像去雨 CNN（RRRN 轻量）
- **输入**：256×256 → 256×256 去雨
- **参数**：~5M
- **效果**：雨滴照片恢复
- **NPU 预判**：✅ 可行
- **量化难度**：低
- **来源**：GitHub `ZhangPeng-L/RRRN`
- **有趣度**：⭐⭐⭐⭐

### D3. 图像去噪 CNN（DnCNN-Small）
- **输入**：128×128 patch → 128×128 去噪
- **参数**：~0.6M（17 层 conv）
- **效果**：高 ISO 噪点照片恢复
- **NPU 预判**：✅ 极快（patch 处理）
- **量化难度**：极低
- **来源**：GitHub `cszn/DnCNN`（官方 PyTorch）
- **有趣度**：⭐⭐⭐⭐（手机夜景照片去噪，日常实用）

### D4. 图像质量评分 CNN（NIQE 深度版）
- **输入**：128×128 patch → 单值质量分
- **参数**：~1M
- **效果**：照片质量评分器，1-100 分
- **NPU 预判**：✅ 极快
- **量化难度**：低
- **来源**：GitHub `Jih-Yuan-Chang/Deep-Image-Perceptual-Quality-Assessment`
- **有趣度**：⭐⭐⭐

---

## E 组 · 分类类（有趣但视觉效果一般，作为 NPU 分类能力补充）

### E1. 场景分类 CNN（Places365 蒸馏版）
- **输入**：224×224 → 365 类场景（咖啡馆/海滩/厨房…）
- **参数**：~20M（Distilled ResNet）
- **效果**：照片自动分类到生活场景
- **NPU 预判**：✅ ResNet 类 NPU 已验证
- **量化难度**：中（365 类 logit 对量化敏感）
- **来源**：HuggingFace `places365-distilled-resnet`；官方 ` Places365` 数据集
- **有趣度**：⭐⭐⭐

### E2. 艺术流派/画家分类 CNN
- **输入**：224×224 → 100+ 画家（Van Gogh / Monet / Picasso…）
- **参数**：~15M
- **效果**：照片识别艺术家风格，与 A1 风格迁移配对做"识别→模拟"闭环
- **NPU 预判**：✅ 可行
- **量化难度**：中
- **来源**：Kaggle `paintings-100-artist` 训练脚本；或 `stanford-cs499` 项目
- **有趣度**：⭐⭐⭐⭐（与 A1 组成"识别-生成"闭环 demo）

### E3. 植物识别 CNN（PlantVillage 38 类）
- **输入**：224×224 → 38 类（健康番茄/患叶斑病苹果…）
- **参数**：~5M
- **效果**：拍叶子认植物 + 病害诊断
- **NPU 预判**：✅ 可行
- **量化难度**：低
- **来源**：GitHub `spMohammed/Plant-Disease-Detection` 官方 ONNX
- **有趣度**：⭐⭐⭐⭐（垂直领域，有实用价值）

### E4. 花卉分类 CNN（Oxford Flowers 102 / 5 类简化）
- **输入**：224×224 → 5~102 类花
- **参数**：~4M
- **效果**：拍照识花
- **NPU 预判**：✅ 可行
- **量化难度**：低
- **来源**：Keras `Flowers` 数据集 + MobileNetV2 微调
- **有趣度**：⭐⭐⭐

---

## F 组 · 其他有意思的小 CNN

### F1. 二维码/条形码角点检测 CNN
- **输入**：320×320 → 4 个角点坐标
- **参数**：~2M
- **效果**：比 OpenCV `detectQRCode` 更快更准，模糊图也能识
- **NPU 预判**：✅ 极快（小目标回归）
- **量化难度**：低
- **来源**：GitHub `kornia` 库或 `OpenCV QRCodeDetectorAruco`
- **有趣度**：⭐⭐⭐（工程实用）

### F2. 图像鉴伪水印检测 CNN（SteganoDetect）
- **输入**：128×128 块 → 有/无隐形水印概率
- **参数**：~1M
- **效果**：判断照片是否被 AIGC 工具处理过
- **NPU 预判**：✅ 极快
- **量化难度**：低
- **来源**：Kaggle `digital-watermark-detection` 基线
- **有趣度**：⭐⭐⭐⭐（与 A6 Deepfake 组成"AI 生成检测"套件）

### F3. 漫画脸 CNN（轻量版 ToonApp，非 GAN）
- **输入**：256×256 人脸 → 256×256 卡通脸
- **参数**：~8M（ResNet-ish 编解码，非 GAN）
- **效果**：自拍漫画化（**注意不是 AnimeGANv2 那种 GAN，是纯 CNN 版**）
- **NPU 预判**：⚠️ 需实测——256 输入 + 编解码 8M 参数，NPU 应可行
- **量化难度**：中
- **来源**：GitHub `dusty-nv/snapseed-style`；或 `ToonApp` 论文复现
- **有趣度**：⭐⭐⭐⭐（视觉效果好，但需测 NPU 效率）

---

## 推荐优先级（我的主观排序）

**Tier 1（强烈建议加入，有趣 + NPU 稳赢）**
1. **A1 CNN 风格迁移** ⭐⭐⭐⭐⭐
2. **A2 老照片上色** ⭐⭐⭐⭐⭐
3. **B1 ArcFace 人脸比对**（InsightFace 现成 ONNX）⭐⭐⭐⭐⭐
4. **D3 图像去噪 DnCNN-S** ⭐⭐⭐⭐
5. **D1 图像去雾 DehazeNet** ⭐⭐⭐⭐

**Tier 2（有趣，值得试）**
6. **A3 微笑强度回归**（配 RetinaFace 联调）
7. **A4 人脸 68 关键点**
8. **C1/C2 MNIST/EMNIST 玩具 OCR**（NPU 速度极致的展示）
9. **E2 艺术流派分类**（配 A1 做闭环）
10. **E3 植物识别**

**Tier 3（备选）**
11. A5 EXIF 旋转 / A6 Deepfake / F2 水印检测
12. B2 年龄性别 / B3 表情分类
13. C3 车牌识别
14. D2 去雨 / D4 质量评分
15. E1 场景分类 / E4 花卉分类
16. F1 二维码定位 / F3 卡通脸 CNN

---

## 需要你确认的问题

1. **要不要 Tier 1 五个全上？** 我推荐从这五个开始，一次跑通能形成"风格化 + 上色 + 人脸 + 去噪 + 去雾"的完整视觉工具箱。
2. **A1 风格迁移选哪个版本？**
   - Johnson 2016 原版（15 层 conv，效果好但推理慢）
   - Fast Neural Style（Johnson 加速版，1x 实时）
   - AdaIN 版（更好但需风格图输入）
3. **B1 ArcFace 直接用 InsightFace 现成 ONNX**（w600k_r50.onnx，社区已量化好）还是重新量化？我推荐直接用现成的省时间。
4. **要不要 C1 MNIST 那种玩具模型？** 意义主要是"展示 NPU 极限速度（可能 0.1ms 级）"，作为速度对比 anchor。
5. **D3 去噪 + D1 去雾**——这两个前后对比效果最震撼，作为"图像修复"套件主力。

---

## 我下一步会做什么（等你审阅后）

审阅通过后，我按你选的清单：
1. 下载/导出 ONNX（用 RyzenAI-SW 1.7.1 官方脚本，或 curl 直连 HF resolve URL 绕过 blob 失败问题）
2. Quark XINT8 量化（用 `PowerOfTwoMethod.NonOverflow` 快标定，避开 MODNet 那种崩溃陷阱）
3. `bench.py MODE=hist` 增量测新模型，跑通后追加到 `cnn_zoo/result/report.html`
4. `parse_context.py` 拉 EP 自报 `No. of Operators : CPU x NPU y` 验证子图数
5. 与 iGPU DML 对比（`bench_dml.py`）看 NPU vs 780M 谁更优
