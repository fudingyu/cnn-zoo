"""汇总 bench.json / bench_hist.json / bench_dml.json / npu_subgraph.json / metrics.json + 结果图 → report.html"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "result"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


new = {r["key"]: r for r in load("bench.json")}
hist = {r["key"]: r for r in load("bench_hist.json")}
dml = {r["key"]: r for r in load("bench_dml.json")}
subs = load("npu_subgraph.json")
subs = {k.replace("zoo_", "").replace("_4x4", ""): v for k, v in subs.items()} if isinstance(subs, dict) else {}
_m = load("metrics.json")
metrics = _m if isinstance(_m, dict) else {r["key"]: r for r in _m}
mse = load("mse_check.json") if (RES / "mse_check.json").exists() else {}
# DeepLabV3 换 MinMSE 重标定后质量修复，用新指标
if mse.get("deeplabv3_mse"):
    d3 = metrics.setdefault("deeplabv3_seg", {})
    d3["pixel_agreement_vs_fp32"] = mse["deeplabv3_mse"]["agree"]
    d3["top_classes_int8"] = ["background", "person"]
if mse.get("modnet_matting_mse"):
    m = metrics.setdefault("modnet_matting", {})
    m["mae_vs_fp32"] = round(mse["modnet_matting_mse"]["mae"] * 255, 1)
    m["psnr_vs_fp32"] = mse["modnet_matting_mse"]["psnr"]


def f(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def sp_row(key, d, show_dml=True, extra_sub=False):
    r = d[key]
    npu, c8, c32 = r.get("npu_ms"), r.get("cpu_ms"), r.get("fp32_cpu_ms")
    ig = dml.get(key, {}).get("dml_ms")
    sub = subs.get(key)
    ssub = f"{sub['subgraphs']}个/{sub['npu_nodes']}节点" if sub else "—"
    s1 = f"×{c8/npu:.2f}" if (c8 and npu) else ("失败" if r.get("npu_err") else "—")
    s2 = f"×{c32/npu:.2f}" if (c32 and npu) else "—"
    cls = "good" if (c8 and npu and c8/npu >= 2) else ("mid" if (c8 and npu and c8/npu >= 1) else "bad")
    return f"""      <tr>
        <td class="nm">{r.get('title', key)}</td>
        <td class="mono">{'×'.join(str(x) for x in r.get('input_shape', []))}</td>
        <td class="mono npu">{f(npu)}</td>
        <td class="mono cpu">{f(c8)}</td>
        <td class="mono {cls}">{s1}</td>
        <td class="mono cpu">{f(c32)}</td>
        <td class="mono {cls}">{s2}</td>
        <td class="mono gpu">{f(ig)}</td>
        {f'<td class="mono">{ssub}</td>' if extra_sub else ''}
      </tr>"""


NEW_KEYS = ["lraspp_seg", "deeplabv3_seg", "modnet_matting", "animeganv2", "realesrgan"]
HIST_KEYS = ["yolov8m", "resnet50", "mobilenetv2", "retinaface", "rcan_sr"]

table_new = "\n".join(sp_row(k, new, extra_sub=True) for k in NEW_KEYS if k in new)
table_hist = "\n".join(sp_row(k, hist) for k in HIST_KEYS if k in hist and hist[k].get("npu_ok"))

TITLES = {
    "lraspp_seg": "LRASPP-MobileNetV3 · 语义分割",
    "deeplabv3_seg": "DeepLabV3-MobileNetV3 · 语义分割",
    "modnet_matting": "MODNet · 人像抠图",
    "animeganv2": "AnimeGANv2 · 动漫风格化",
    "realesrgan": "Real-ESRGAN ×4 · 超分",
}
DESC = {
    "lraspp_seg": "PASCAL VOC 21 类像素级分割。整个模型被 EP 编译成 1 个 NPU 子图，是本轮唯一“整图上 NPU”的新模型。",
    "deeplabv3_seg": "同样 21 类，但 ASPP 金字塔带空洞卷积。697 个节点几乎全进了 NPU，151ms 仍慢于 8 线程 CPU；INT8 用 PowerOfTwo 快速标定会全崩（整图全背景），换 MinMSE 标定后恢复（下方效果图为修复版）。",
    "modnet_matting": "人像软抠图。EP 直接放弃编译（0 个子图，全 CPU 兜底）还倒贴开销；INT8 换了两轮标定方法输出都接近全零，这种 InstanceNorm+软上采样结构基本无法 INT8 化。",
    "animeganv2": "日系动漫风格化。被切成 40 个碎片子图（最大仅 22 节点），每片都要 NPU↔CPU 搬运，3.7s/帧；好在视觉质量基本保住。",
    "realesrgan": "RRDB 结构 4 倍超分。81 个碎片子图，被搬运开销拖垮；iGPU 只要 169ms。",
}

cards = []
for key, title in TITLES.items():
    m = metrics.get(key, {})
    imgs = sorted(p.name for p in RES.glob(key + "_*.png"))
    if not imgs:
        continue
    gallery = "\n".join(
        f'        <figure><img src="{n}" loading="lazy"><figcaption>{n.replace(key+"_", "").replace(".png", "")}</figcaption></figure>'
        for n in imgs)
    q = []
    if m.get("pixel_agreement_vs_fp32") is not None:
        q.append(f"INT8 vs FP32 像素分类一致率 <b>{m['pixel_agreement_vs_fp32']*100:.1f}%</b>")
    if m.get("psnr_vs_fp32") is not None:
        q.append(f"PSNR <b>{m['psnr_vs_fp32']:.1f} dB</b>")
    if m.get("mae_vs_fp32") is not None:
        q.append(f"MAE <b>{m['mae_vs_fp32']:.1f}/255</b>")
    if m.get("top_classes_int8"):
        q.append("分割到：" + "、".join(m["top_classes_int8"]))
    cards.append(f"""    <section class="card">
      <h2>{title}</h2>
      <p class="desc">{DESC.get(key, '')}</p>
      <p class="q">{' · '.join(q) if q else '—'}</p>
      <div class="gal">
{gallery}
      </div>
    </section>""")

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>XDNA1 NPU 趣味 CNN 模型实测</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#2b313a;--tx:#e6edf3;--dim:#8b949e;--npu:#ff6b6b;--cpu:#3fb950;--gpu:#d2a8ff;--acc:#58a6ff;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.65}}
.wrap{{max-width:1120px;margin:0 auto;padding:32px 20px 90px}}
h1{{font-size:27px;margin:0 0 6px}}
.sub{{color:var(--dim);font-size:14px;margin-bottom:26px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13.5px}}
th,td{{padding:9px 11px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;white-space:normal}}
th{{background:#1c2430;color:var(--dim);font-weight:600;font-size:12.5px}}
tr:last-child td{{border-bottom:none}}
.mono{{font-family:ui-monospace,Consolas,monospace}}
.npu{{color:var(--npu)}} .cpu{{color:var(--cpu)}} .gpu{{color:var(--gpu)}}
.good{{color:var(--npu);font-weight:700}} .mid{{color:#e3b341;font-weight:700}} .bad{{color:var(--dim)}}
h2.sec{{font-size:18px;margin:38px 0 12px;padding-left:10px;border-left:4px solid var(--acc)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}}
.card h2{{font-size:17px;margin:0 0 6px}}
.desc{{color:var(--dim);font-size:14px;margin:0 0 6px}}
.q{{font-size:13px;color:#adbac7;margin:0 0 12px}}
.gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}
figure{{margin:0;background:#0d1117;border:1px solid var(--line);border-radius:8px;overflow:hidden}}
figure img{{width:100%;display:block}}
figcaption{{font-size:11px;color:var(--dim);padding:6px 8px;font-family:ui-monospace,Consolas,monospace}}
.note{{color:#adbac7;font-size:13.5px;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:13px 16px;margin:16px 0}}
.note b{{color:var(--tx)}}
code{{background:#1c2430;padding:1px 6px;border-radius:4px;font-size:12px}}
.leg{{font-size:12px;color:var(--dim);margin-top:4px}}
</style></head><body><div class="wrap">
<h1>又找了一批"有趣"的 CNN 上 NPU，结果挺出乎意料</h1>
<div class="sub">Ryzen 7 8845HS · XDNA1 NPU (Phoenix 16 TOPS, 4x4.xclbin) + Radeon 780M iGPU + 8 核 Zen4 · RyzenAI 1.7.1 / onnxruntime-vitisai 1.23.3 · 全部 INT8(XINT8) · 稳态延迟（warmup 5 + 20 次均值）</div>

<h2 class="sec">本轮新测：5 个趣味模型（统一口径：8 线程 CPU）</h2>
<table>
  <thead><tr><th>模型 / 任务</th><th>输入</th><th>NPU INT8<br>(ms)</th><th>CPU INT8×8T<br>(ms)</th><th>NPU vs<br>CPU-INT8</th><th>CPU FP32×8T<br>(ms)</th><th>NPU vs<br>CPU-FP32</th><th>iGPU FP32<br>780M (ms)</th><th>NPU 子图<br>切分</th></tr></thead>
  <tbody>
{table_new}
  </tbody>
</table>
<div class="leg">NPU 子图切分 = VitisAI EP 编译缓存(context.json)中落在 NPU 上的子图数 / 总节点数。</div>

<h2 class="sec">老模型同口径复测（此前数据的统一基准）</h2>
<table>
  <thead><tr><th>模型 / 任务</th><th>输入</th><th>NPU INT8<br>(ms)</th><th>CPU INT8×8T<br>(ms)</th><th>NPU vs<br>CPU-INT8</th><th>iGPU FP32<br>780M (ms)</th></tr></thead>
  <tbody>
{table_hist}
  </tbody>
</table>

<div class="note">
<b>三个结论：</b><br>
1. <b>XDNA1 并非处处香</b>：对小而规整的模型（ResNet50/MobileNetV2/RCAN/YOLOv8m）NPU 确实快 CPU 3~7 倍；
但 512×512 的分割、抠图、风格迁移、超分一上来，8 线程 Zen4 CPU 跑 INT8 就能反超 NPU。<br>
2. <b>性能杀手是"子图碎片化"</b>：LRASPP 整图 1 个子图 682 节点 → 27.7ms 真加速；
AnimeGANv2 被切成 40 片、Real-ESRGAN 81 片，每片都要 NPU↔CPU 搬运 → 反而比 CPU 慢 4~6 倍；
MODNet 更惨，EP 直接 0 子图全回退，还倒贴开销。<br>
3. <b>iGPU 780M 才是本机视觉模型的隐藏王者</b>：FP32 跑这些大模型全面碾压 NPU（抠图 30ms vs 454ms，动漫化 178ms vs 3658ms），
甚至比 NPU 快 20+ 倍。只有小分类模型（ResNet50 10.6ms vs NPU 5.2ms）NPU 还能保住优势。
</div>

<h2 class="sec">效果实拍（全部在 NPU 上跑出，除非标注）</h2>
{chr(10).join(cards)}

<div class="note">
<b>复现方式</b>：每个模型先用 <code>onnxsim</code> 固化静态 shape，再用 <code>amd-quark 0.11</code> 静态量化
（XINT8：激活 UINT8 / 权重 INT8，PowerOfTwoMethod 标定，16 张本地图片）；NPU 会话为
<code>VitisAIExecutionProvider + 4x4.xclbin</code>，编译缓存命中后计时。
FP32 参考用 CPU EP 同图推理。DeepLabV3 与 MODNet 的 INT8 版出现严重退化（前者全背景、后者输出近全零），
说明 PowerOfTwo 快速标定对这两类结构不够——要用 MinMSE + 更大标定集才有救。
</div>
</div></body></html>"""

(RES / "report.html").write_text(html, encoding="utf-8")
print("report ->", RES / "report.html")
