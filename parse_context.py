"""解析 VitisAI EP 的编译缓存 context.json，统计有多少子图/节点真正落到 NPU。"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".vitis_cache"
NAMES = {
    "zoo_lraspp_seg_4x4": "LRASPP-MobileNetV3 分割",
    "zoo_deeplabv3_seg_4x4": "DeepLabV3-MobileNetV3 分割",
    "zoo_modnet_matting_4x4": "MODNet 抠图",
    "zoo_animeganv2_4x4": "AnimeGANv2 动漫化",
    "zoo_realesrgan_4x4": "Real-ESRGAN ×4 超分",
}
out = {}
for d in sorted(CACHE.iterdir()):
    if not d.is_dir():
        continue
    ctx = d / "context.json"
    if not ctx.exists():
        continue
    try:
        j = json.loads(ctx.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(d.name, "parse err", str(e)[:80]); continue
    subs = j.get("metaDef", [])
    nsub = len(subs)
    nodes = sum(len(s.get("nodes", [])) for s in subs)
    biggest = max((len(s.get("nodes", [])) for s in subs), default=0)
    idlist = [s.get("id", "") for s in subs[:3]]
    out[d.name] = dict(subgraphs=nsub, npu_nodes=nodes, biggest_subgraph_nodes=biggest, sample_ids=idlist)
    print(f"{NAMES.get(d.name,d.name):28s} 子图={nsub:4d}  NPU节点={nodes:5d}  最大子图节点={biggest:5d}")
(ROOT / "result" / "npu_subgraph.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved result/npu_subgraph.json")
