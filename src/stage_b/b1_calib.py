"""B1: every method and setting on the calibration set, at 448 and 896 pixels (plan docs/records/STAGE_B_EFFICIENCY_PLAN.md).
Per image and method: FLOPs (and ratio to dense), cosine of final features to the dense model (patch mean and CLS,
plus the per-patch map, so any region-of-interest weighting can be applied afterwards), the probe's class per patch,
kNN correctness against the dense A0 bank (ImageNet-100 images) or the VOC confusion counts with the dense A0 probe
(VOC images). Chunks follow a2g-calib's stratified order, so the oracle loads one a2g tree file per chunk; workers
claim chunks with a .claim file. Usage: python -m src.stage_b.b1_calib [config] [resolutions]"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.ablate.runinfo import git_dirty, git_hash
from src.stage_a.a2g_calib import order
from src.stage_a.data import VOC, ImageList
from src.stage_a.evaluate import knn_correct, probe_logits
from src.stage_a.model import N_PREFIX, load_backbone

from .flops import dense_flops, dense_flops_tile, image_flops
from .methods import BlockOracle, Combo, Graph, Group, Oracle, Prune, ToMe, Windows, ops_of
from .model import EffModel

CLASSES, IGNORE = 21, 255


def build_methods(spec, grid):
    out = []
    if "tome" in spec:
        t = spec["tome"]
        out += [ToMe(r, first=f) for f, r in itertools.product(t["first"], t["r"])]
    if "prune" in spec:
        p = spec["prune"]
        out += [Prune(k, at=tuple(a), restore_at=r) for r, a, k in itertools.product(p.get("restore", [None]), p["at"], p["keep"])]
    if "group" in spec:
        g = spec["group"]
        out += [Group(K, m=m, tau=tau) for K, m, tau in itertools.product(g["clusters"], g["m"], g["tau"]) if m < K]
    out += [Oracle(s) for s in spec.get("oracle", [])]
    out += [Windows(w, grid) for w in spec.get("windows", [])]
    out += [Graph(s, grid) for s in spec.get("graph", [])]
    out += [BlockOracle(s) for s in spec.get("blockoracle", [])]
    if spec.get("named"):                                  # any method by its setting name, including combos "a+b"
        from .timing import from_name
        out += [from_name(n, grid) for n in spec["named"]]
    return out


def family(m):
    return type(m).__name__.lower()


def mask_of(ds, i):
    rel = ds.rows[i][1]
    m = Image.open(VOC / "SegmentationClass" / f"{Path(rel).stem}.png")
    return torch.from_numpy(np.asarray(ds.ltf(m), dtype=np.uint8).copy())


def run_resolution(cfg, res, model, out_dir, dev):
    g, N = res // 16, (res // 16) ** 2
    ds = ImageList("calibration", res)
    idx = order(len(ds), cfg["stride"])
    bs, chunk = cfg["batch"][res], cfg["chunk"]
    dtype = getattr(torch, cfg["dtype"])
    bank = torch.load(Path(cfg["a0_cache"]) / f"knn_bank_{res}.pt")
    bank_f, bank_y = bank["feats"].to(dev).float(), bank["labels"].to(dev)
    probe = torch.nn.Linear(384, CLASSES).to(dev)
    probe.load_state_dict(torch.load(Path(cfg["a0_runs"]) / f"probe_{res}.pt", map_location=dev))
    methods = build_methods(cfg["methods"][res], g)
    d_flops, d_tile = dense_flops(N), dense_flops_tile(N)
    chunk_dir = out_dir / f"chunks_{res}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for c0 in range(0, len(idx), chunk):
        ci = c0 // chunk
        if cfg.get("smoke_images") and ci > 0:
            break
        if ci >= cfg.get("max_chunks", {}).get(res, 10 ** 9):     # keep to the images the other B1 runs cover
            break
        spec = cfg["methods"][res]
        tree_file = Path(cfg["trees_dir"]) / str(res) / f"calibration_{ci:03d}.pt"
        block_file = Path(cfg.get("block_dir", "/work/stage_b/block_teacher")) / str(res) / f"calibration_{ci:03d}.pt"
        named = spec.get("named", [])
        need_tree = bool(spec.get("oracle") or spec.get("graph") or any(n.startswith(("oracle", "graph")) for n in named))
        need_block = bool(spec.get("blockoracle") or any(n.startswith("blockoracle") for n in named))
        if (chunk_dir / f"{ci:03d}.pt").exists() or (need_tree and not tree_file.exists()) or \
                (need_block and not block_file.exists()):   # oracle structures not saved yet: later run
            continue
        try:
            os.close(os.open(chunk_dir / f"{ci:03d}.claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        trees, graphs, blocks = {}, {}, {}
        if need_tree:
            raw = torch.load(tree_file, weights_only=False)
            pick = lambda fam: {n.split("+")[0].split("_", 1)[1] for n in named if n.startswith(fam + "_")}
            wanted, gw = set(spec.get("oracle", [])) | pick("oracle"), set(spec.get("graph", [])) | pick("graph")
            trees = {img: {e: s["head"][e] for e in wanted} for img, s in raw.items()}
            graphs = {img: {e: s["graph"][e] for e in gw} for img, s in raw.items()}
            del raw
        if need_block:
            blocks = torch.load(block_file, weights_only=False)
        rows = []
        imgs = idx[c0:c0 + chunk][: cfg.get("smoke_images")]
        for b0 in range(0, len(imgs), bs):
            batch = imgs[b0:b0 + bs]
            x = torch.stack([ds[i][0] for i in batch]).to(dev)
            labels = [int(ds.rows[i][2]) for i in batch]
            masks = {i: mask_of(ds, i).to(dev).long() for i in batch if ds.rows[i][0] == "voc"}
            with torch.no_grad():
                ref = model(x, dtype=dtype).final
            for m in [None] + methods:
                t = m.attn if isinstance(m, Combo) else m
                if isinstance(t, BlockOracle):
                    t.current = [blocks[i][t.setting] for i in batch]
                elif isinstance(t, Oracle):
                    t.current = [trees[i][t.setting] for i in batch]
                if isinstance(t, Graph):
                    t.trees = [graphs[i][t.setting] for i in batch]
                sts = []
                with torch.no_grad():
                    out = ref if m is None else model(x, structure=m, dtype=dtype, state_out=sts).final
                    cos_map = F.cosine_similarity(out[:, N_PREFIX:], ref[:, N_PREFIX:], dim=-1)    # (B, N)
                    cos_p = cos_map.mean(1)
                    pred_map = probe(out[:, N_PREFIX:].float()).argmax(-1)                     # (B, N) patch-level class
                    cos_c = F.cosine_similarity(out[:, 0], ref[:, 0], dim=-1)
                    cls = F.normalize(out[:, 0], dim=-1)
                    pred = probe_logits(probe, out[:, N_PREFIX:], res).argmax(1)
                for j, i in enumerate(batch):
                    if m is None:
                        fl = {"flops": d_flops, "flops_attention": None}
                        name, setting = "dense", "dense"
                    else:
                        fl = image_flops(sts[0], ops_of(m), j, N)
                        name, setting = ("combo" if "+" in m.name else family(m)), m.name
                    r = {"structure": name, "setting": setting, "image": i, "resolution": res, **fl,
                         "flops_ratio": fl["flops"] / d_flops,
                         "flops_tile_ratio": fl.get("flops_tile", d_tile) / d_tile,
                         "cos_patch": float(cos_p[j]), "cos_cls": float(cos_c[j]),
                         "tokens_final": int(sts[0].tokens[-1][1]) - N_PREFIX if sts else N,
                         "cos_map": cos_map[j].half().cpu(), "pred_map": pred_map[j].to(torch.uint8).cpu()}
                    if isinstance(m, Group) and m.dense_heads:
                        r["dense_head_share"] = float(np.mean(m.dense_heads))
                    if ds.rows[i][0] == "in100":
                        r["knn_correct"] = bool(knn_correct(bank_f, bank_y, cls[j:j + 1], torch.tensor([labels[j]], device=dev),
                                                            **cfg["knn"])[0])
                    else:
                        t, p = masks[i], pred[j]
                        keep = t != IGNORE
                        r["conf"] = torch.bincount(t[keep] * CLASSES + p[keep], minlength=CLASSES * CLASSES).cpu().tolist()
                    rows.append(r)
        torch.save(rows, chunk_dir / f"{ci:03d}.pt")
        print(f"[{res}] chunk {ci} images {c0 + len(imgs)}", flush=True)


def main(cfg_path="configs/stage_b/b1_calib.yaml", resolutions=None):
    """resolutions: optional comma list overriding the config (e.g. 448 while the GPU is shared)."""
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    if resolutions:
        cfg["resolutions"] = [int(r) for r in resolutions.split(",")]
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": cfg["run_id"], "config": cfg, "git": git_hash(), "dirty": git_dirty(),
            "image_id": os.environ.get("IMAGE_ID", "unknown")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EffModel(load_backbone(dev)).to(dev)
    for res in cfg["resolutions"]:
        run_resolution(cfg, res, model, out, dev)
        print(f"[{res}] done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
