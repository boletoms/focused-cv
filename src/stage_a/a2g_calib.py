"""A2-graph on the calibration set (owner decisions 2026-11-17, after review stop 2): the graph teacher (levels split
independently, up to 3 memberships, clusters linked across levels by overlap) and the teacher built per head and
per block, with two intermediate forms that separate the effect of heads from that of blocks:

- graph_full, graph_key_only, graph_strict: graph teacher, one structure per level shared by heads and blocks;
- head_teacher: one partition per block and head, from that head's own dense attention;
- head_level_teacher: one partition per level and head, from the head's attention averaged over the level's blocks;
- block_teacher: one partition per block, from the block's attention averaged over heads.

The last three have no summaries (like k-means). Severed attention and cost are defined as in a2_calib; the
comparison structures (windows, random groups, k-means) are taken from the a2_calib rows of the same images.
Images run in a stratified order (every 4th image first) so a partial run is a balanced subset.
Usage: python -m src.stage_a.a2g_calib [config]. Several workers may run at once; each claims a chunk with a
.claim file (remove stale claims after a crash)."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import sys
from pathlib import Path

import torch
import yaml

from src.ablate.runinfo import git_dirty, git_hash

from .a2_calib import per_level
from .cost import model_cost
from .data import ImageList
from .model import LEVEL_BLOCKS, LEVEL_OF_BLOCK, N_PREFIX, Wrapped, load_backbone
from .structures import HeadTeacherStructure, TreeStructure
from .teacher import TreeConfig, build_single, build_tree, tree_stats


def order(n, stride):
    """Stratified image order: 0, stride, 2*stride, ..., then 1, 1+stride, ..."""
    return [i for r in range(stride) for i in range(r, n, stride)]


def compact(L):
    return {"memb": L.memb.short(), "core": L.core.short()}


def sev_rows(img, name, setting, sev_blocks, cost, extra=None):
    lv, allb = per_level(sev_blocks)
    return {"structure": name, "setting": setting, "seed": 0, "image": img, **cost,
            **{f"severed_l{l + 1}": v for l, v in enumerate(lv)}, "severed": allb, **(extra or {})}


def run_image(cfg, res, model, x, img, out_rows, store):
    g, N = res // 16, (res // 16) ** 2
    dev = "cuda"
    P = [None] * 12

    def cb(b, p, q, k):
        P[b] = p[0, :, N_PREFIX:, N_PREFIX:].float()             # (H, N, N)

    with torch.no_grad():
        model(x[None].cuda(), dtype=torch.bfloat16, attn_cb=cb)
    tot = [P[b].sum().clamp_min(1e-12) for b in range(12)]
    heads_sum = [P[b].sum(0) for b in range(12)]
    t = cfg["teacher"]
    eps = t["eps"][res]
    hp = t["head_eps"][res]

    # graph teacher
    la = [sum(heads_sum[b] for b in bl) / (6 * len(bl)) for bl in LEVEL_BLOCKS]
    caches = {}
    for strict in (False, True):
        for e in eps:
            tc = TreeConfig(e, 2.0 if strict else t["beta"], t["s_min"], max_memb=t["max_memb"], graph=True)
            tree = build_tree(la, tc, caches)
            td = [L.to_dict() for L in tree]
            store["graph"][tc.name] = td
            stats = {f"tree_{k}": v for k, v in tree_stats(tree).items()}
            for name, kw in ((("graph_strict", {}),) if strict else (("graph_full", {}), ("graph_key_only", {"key_only": True}))):
                ts = TreeStructure([td], N, g, dev, **kw)
                allowed = [ts.allowed(l)[0] for l in range(4)]
                sev = [float((heads_sum[b] * (~allowed[LEVEL_OF_BLOCK[b]]).float()).sum() / tot[b]) for b in range(12)]
                out_rows.append(sev_rows(img, name, tc.name, sev, model_cost(N, ts.block_structures(0)),
                                         stats if name != "graph_key_only" else None))

    # teachers per head and/or per block (one level each, split from the whole image)
    def single(A, e, cache):
        return build_single(A, TreeConfig(e, t["beta"], t["s_min"], max_memb=t["max_memb"]), cache)

    forms = {
        "head_teacher": lambda e, c: [[single(P[b][h], e, c.setdefault((b, h), {})) for h in range(6)] for b in range(12)],
        "head_level_teacher": None,
        "block_teacher": lambda e, c: [[single(heads_sum[b] / 6, e, c.setdefault(b, {}))] for b in range(12)],
    }
    lvl_head = [[sum(P[b][h] for b in bl) / len(bl) for h in range(6)] for bl in LEVEL_BLOCKS]

    def head_level(e, c):
        per_l = [[single(lvl_head[l][h], e, c.setdefault((l, h), {})) for h in range(6)] for l in range(4)]
        return [per_l[LEVEL_OF_BLOCK[b]] for b in range(12)]

    forms["head_level_teacher"] = head_level
    for name, fn in forms.items():
        c = {}
        for e in hp:
            levels = fn(e, c)
            hs = HeadTeacherStructure(levels, N, dev)
            sev = []
            for b in range(12):
                al = hs.allowed_block(b)                                  # (H or 1, N, N)
                sev.append(float((P[b] * (~al).float()).sum() / tot[b]))
            n_cl = sum(len(L.clusters) for hl in levels for L in hl) / sum(len(hl) for hl in levels)
            memb = sum(float((L.memb >= 0).sum(1).float().mean()) for hl in levels for L in hl) / sum(len(hl) for hl in levels)
            out_rows.append(sev_rows(img, name, f"eps{e:g}", sev, model_cost(N, hs.block_structures()),
                                     {"clusters_per_partition": n_cl, "memberships_per_token": memb}))
            if name == "head_teacher":
                store["head"][f"eps{e:g}"] = [[compact(L) for L in hl] for hl in levels]


def run_resolution(cfg, res, model, out_dir, tree_dir):
    ds = ImageList("calibration", res)
    idx = order(len(ds), cfg["stride"])
    chunk = cfg["chunk"]
    chunk_dir = out_dir / f"chunks_{res}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    tree_dir.mkdir(parents=True, exist_ok=True)
    for c0 in range(0, len(idx), chunk):
        ci = c0 // chunk
        if (chunk_dir / f"{ci:03d}.json").exists():
            continue
        try:                                   # claim the chunk, so several workers can share a run
            os.close(os.open(chunk_dir / f"{ci:03d}.claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        rows, trees = [], {}
        for img in idx[c0:c0 + chunk]:
            x = ds[img][0]
            store = {"graph": {}, "head": {}}
            run_image(cfg, res, model, x, img, rows, store)
            trees[img] = store
        torch.save(trees, tree_dir / f"calibration_{ci:03d}.pt")
        (chunk_dir / f"{ci:03d}.json").write_text(json.dumps(rows))
        print(f"[{res}] chunk {ci} images {c0 + len(idx[c0:c0 + chunk])}", flush=True)


def main(cfg_path="configs/stage_a/a2g_calib.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": cfg["run_id"], "config": cfg, "git": git_hash(), "dirty": git_dirty(),
            "image_id": os.environ.get("IMAGE_ID", "unknown")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    model = Wrapped(load_backbone("cuda")).cuda()
    for res in cfg["resolutions"]:
        run_resolution(cfg, res, model, out, Path(cfg["trees_dir"]) / str(res))
        print(f"[{res}] done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
