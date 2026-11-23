"""A2 on the calibration set: teacher trees for every variant and setting, and severed attention and cost for every
structure (brief sections 5, 6 and 8; owner decision 2026-11-17: calibration first).

Severed attention is per block, from the dense run only: dense patch-to-patch attention mass the structure disallows,
divided by all patch-to-patch mass, summed over heads; then averaged over a level's blocks (per level) and over all
12 blocks (overall). Resumable: per-chunk results are saved and skipped on restart.
Usage: python -m src.stage_a.a2_calib [config]"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.ablate.runinfo import git_dirty, git_hash

from .cost import model_cost
from .evaluate import loader
from .model import LEVEL_BLOCKS, LEVEL_OF_BLOCK, N_PREFIX, Wrapped, load_backbone
from .structures import KMeansStructure, RandomStructure, TreeStructure, WindowStructure, kmeans_ids
from .teacher import TreeConfig, build_tree, tree_stats


def tree_configs(cfg, res):
    t = dict(cfg["trees"])
    t["eps"] = t["eps"][res]
    out = [TreeConfig(e, t["beta"], t["s_min"]) for e in t["eps"]]
    out += [TreeConfig(e, b, t["s_min"]) for b in t["beta_sweep"] for e in t["eps_side"]]
    out += [TreeConfig(e, t["beta"], s) for s in t["s_sweep"] for e in t["eps_side"]]
    out += [TreeConfig(e, 2.0, t["s_min"]) for e in t["eps"]]
    out += [TreeConfig(e, 2.0, s) for s in t["s_sweep"] for e in t["eps_side"]]
    out += [TreeConfig(e, t["beta"], t["s_min"], nested=False) for e in t["eps"]]
    return out


def tree_uses(c: TreeConfig):
    """(structure name, TreeStructure kwargs) evaluated on one tree."""
    if not c.nested:
        return [("tree_nonnested", {"nested": False})]
    if c.beta > 1:
        return [("tree_strict", {})]
    return [("tree_full", {}), ("tree_key_only", {"key_only": True}),
            ("tree_full_no_summaries", {"summaries": False}), ("tree_full_no_globals", {"globals_visible": False})]


def per_level(sev_blocks):
    """sev_blocks: (12,) -> per-level means and the overall mean."""
    lv = [float(np.mean([sev_blocks[b] for b in bs])) for bs in LEVEL_BLOCKS]
    return lv, float(np.mean(sev_blocks))


def run_resolution(cfg, res, model, out_dir, tree_dir):
    g, N = res // 16, (res // 16) ** 2
    dev = "cuda"
    windows = [WindowStructure(g, w, dev) for w in cfg["windows"][res]]
    randoms = [RandomStructure(N, n, 6, s, dev) for n in cfg["groups"] for s in cfg["seeds"]]
    tcfgs = tree_configs(cfg, res)
    chunk_dir = out_dir / f"chunks_{res}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    tree_dir.mkdir(parents=True, exist_ok=True)
    bs = cfg["batch"][res]
    chunk, rows, trees_chunk, ci = cfg["chunk"], [], {}, 0
    dl = loader("calibration", res, bs, cfg["loader_workers"])
    for x, _, idx in dl:
        ci_now = int(idx[0]) // chunk
        if (chunk_dir / f"{ci_now:03d}.json").exists():
            continue
        B = len(idx)
        blocks = [None] * 12
        nontree = {}  # name -> (B, 12) severed

        def cb(b, p, q, k):
            pp = p[:, :, N_PREFIX:, N_PREFIX:]                      # (B, H, N, N)
            ph = pp.sum(1)                                           # heads summed
            tot = ph.sum((1, 2)).clamp_min(1e-12)
            blocks[b] = ph
            for ws in windows:
                al = ws.allowed_block(b)
                v = torch.zeros(B, device=dev) if al is None else (ph * (~al[0]).float()).sum((1, 2)) / tot
                nontree.setdefault(("windows", f"w{ws.w}", 0), torch.zeros(B, 12, device=dev))[:, b] = v
            for rs in randoms:
                al = rs.allowed_block(b)                              # (H, N, N)
                v = (pp * (~al).float()[None]).sum((1, 2, 3)) / tot
                nontree.setdefault(("random", f"g{rs.n_groups}", rs.seed), torch.zeros(B, 12, device=dev))[:, b] = v
            for kk in cfg["kmeans"]:
                for s in cfg["seeds"]:
                    qi, ki = kmeans_ids(q[:, :, N_PREFIX:], k[:, :, N_PREFIX:], kk, cfg["kmeans_iters"], s)
                    al = qi[..., :, None] == ki[..., None, :]
                    v = (pp * (~al).float()).sum((1, 2, 3)) / tot
                    key = ("kmeans", f"k{kk}", s)
                    nontree.setdefault(key, torch.zeros(B, 12, device=dev))[:, b] = v
                    oq = torch.nn.functional.one_hot(qi, kk).sum(2).float()   # (B, H, k)
                    ok = torch.nn.functional.one_hot(ki, kk).sum(2).float()
                    pairs = (oq * ok).sum(-1).mean(-1)                         # (B,) patch pairs, mean over heads
                    nontree.setdefault(key + ("pairs",), torch.zeros(B, 12, device=dev))[:, b] = pairs

        la = [None] * 4
        with torch.no_grad():
            model(x.cuda(), dtype=torch.bfloat16, attn_cb=cb)
            for l, bl in enumerate(LEVEL_BLOCKS):
                la[l] = sum(blocks[b] for b in bl) / (6 * len(bl))  # level attention, heads and blocks averaged
        for i in range(B):
            img = int(idx[i])
            caches, img_trees = {}, {}
            A = [a[i] for a in la]
            for tc in tcfgs:
                tree = build_tree(A, tc, caches)
                stats = tree_stats(tree)
                td = [L.to_dict() for L in tree]
                img_trees[tc.name] = td
                for name, kw in tree_uses(tc):
                    ts = TreeStructure([td], N, g, dev, **kw)
                    allowed = [ts.allowed(l)[0] for l in range(4)]
                    sev = torch.stack([(blocks[b][i] * (~allowed[LEVEL_OF_BLOCK[b]]).float()).sum()
                                       / blocks[b][i].sum().clamp_min(1e-12) for b in range(12)]).tolist()
                    lv, allb = per_level(sev)
                    c = model_cost(N, ts.block_structures(0))
                    rows.append({"structure": name, "setting": tc.name, "seed": 0, "image": img, **c,
                                 **{f"severed_l{l + 1}": v for l, v in enumerate(lv)}, "severed": allb,
                                 **({f"tree_{k}": v for k, v in stats.items()} if name in ("tree_full", "tree_strict", "tree_nonnested") else {})})
            trees_chunk[img] = img_trees
            for key, v in nontree.items():
                if len(key) == 4:
                    continue
                fam, setting, seed = key
                lv, allb = per_level(v[i].tolist())
                if fam == "windows":
                    c = model_cost(N, next(w for w in windows if f"w{w.w}" == setting).block_structures())
                elif fam == "random":
                    c = model_cost(N, next(r for r in randoms if f"g{r.n_groups}" == setting and r.seed == seed).block_structures())
                else:
                    from .cost import BlockStructure
                    pairs = nontree[key + ("pairs",)][i].tolist()
                    c = model_cost(N, [BlockStructure(groups=[(1, p)]) for p in pairs])
                rows.append({"structure": fam, "setting": setting, "seed": seed, "image": img, **c,
                             **{f"severed_l{l + 1}": v for l, v in enumerate(lv)}, "severed": allb})
        last = int(idx[-1])
        if (last + 1) % chunk == 0 or last + 1 == len(dl.dataset):
            ci = last // chunk
            torch.save(trees_chunk, tree_dir / f"calibration_{ci:03d}.pt")
            (chunk_dir / f"{ci:03d}.json").write_text(json.dumps(rows))
            print(f"[{res}] chunk {ci} images {last + 1}", flush=True)
            rows, trees_chunk = [], {}


def main(cfg_path="configs/stage_a/a2_calib.yaml"):
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
