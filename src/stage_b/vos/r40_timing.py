"""R40 plan section J3 (reported): wall clock of the expectation focus map beside today's design, ALGM 0.98, ToMe r64 and dense, ViT-S,
batch 1, bf16, idle GPU, following R8 (`src/stage_b/r8_timing.py`: DAVIS frames 448 x 784 of three sequences, 40 frames
each, median after warm-up). Video order within each sequence, so the expectation focus map's expectation state carries over as in
the closed loop; the previous frame's ground-truth ids stand in for the prediction (the map of the map-based methods and
the expectation focus map's object states). The expectation focus map's timed step includes its per-frame map motion (`shift_by_velocity`) and
object-state update (`objects_from_predictions`) besides the encoder with the expectation map, surprise check and tiers inside; frames
where every object is lost run its fallback (plain ALGM 0.98), as the evaluator. Eager for all; CUDA graphs for dense,
ToMe and today's design (as R8); ALGM's token counts and the expectation focus map's tiers depend on the image, so no graph.
Usage (ml-base:vos, idle GPU): python -m src.stage_b.vos.r40_timing [out json]
       python -m src.stage_b.vos.r40_timing check   (CPU, fp32, a few frames, no graphs, no timing claim: runs the code path)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.stage_a.model import load_backbone
from src.stage_b.expect import ExpectMap, ExpectState, config_from_name, objects_from_predictions
from src.stage_b.graphs import GraphRunner
from src.stage_b.model import EffModel
from src.stage_b.p2_davis import load_frame, load_mask, shift_by_velocity
from src.stage_b.p2_timing import HW, ROOT, SEQS
from src.stage_b.timing import from_name, gpu_idle

from .r40_compare import Q, S_FINAL

OURS = "ftome_l0.7_t0.01_floor64_dil1_noprop_cls64"
OTHERS = [OURS, "algm_t0.98", "tome_r64_from1_noprop"]
N_FRAMES, WARMUP_SEQS = 40, 1


def cov(m):
    """(H, W) bool or id map -> (1, N) patch coverage (the interest the encoder receives)."""
    return F.avg_pool2d((m > 0).float()[None, None], 16)[0, 0].flatten()[None]


def sequences(dev="cuda", n_frames=N_FRAMES, n_seqs=len(SEQS)):
    """Per sequence, from the third frame: (frame (1, 3, H, W), objects seen before t (ids), id map at t − 1, at t − 2,
    patch coverage of t − 1 (the map of today's design))."""
    out = []
    for seq in SEQS[:n_seqs]:
        imgs = sorted((ROOT / "JPEGImages/480p" / seq).glob("*.jpg"))[:n_frames]
        ids = [load_mask(a, HW, dev) for a in sorted((ROOT / "Annotations/480p" / seq).glob("*.png"))[:n_frames]]
        present = [set(int(o) for o in torch.unique(m).tolist() if o > 0) for m in ids]
        seen = [sorted(set().union(*present[:t])) for t in range(len(ids))]
        out.append([(load_frame(i, HW, dev)[None], seen[t], ids[t - 1], ids[t - 2], cov(ids[t - 1]))
                    for t, i in enumerate(imgs) if t >= 2])
    return out


def run_timed(seqs, step, reset, cuda=True):
    """Median and quartiles (ms) of `step` over every frame, after one warm-up pass over the first sequence."""
    ts = []
    for k, seq in enumerate(seqs[:WARMUP_SEQS] + seqs):
        reset()
        for fr in seq:
            if cuda:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
            else:
                t0 = time.perf_counter()
            step(fr)
            if cuda:
                e.record()
                torch.cuda.synchronize()
            if k >= WARMUP_SEQS:
                ts.append(s.elapsed_time(e) if cuda else 1000 * (time.perf_counter() - t0))
    return {"ms_per_frame": float(np.median(ts)), "ms_iqr": [float(np.percentile(ts, 25)), float(np.percentile(ts, 75))],
            "frames": len(ts)}


@torch.no_grad()
def main(out_json=None):
    check = out_json == "check"
    dev, dt = ("cpu", torch.float32) if check else ("cuda", torch.bfloat16)
    if not check and not gpu_idle():
        sys.exit("r40_timing: another process is using the GPU; wall clock needs an idle GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = EffModel(load_backbone(dev)).to(dev)
    model.fast = not check
    blocks = len(model.bb.model.layer)
    seqs = sequences(dev, 5, 2) if check else sequences()
    res = []
    none = lambda: None

    def add(name, mode, step, reset=none, **extra):
        r = {"setting": name, "mode": mode, **run_timed(seqs, step, reset, cuda=not check), **extra}
        res.append(r)
        print(f"{name:60s} {mode:6s} {r['ms_per_frame']:.2f} ms/frame", flush=True)

    add("dense", "eager", lambda fr: model(fr[0], dtype=dt))
    if not check:
        gd = GraphRunner(model, "dense", HW)
        add("dense", "graph", lambda fr: gd.run(fr[0]))
    for name in OTHERS:
        per_image = getattr(from_name(name, HW[0] // 16, blocks), "per_image", False)
        add(name, "eager", lambda fr, name=name, per_image=per_image: model(
            fr[0], structure=from_name(name, HW[0] // 16, blocks), dtype=dt, interest=fr[4] if per_image else None))
        if name.startswith("algm") or check:
            continue
        gr = GraphRunner(model, name, HW)
        for seq in seqs:
            for fr in seq:
                gr.run(fr[0], fr[4])
        add(name, "graph", lambda fr, gr=gr: gr.run(fr[0], fr[4]), classes=len(gr.graphs))

    inner, rest = S_FINAL.split("~")
    fb_name, exp_name = rest.split("@")
    cfg = {}

    def reset():
        st = ExpectState()
        cfg.update(cfg=config_from_name(exp_name), st=st, ostate={},
                   main=ExpectMap(from_name(inner, HW[0] // 16, blocks), st, config_from_name(exp_name)),
                   fb=ExpectMap(from_name(fb_name, HW[0] // 16, blocks), st, config_from_name(exp_name), passive=True))

    lost = []

    def step(fr):                                    # as the evaluator: the map moves with the velocity, then states
        x, ids, prev1, prev2, _ = fr
        imap = cov((prev1 > 0) | (shift_by_velocity(prev1, prev2) > 0))
        cfg["st"].objects, all_lost = objects_from_predictions(prev1, prev2, ids, cfg["ostate"], cfg["cfg"].motion)
        m = cfg["fb"] if (not bool(imap.any()) or all_lost) else cfg["main"]
        lost.append(m is cfg["fb"])
        model(x, structure=m, dtype=dt, interest=imap)

    add(S_FINAL, "eager", step, reset, fallback_share=None)
    res[-1]["fallback_share"] = float(np.mean(lost[-res[-1]["frames"]:]))
    if check:
        print("check passed: every setting ran on the CPU; fallback share", res[-1]["fallback_share"])
        return
    dense = {r["mode"]: r["ms_per_frame"] for r in res if r["setting"] == "dense"}
    for r in res:
        r["speedup_vs_dense_eager"] = dense["eager"] / r["ms_per_frame"]
        r["speedup_vs_dense_graph"] = dense["graph"] / r["ms_per_frame"]
    out = Path(out_json) if out_json else Q / "r40_timing.json"
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
