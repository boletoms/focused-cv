"""N7 timing: per-frame wall clock on the DAVIS frames of p2_timing (batch 1, bf16, idle GPU), eager against CUDA
graphs, dense against the fast budget with size classes. `check` compares graph and eager outputs on the same frames
(same promoted map) instead of timing, and may run on a busy GPU.
Usage: python -m src.stage_b.n7_timing <out.json|check> [backbone|-] [budget setting,...]"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.stage_a.model import load_backbone

from .graphs import GraphRunner
from .model import EffModel
from .p2_timing import HW, WARMUP
from .timing import from_name, gpu_idle

SETTINGS = ["ftome_l0.5_t0.01_floor128_dil2_noprop_cls64", "ftome_l0.5_t0.01_floor128_dil2_noprop_cls128"]


def frames():
    import torch.nn.functional as F

    from .p2_davis import load_frame, load_mask
    from .p2_timing import ROOT, SEQS
    out = []
    for seq in SEQS:
        imgs = sorted((ROOT / "JPEGImages/480p" / seq).glob("*.jpg"))[:40]
        annos = sorted((ROOT / "Annotations/480p" / seq).glob("*.png"))[:40]
        for i, a in zip(imgs, annos):
            m = load_mask(a, HW, "cuda")
            out.append((load_frame(i, HW, "cuda")[None], F.avg_pool2d((m > 0).float()[None, None], 16)[0, 0].flatten()[None]))
    return out


def timed(fn, fr):
    ts = []
    for k, (x, imap) in enumerate(fr[:WARMUP] + fr):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(x, imap)
        e.record()
        torch.cuda.synchronize()
        if k >= WARMUP:
            ts.append(s.elapsed_time(e))
    return {"ms_per_frame": float(np.median(ts)), "ms_iqr": [float(np.percentile(ts, 25)), float(np.percentile(ts, 75))],
            "frames": len(ts)}


@torch.no_grad()
def main(out_path, backbone=None, settings=None):
    backbone = None if backbone in (None, "-") else backbone
    names = settings.split(",") if settings else SETTINGS
    check = out_path == "check"
    if not check and not gpu_idle():
        sys.exit("n7_timing: another process is using the GPU; wall clock needs an idle GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = EffModel(load_backbone("cuda", backbone) if backbone else load_backbone("cuda")).cuda()
    model.fast = True
    fr = frames()
    if check:
        for name in ["dense"] + names:
            gr = GraphRunner(model, name, HW)
            worst, classes = 0.0, set()
            for x, imap in fr[::10]:
                key, pm = gr.prepare(imap)
                classes.add(key)
                g = gr.run(x, imap).clone()
                m = None if name == "dense" else from_name(name, HW[0] // 16)
                if m is not None:
                    m.dilate, m.n_high_fixed = 0, key
                e = model(x, structure=m, dtype=torch.bfloat16, interest=pm).final
                worst = max(worst, (g - e).abs().max().item() / e.abs().max().item())
            print(f"{name:48s} classes {sorted(classes, key=str)}  max relative difference graph vs eager {worst:.2e}",
                  flush=True)
        return
    res = []

    def add(name, mode, fn):
        r = {"setting": name, "mode": mode, **timed(fn, fr)}
        res.append(r)
        print(f"{name:48s} {mode:6s} {r['ms_per_frame']:.2f} ms/frame", flush=True)

    add("dense", "eager", lambda x, i: model(x, dtype=torch.bfloat16))
    gd = GraphRunner(model, "dense", HW)
    add("dense", "graph", lambda x, i: gd.run(x))
    for name in names:
        add(name, "eager", lambda x, i: model(x, structure=from_name(name, HW[0] // 16), dtype=torch.bfloat16, interest=i))
        gr = GraphRunner(model, name, HW)
        for x, i in fr:                                                  # capture every class before timing
            gr.run(x, i)
        add(name, "graph", lambda x, i: gr.run(x, i))
        res[-1]["classes"] = len(gr.graphs)
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
