"""R40 plan section J3b (reported, no rule): where the redesign's wall clock goes, ViT-S and ViT-B, after J3 found it
slower than dense at ViT-S (`queue/r40_timing.json`). Same frames, maps and protocol as J3 (`r40_timing.py`). Per frame
of the redesign, CUDA events split the step into (a) the run-time state update outside the encoder (map motion and object
states), (b) the encoder call, and inside it (c) the expectation step (`ExpectMap._step`: expectation map, surprise check,
tiers); the host clock also gives the time until the encoder call returns, before the GPU is waited on (host time close
to the total means the step is bound by Python and kernel launches, not by GPU work). Beside it, as J3: dense (eager and
graph), today's design (eager and graph) and the backbone's ALGM fallback (eager), and the per-frame GPU kernels of
dense and the redesign, both eager, from the PyTorch profiler.
Usage (ml-base:vos, idle GPU): python -m src.stage_b.vos.r40_timing_parts <vits|vitb> [out json]
       python -m src.stage_b.vos.r40_timing_parts check   (CPU, fp32, a few frames: runs the code path, no timing claim)"""
import json
import sys
import time

import numpy as np
import torch

from src.stage_a.model import load_backbone
from src.stage_b.expect import ExpectMap, ExpectState, config_from_name, objects_from_predictions
from src.stage_b.graphs import GraphRunner
from src.stage_b.model import EffModel
from src.stage_b.p2_davis import shift_by_velocity
from src.stage_b.p2_timing import HW
from src.stage_b.r8_timing import REPO
from src.stage_b.timing import from_name, gpu_idle

from .r40_compare import Q, S_FINAL, SB_FINAL
from .r40_timing import WARMUP_SEQS, cov, run_timed, sequences

POINT = {"vits": (S_FINAL, "ftome_l0.7_t0.01_floor64_dil1_noprop_cls64"),
         "vitb": (SB_FINAL, "ftome_l0.7_t0.01_floor128_dil2_noprop_cls64")}


class Redesign:
    """The redesign's closed-loop step as in `r40_timing.main` (state reset per sequence), with optional CUDA-event
    marks around its parts."""

    def __init__(self, model, name, blocks, dt):
        self.model, self.dt = model, dt
        self.inner, rest = name.split("~")
        self.fb_name, self.exp_name = rest.split("@")
        self.blocks = blocks
        self.marks = None                                    # list of (label, start event, end event) for this frame
        self.lost = []

    def reset(self):
        st = ExpectState()
        cfg = config_from_name(self.exp_name)
        self.cfg, self.st, self.ostate = cfg, st, {}
        self.main = ExpectMap(from_name(self.inner, HW[0] // 16, self.blocks), st, config_from_name(self.exp_name))
        self.fb = ExpectMap(from_name(self.fb_name, HW[0] // 16, self.blocks), st, config_from_name(self.exp_name),
                            passive=True)
        for m in (self.main, self.fb):                       # (c): event marks around the expectation step
            step = m._step

            def timed_step(*a, step=step):
                if self.marks is None:
                    return step(*a)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                out = step(*a)
                e.record()
                self.marks.append(("expect_step", s, e))
                return out
            m._step = timed_step

    def step(self, fr):
        x, ids, prev1, prev2, _ = fr
        ev = self.marks is not None
        if ev:
            a = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            a[0].record()
        imap = cov((prev1 > 0) | (shift_by_velocity(prev1, prev2) > 0))
        self.st.objects, all_lost = objects_from_predictions(prev1, prev2, ids, self.ostate, self.cfg.motion)
        m = self.fb if (not bool(imap.any()) or all_lost) else self.main
        self.lost.append(m is self.fb)
        if ev:
            a[1].record()
        self.model(x, structure=m, dtype=self.dt, interest=imap)
        if ev:
            a[2].record()
            self.marks += [("state_update", a[0], a[1]), ("encoder", a[1], a[2])]


def parts(seqs, rd):
    """Per-frame medians (ms) of each part, and the host time until the step returns, after one warm-up sequence."""
    out = {"state_update": [], "encoder": [], "expect_step": [], "total": [], "host": []}
    for k, seq in enumerate(seqs[:WARMUP_SEQS] + seqs):
        rd.reset()
        for fr in seq:
            torch.cuda.synchronize()
            rd.marks = []
            t0 = time.perf_counter()
            rd.step(fr)
            host = 1000 * (time.perf_counter() - t0)
            torch.cuda.synchronize()
            if k >= WARMUP_SEQS:
                by = {}
                for label, s, e in rd.marks:
                    by[label] = by.get(label, 0.0) + s.elapsed_time(e)
                for label in ("state_update", "encoder", "expect_step"):
                    out[label].append(by.get(label, 0.0))
                out["total"].append(by["state_update"] + by["encoder"])
                out["host"].append(host)
            rd.marks = None
    return {f"{k}_ms": float(np.median(v)) for k, v in out.items()} | {"frames": len(out["total"])}


def launches(fn, frs):
    """Mean CUDA kernel launches per frame over `frs` (PyTorch profiler)."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for fr in frs:
            fn(fr)
        torch.cuda.synchronize()
    n = sum(e.count for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA)
    return n / len(frs)


@torch.no_grad()
def main(bb, out_json=None):
    check = bb == "check"
    bb = "vits" if check else bb
    dev, dt = ("cpu", torch.float32) if check else ("cuda", torch.bfloat16)
    if not check and not gpu_idle():
        sys.exit("r40_timing_parts: another process is using the GPU; wall clock needs an idle GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = EffModel(load_backbone(dev, REPO[bb]) if REPO[bb] else load_backbone(dev)).to(dev)
    model.fast = not check
    blocks = len(model.bb.model.layer)
    seqs = sequences(dev, 5, 2) if check else sequences()
    name, ours = POINT[bb]
    fb_name = name.split("~")[1].split("@")[0]
    rd = Redesign(model, name, blocks, dt)
    if check:
        for seq in seqs:
            rd.reset()
            for fr in seq:
                rd.step(fr)
        print("check passed: the redesign's step ran on the CPU; fallback share", float(np.mean(rd.lost)))
        return
    res = {"backbone": bb, "setting": name, "rows": []}

    def add(setting, mode, step, reset=lambda: None, **extra):
        r = {"setting": setting, "mode": mode, **run_timed(seqs, step, reset), **extra}
        res["rows"].append(r)
        print(f"{bb} {setting[:60]:60s} {mode:6s} {r['ms_per_frame']:.2f} ms/frame", flush=True)

    dense = lambda fr: model(fr[0], dtype=dt)
    add("dense", "eager", dense)
    gd = GraphRunner(model, "dense", HW)
    add("dense", "graph", lambda fr: gd.run(fr[0]))
    add(ours, "eager", lambda fr: model(fr[0], structure=from_name(ours, HW[0] // 16, blocks), dtype=dt, interest=fr[4]))
    go = GraphRunner(model, ours, HW)
    for seq in seqs:
        for fr in seq:
            go.run(fr[0], fr[4])
    add(ours, "graph", lambda fr: go.run(fr[0], fr[4]), classes=len(go.graphs))
    add(fb_name, "eager", lambda fr: model(fr[0], structure=from_name(fb_name, HW[0] // 16, blocks), dtype=dt))
    add(name, "eager", rd.step, rd.reset)
    res["fallback_share"] = float(np.mean(rd.lost[-res["rows"][-1]["frames"]:]))
    res["parts"] = parts(seqs, rd)
    print(bb, "parts", {k: round(v, 2) for k, v in res["parts"].items()}, flush=True)
    rd.reset()
    frs = seqs[0][:10]
    res["launches_per_frame"] = {"dense eager": launches(dense, frs), "redesign eager": launches(rd.step, frs)}
    print(bb, "launches", res["launches_per_frame"], flush=True)
    d = {r["mode"]: r["ms_per_frame"] for r in res["rows"] if r["setting"] == "dense"}
    for r in res["rows"]:
        r["speedup_vs_dense_eager"] = d["eager"] / r["ms_per_frame"]
        r["speedup_vs_dense_graph"] = d["graph"] / r["ms_per_frame"]
    out = out_json or str(Q / f"r40_timing_parts_{bb}.json")
    with open(out, "w") as f:
        f.write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
