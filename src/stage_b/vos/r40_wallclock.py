"""R40 plan section L on the idle GPU (rules in the plan).
`check <vits|vitb|vitl>` (L2): over the K2 frames (DAVIS, maps and object states from the previous ground-truth masks, as
J3), the frame runner with CUDA graphs against the same runner eager: decisions, token counts and patch assignment
equal and outputs within 1e-3 relative (fast path); agreement with the current path reported (R-37); capacity
overflows counted. Writes `r40_wallclock_check_<bb>.json`; exits non-zero on failure.
`time <vits|vitb|vitl> [variant ...]` (L3): J3's protocol (three DAVIS sequences, 40 frames, median after a warm-up
sequence, CUDA events around the whole step) per variant: `dense` (graphs), `K` (plan K's runner with the current
state update), `L` (the frame runner, margin 32), `L_m0`, `L_m64` (margins), `L_varlen` (variable-length flash in
place of the masked kernel). Counted FLOPs from the timed frames' states after timing; overflow share. Rows are
merged into `r40_wallclock_<bb>.json` by variant.
`profile <vits|vitb> [variant]`: kernels per frame and their summed GPU time (eager, the same kernels as the graphs),
and the graph run's wall time split at the decision read and the expectation step. Writes `r40_wallclock_prof_<bb>_<variant>.json`.
Usage (ml-base:vos; ViT-B with VOS_VIT=facebook/dinov3-vitb16-pretrain-lvd1689m, ViT-L (plan section N, after
`r34_algm.txt`) with VOS_VIT=facebook/dinov3-vitl16-pretrain-lvd1689m): python -m src.stage_b.vos.r40_wallclock <cmd> <bb> ..."""
import json
import sys

import numpy as np
import torch

from .r40_compare import Q
from .r40_graph import POINT, Loop, _setup


def _frame_runner(model, bb, **kw):
    from src.stage_b.frame_runner import FrameRunner
    from src.stage_b.p2_timing import HW
    return FrameRunner(model, HW, POINT[bb], **kw)


@torch.no_grad()
def check(bb, variant="L", n_frames=20):
    from src.stage_b.p2_timing import HW
    from .eval_lvos import deterministic
    from .r40_timing import sequences
    deterministic()
    model, blocks = _setup(bb)
    model.fast = True
    seqs = sequences("cuda", n_frames)
    kw = VARIANTS[variant]
    eager, graph = _frame_runner(model, bb, graphs=False, **kw), _frame_runner(model, bb, graphs=True, **kw)
    loop = Loop(POINT[bb], blocks)
    rows = []
    for seq in seqs:
        eager.reset(), graph.reset(), loop.reset()
        for t, fr in enumerate(seq):
            x, ids, p1, p2, _ = fr
            oe, se, _, fe = eager.run(x, ids, p1, p2)
            og, sg, _, fg = graph.run(x, ids, p1, p2)
            xc, mc, ic = loop.prepare(fr)
            sts = []
            ref = model(xc, structure=mc, dtype=torch.bfloat16, interest=ic, state_out=sts).final
            tok = lambda s: [tuple(map(int, a)) for a in s.tokens]
            rows.append({"decision": fe == fg == (mc is loop.fb), "tokens": tok(se) == tok(sg),
                         "src": bool(torch.equal(se.src, sg.src)), "rel": float((oe - og).abs().max() / oe.abs().max()),
                         "cur_tokens": tok(sts[0]) == tok(se), "cur_src": bool(torch.equal(sts[0].src, se.src))})
    r = {"backbone": bb, "variant": variant, "frames": len(rows), "decisions_equal": sum(x["decision"] for x in rows),
         "tokens_equal": sum(x["tokens"] for x in rows), "src_equal": sum(x["src"] for x in rows),
         "max_rel": max(x["rel"] for x in rows), "tolerance": 1e-3,
         "current_tokens_equal": sum(x["cur_tokens"] for x in rows), "current_src_equal": sum(x["cur_src"] for x in rows),
         "overflows_graph": graph.stats["overflow"], "graphs_captured": len(graph.runner.cache)}
    r["pass"] = bool(r["decisions_equal"] == r["tokens_equal"] == r["src_equal"] == r["frames"] and r["max_rel"] <= 1e-3)
    print(json.dumps(r), flush=True)
    import os
    tag = ("" if variant == "L" else "_" + variant) + ("_" + os.environ["R40_TAG"] if os.environ.get("R40_TAG") else "")
    r["inductor_opts"] = os.environ.get("R40_INDUCTOR_OPTS", "")
    (Q / f"r40_wallclock_check_{bb}{tag}.json").write_text(json.dumps(r, indent=1) + "\n")
    if not r["pass"]:
        sys.exit("r40_wallclock check: FAILED")


@torch.no_grad()
def diag(bb, variant="L", n_frames=20):
    """Frame by frame, the frame runner eager against the same with graphs: tiers fed to the encoder, the capacities
    chosen, overflow, token counts; then the same with capacities read after each block (margin None), and with the
    steps before the encoder eager (graphs for the encoder only). Prints; writes r40_wallclock_diag_<bb>.json."""
    from .eval_lvos import deterministic
    from .r40_timing import sequences
    deterministic()
    model, blocks = _setup(bb)
    model.fast = True
    seqs = sequences("cuda", n_frames)
    out = {}
    for tag, kw in (("spec", {}), ("no_spec", {"margin": None}), ("pre_eager", {"pre_graphs": False}),
                    ("no_spec_pre_eager", {"margin": None, "pre_graphs": False})):
        kw = {**VARIANTS[variant], **kw}
        e = _frame_runner(model, bb, graphs=False, **{k: v for k, v in kw.items() if k != "pre_graphs"})
        g = _frame_runner(model, bb, graphs=True, **kw)
        rows = []
        for seq in seqs:
            e.reset(), g.reset()
            for t, fr in enumerate(seq):
                x, ids, p1, p2, _ = fr
                ce, cg = e.caps(False), g.caps(False)
                oe, se, _, fe = e.run(x, ids, p1, p2)
                ti = e.runner.buf["interest0"].clone()
                og, sg, _, fg = g.run(x, ids, p1, p2)
                tg_ = g.runner.buf["interest0"]
                rows.append({"t": t, "fb": [fe, fg], "tiers": bool(torch.equal(ti, tg_)), "caps": ce == cg,
                             "tokens": [tuple(a) for a in se.tokens] == [tuple(a) for a in sg.tokens],
                             "src": bool(torch.equal(se.src, sg.src)), "max": float((oe - og).abs().max())})
        out[tag] = {"frames": len(rows), "tiers_equal": sum(r["tiers"] for r in rows), "caps_equal": sum(r["caps"] for r in rows),
                    "tokens_equal": sum(r["tokens"] for r in rows), "src_equal": sum(r["src"] for r in rows),
                    "first_bad": next((r for r in rows if not (r["tiers"] and r["src"])), None)}
        print(tag, json.dumps(out[tag]), flush=True)
    (Q / f"r40_wallclock_diag_{bb}.json").write_text(json.dumps(out, indent=1) + "\n")


VARIANTS = {"L": {}, "L_m0": {"margin": 0}, "L_m64": {"margin": 64}, "L_varlen": {"kernel": "varlen"},
            "L_c": {"compile": True}, "L_cv": {"compile": True, "kernel": "varlen"},
            "L_cs": {"compile": "segment"}, "L_csv": {"compile": "segment", "kernel": "varlen"},
            "L_csv16": {"compile": "segment", "kernel": "varlen", "step": 16},
            "L_csvo": {"compile": "segment", "kernel": "varlen", "overlap": True}}


@torch.no_grad()
def time_it(bb, variants):
    from src.stage_b.flops import dense_flops, image_flops
    from src.stage_b.graphs import GraphRunner
    from src.stage_b.methods import ops_of
    from src.stage_b.p2_timing import HW
    from src.stage_b.tiered_graph import TieredRunner
    from src.stage_b.timing import gpu_idle
    from .r40_timing import run_timed, sequences
    import os
    if not gpu_idle():
        sys.exit("r40_wallclock time: another process is using the GPU; wall clock needs an idle GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    det = os.environ.get("R40_DET") == "1"                       # the validation's deterministic kernels (R-29)
    if det:
        from .eval_lvos import deterministic
        deterministic()
    model, blocks = _setup(bb)
    seqs = sequences()
    N0 = seqs[0][0][4].shape[1]
    dn = dense_flops(N0, width=model.bb.config.hidden_size, blocks=blocks)
    path = Q / f"r40_wallclock_{bb}.json"
    res = json.loads(path.read_text()) if path.exists() else {"backbone": bb, "setting": POINT[bb], "rows": {}}
    for v in variants:
        saved, extra = [], {}
        if v == "dense":
            gd = GraphRunner(model, "dense", HW)
            step, reset = (lambda fr: gd.run(fr[0])), (lambda: None)
        elif v == "K":
            lp, runner = Loop(POINT[bb], blocks), TieredRunner(model, HW, fast=True, graphs=True)

            def step(fr, lp=lp, runner=runner):
                x, m, imap = lp.prepare(fr)
                _, st = runner.run(x, m, imap)
                saved.append((st, list(ops_of(m))))
            reset = lp.reset
        elif v.startswith("ALGM"):                               # the fallback (plain ALGM) on the same implementation
            from src.stage_b.model import State
            from src.stage_b.tiered_graph import P as NP
            fr_ = _frame_runner(model, bb, graphs=True, **VARIANTS[v.replace("ALGM", "L")])
            runner, spec, zt = fr_.runner, fr_.spec[True], fr_.zeros
            last = {}

            def step(fr, runner=runner, spec=spec, zt=zt, last=last, fr_=fr_):
                ops = []
                st = State(size=None, coords=None, src=None, carried=None, pairs=[], tokens=[], tiles=[], interest=None)
                fr_.last[True] = last.get("n")
                _, st = runner.encode(fr[0], spec, zt, ops, st, fr_.caps(True))
                last["n"] = [fr_.N0] + [t - NP for _, t in st.tokens]
                fr_.stats["frames"] += 1
                fr_.stats["overflow"] += runner.overflow
                saved.append((st, ops))
            reset = last.clear
        else:
            fr_ = _frame_runner(model, bb, graphs=True, **VARIANTS[v])

            def step(fr, fr_=fr_):
                x, ids, p1, p2, _ = fr
                _, st, ops, _ = fr_.run(x, ids, p1, p2)
                saved.append((st, ops))
            reset = fr_.reset
        for seq in seqs:                                         # capture pass: graphs for these frames
            reset()
            for fr in seq:
                step(fr)
        saved.clear()
        if v.startswith(("L", "ALGM")):
            fr_.stats.update(frames=0, fallback=0, overflow=0)
        r = run_timed(seqs, step, reset)
        if saved:
            k = r["frames"]
            fl = [image_flops(st, ops, 0, N0)["flops"] / dn for st, ops in saved[-k:]]
            r["counted_flops_ratio"] = float(np.mean(fl))
        if v.startswith(("L", "ALGM")):
            r["overflow_share"] = fr_.stats["overflow"] / max(1, fr_.stats["frames"])
            r["fallback_share"] = fr_.stats["fallback"] / max(1, fr_.stats["frames"])
            r["graphs"] = len(fr_.runner.cache)
        tag = v + ("_det" if det else "") + ("_" + os.environ["R40_TAG"] if os.environ.get("R40_TAG") else "")
        res["rows"][tag] = r
        print(bb, tag, json.dumps(r), flush=True)
        for name, row in res["rows"].items():                    # against dense with graphs under the same kernels
            d = res["rows"].get("dense_det" if name.endswith("_det") else "dense", {}).get("ms_per_frame")
            if d:
                row["speedup_vs_dense_graph"] = d / row["ms_per_frame"]
        path.write_text(json.dumps(res, indent=1) + "\n")       # after every variant (a later crash keeps these)


@torch.no_grad()
def tokens(bb, variants):
    """After L4 (reported, no rule): the final rows count more FLOPs than the eager rows at equal tiers and fallback
    frames (DAVIS: 26 of 30 videos), so fewer tokens merge. Over J3's frames (maps and object states from the previous
    ground-truth masks; the validation's deterministic kernels), each variant's token count entering every block
    against the current path's: `cur` (EffModel with the wrappers, fast bf16, as the eager rows), the frame-runner
    variants of VARIANTS (graphs), `ALGM_<v>` (the fallback on that implementation) against plain ALGM on the current
    path. Inductor options from R40_INDUCTOR_OPTS (e.g. emulate_precision_casts=1); rows tagged with R40_TAG. Merged
    into `r40_wallclock_tokens_<bb>.json`."""
    import os
    from src.stage_b.model import State
    from src.stage_b.tiered_graph import P as NP
    from .eval_lvos import deterministic
    from .r40_timing import sequences
    deterministic()
    model, blocks = _setup(bb)
    model.fast = True
    seqs = sequences()
    path = Q / f"r40_wallclock_tokens_{bb}.json"
    res = json.loads(path.read_text()) if path.exists() else {"backbone": bb, "setting": POINT[bb], "rows": {}}
    loops = {False: Loop(POINT[bb], blocks), True: Loop(POINT[bb], blocks)}    # one per path: prepare advances states
    cur, cur_fb = {False: [], True: []}, []                     # per frame, tokens entering each block (current path)
    for seq in seqs:
        for lp in loops.values():
            lp.reset()
        for fr in seq:
            for fb, lp in loops.items():
                xc, mc, ic = lp.prepare(fr, force_fb=fb)
                sts = []
                model(xc, structure=mc, dtype=torch.bfloat16, interest=ic, state_out=sts)
                cur[fb].append([int(t) for _, t in sts[0].tokens])
                if not fb:
                    cur_fb.append(mc is lp.fb)
    for v in variants:
        algm = v.startswith("ALGM")
        fr_ = _frame_runner(model, bb, graphs=True, **VARIANTS[v.replace("ALGM", "L")])
        got, same_fb = [], 0
        for seq in seqs:
            fr_.reset()
            for fr in seq:
                if algm:
                    st = State(size=None, coords=None, src=None, carried=None, pairs=[], tokens=[], tiles=[], interest=None)
                    _, st = fr_.runner.encode(fr[0], fr_.spec[True], fr_.zeros, [], st, fr_.caps(True))
                    fr_.last[True] = [fr_.N0] + [t - NP for _, t in st.tokens]
                else:
                    _, st, _, fbv = fr_.run(*fr[:4])
                    same_fb += fbv == cur_fb[len(got)]
                got.append([int(t) for _, t in st.tokens])
        ref = np.array(cur[algm], dtype=np.int64)
        d = np.array(got, dtype=np.int64) - ref
        tot = d.sum(1)
        first = [int(np.nonzero(r)[0][0]) for r in d if r.any()]
        tag = v + ("_" + os.environ["R40_TAG"] if os.environ.get("R40_TAG") else "")
        r = {"frames": len(got), "decisions_equal": len(got) if algm else same_fb, "inductor_opts": os.environ.get("R40_INDUCTOR_OPTS", ""),
             "tokens_mean": float(np.array(got).sum(1).mean()), "current_tokens_mean": float(ref.sum(1).mean()),
             "diff_mean": float(tot.mean()), "frames_more": int((tot > 0).sum()), "frames_fewer": int((tot < 0).sum()),
             "frames_equal": int((tot == 0).sum()), "frames_identical": int((~d.any(1)).sum()),
             "diff_mean_per_block": d.mean(0).round(3).tolist(),
             "first_block_differing": {str(b): first.count(b) for b in sorted(set(first))}}
        res["rows"][tag] = r
        print(bb, tag, json.dumps(r), flush=True)
        path.write_text(json.dumps(res, indent=1) + "\n")
        del fr_
        torch.cuda.empty_cache()


@torch.no_grad()
def profile(bb, variant="L", n=30):
    import time
    from torch.profiler import ProfilerActivity, profile as tprof
    from .r40_timing import sequences
    model, blocks = _setup(bb)
    seqs = sequences()
    frames = [fr for seq in seqs for fr in seq][:n]
    out = {"backbone": bb, "variant": variant}
    for graphs in (False, True):
        fr_ = _frame_runner(model, bb, graphs=graphs, **VARIANTS[variant])
        for _ in range(2):                                         # warm-up and capture
            fr_.reset()
            for f in frames:
                fr_.run(f[0], f[1], f[2], f[3])
        fr_.reset()
        if not graphs:
            with tprof(activities=[ProfilerActivity.CUDA]) as prof:
                for f in frames:
                    fr_.run(f[0], f[1], f[2], f[3])
                torch.cuda.synchronize()
            ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
            out["kernels_per_frame"] = len(ev) / n
            out["gpu_ms_per_frame"] = sum(e.device_time_total for e in ev) / 1000 / n
            top = {}
            for e in ev:
                top[e.name[:60]] = top.get(e.name[:60], 0) + e.device_time_total
            out["top_kernels_ms_per_frame"] = {k: round(v / 1000 / n, 4) for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:25]}
        else:
            ts, parts = [], []
            for f in frames:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fr_.marks = []
                fr_.run(f[0], f[1], f[2], f[3])
                torch.cuda.synchronize()
                ts.append(1000 * (time.perf_counter() - t0))
                m = fr_.marks
                parts.append([m[i].elapsed_time(m[i + 1]) for i in range(len(m) - 1)])
            fr_.marks = None
            out["graph_wall_ms_median"] = float(np.median(ts))
            out["stages_ms_median"] = dict(zip(["pre_to_read1", "read1_to_encoder", "encoder"],
                                               np.median(np.array(parts), 0).round(4).tolist()))
    print(json.dumps(out, indent=1), flush=True)
    (Q / f"r40_wallclock_prof_{bb}_{variant}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "check":
        check(a[1], *(a[2:3] or ["L"]))
    elif a[0] == "time":
        time_it(a[1], a[2:] or ["dense", "K", "L"])
    elif a[0] == "diag":
        diag(a[1], *(a[2:3] or ["L"]))
    elif a[0] == "profile":
        profile(a[1], *(a[2:3] or ["L"]))
    elif a[0] == "tokens":
        tokens(a[1], a[2:] or ["L", "L_varlen", "L_cs", "L_csv", "L_csvo", "ALGM_csvo"])
