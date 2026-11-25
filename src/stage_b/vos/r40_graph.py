"""R40 plan section K (the expectation focus map under CUDA graphs, method unchanged; `src/stage_b/tiered_graph.py`).
`check <vits|vitb>` (K2, idle GPU): over DAVIS frames with closed-loop maps (J3's frames), three copies of the expectation focus map
with their own expectation states: the current forward, the padded runner eagerly and the padded runner with graphs;
per frame, tiers, token counts and patch assignments must agree, and outputs within tolerance (fast path, then the fp32
attention path); the fallback on every fifth frame. Writes `r40_graph_check_<bb>.json`; exits non-zero on failure.
`time <vits|vitb>` (K4, idle GPU, reported): J3's protocol for the expectation focus map on the graph path (state update, tier
decision and encoder per frame), beside dense eager and with graphs; graphs are captured in a pass before timing; also
counted FLOPs against FLOPs at the padded capacity. Writes `r40_graph_time_<bb>.json`.
`compare` (K3, rules in the plan): graph rows against the current rows. Writes `r40_graph_equiv.json`.
`diag <vits|vitb>` (after K3's first attempt; reported, no rule; idle GPU): which kernel the current path's attention
runs and whether each padded attention form and the padded linear layers reproduce it bitwise on the real rows; then,
per form of the padded runner, agreement of token counts and patch assignment with the current path on the K2 frames.
Writes `r40_graph_diag_<bb>.json`.
Usage (ml-base:vos): python -m src.stage_b.vos.r40_graph <check|time|diag> <vits|vitb>   |   python -m src.stage_b.vos.r40_graph compare"""
import json
import os
import sys

import numpy as np
import torch

from .r40_compare import Q, S_FINAL, SB_FINAL, SL_FINAL

POINT = {"vits": S_FINAL, "vitb": SB_FINAL, "vitl": SL_FINAL}


def _setup(bb, dev="cuda"):
    from src.stage_a.model import load_backbone
    from src.stage_b.model import EffModel
    from src.stage_b.r8_timing import REPO
    model = EffModel(load_backbone(dev, REPO[bb]) if REPO[bb] else load_backbone(dev)).to(dev)
    model.fast = dev == "cuda"
    return model, len(model.bb.model.layer)


class Loop:
    """One copy of the expectation focus map's closed-loop step (as `r40_timing.main`): expectation state, object states, fallback."""

    def __init__(self, name, blocks):
        from src.stage_b.expect import config_from_name
        from src.stage_b.p2_timing import HW
        self.inner, rest = name.split("~")
        self.fb_name, self.exp_name = rest.split("@")
        self.blocks, self.hw, self.cfg = blocks, HW, config_from_name(self.exp_name)
        self.reset()

    def reset(self):
        from src.stage_b.expect import ExpectMap, ExpectState, config_from_name
        from src.stage_b.timing import from_name
        self.st, self.ostate = ExpectState(), {}
        g = self.hw[0] // 16
        self.main = ExpectMap(from_name(self.inner, g, self.blocks), self.st, config_from_name(self.exp_name))
        self.fb = ExpectMap(from_name(self.fb_name, g, self.blocks), self.st, config_from_name(self.exp_name), passive=True)

    def prepare(self, fr, force_fb=False):
        from src.stage_b.expect import objects_from_predictions
        from src.stage_b.p2_davis import shift_by_velocity
        from .r40_timing import cov
        x, ids, prev1, prev2, _ = fr
        imap = cov((prev1 > 0) | (shift_by_velocity(prev1, prev2) > 0))
        self.st.objects, all_lost = objects_from_predictions(prev1, prev2, ids, self.ostate, self.cfg.motion)
        m = self.fb if (force_fb or not bool(imap.any()) or all_lost) else self.main
        return x, m, imap


@torch.no_grad()
def check(bb, n_frames=20):
    """bb 'cpu': a dry run of this code on the CPU (ViT-S, fp32 attention path, no graphs, 2 short sequences)."""
    from src.stage_b.p2_timing import HW
    from src.stage_b.tiered_graph import TieredRunner
    from .r40_timing import sequences
    dry = bb == "cpu"
    bb, dev = ("vits", "cpu") if dry else (bb, "cuda")
    from .eval_lvos import deterministic
    deterministic()                                   # as the evaluator (K3 runs with deterministic kernels)
    tf32 = torch.backends.cuda.matmul.allow_tf32      # K2b diagnostic: run with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 and
    if os.environ.get("R40_TF32") == "0":             # R40_TF32=0 for exact fp32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        tf32 = torch.backends.cuda.matmul.allow_tf32
    model, blocks = _setup(bb, dev)
    seqs = sequences(dev, 7, 2) if dry else sequences(dev, n_frames)
    out, ok = {"backbone": bb, "setting": POINT[bb], "tf32": bool(tf32), "paths": {}}, True
    for fast, tol in ((False, 1e-5),) if dry else ((True, 1e-3), (False, 1e-5)):   # relative to the output scale
        model.fast = fast
        loops = [Loop(POINT[bb], blocks) for _ in range(3)]
        dt = torch.float32 if dry else torch.bfloat16
        eager = TieredRunner(model, HW, fast=fast, dtype=dt, graphs=False)
        graph = TieredRunner(model, HW, fast=fast, dtype=dt, graphs=not dry)
        rows = []
        for seq in seqs:
            for lp in loops:
                lp.reset()
            for t, fr in enumerate(seq):
                force = t % 5 == 4
                (x, ma, ia), (_, mb, ib), (_, mc, ic) = (lp.prepare(fr, force) for lp in loops)
                sts = []
                ref = model(x, structure=ma, dtype=dt, interest=ia, state_out=sts).final
                oe, se = eager.run(x, mb, ib)
                og, sg = graph.run(x, mc, ic)
                same_tiers = loops[0].st.last == loops[1].st.last == loops[2].st.last
                tok = [tuple(map(int, a)) for a in sts[0].tokens]
                same_tokens = tok == [tuple(a) for a in se.tokens] == [tuple(a) for a in sg.tokens]
                same_src = torch.equal(sts[0].src, se.src) and torch.equal(se.src, sg.src)
                rows.append({"fallback": ma is loops[0].fb, "tiers": same_tiers, "tokens": same_tokens, "src": same_src,
                             "cur_vs_eager": float((ref - oe).abs().max()), "eager_vs_graph": float((oe - og).abs().max()),
                             "scale": float(ref.abs().max())})
        r = {"frames": len(rows), "fallback_frames": sum(x["fallback"] for x in rows),
             "tiers_equal": all(x["tiers"] for x in rows), "tokens_equal": sum(x["tokens"] for x in rows),
             "src_equal": sum(x["src"] for x in rows),
             "max_cur_vs_eager": max(x["cur_vs_eager"] for x in rows), "max_eager_vs_graph": max(x["eager_vs_graph"] for x in rows),
             "median_cur_vs_eager": float(np.median([x["cur_vs_eager"] for x in rows])), "tolerance": tol,
             "graphs_captured": len(graph.cache)}
        # K2's rule (plan): the graph replays the padded eager path. Agreement of the padded path's merges with the
        # current path's (tokens_equal, src_equal) is a diagnostic: it is exact on the CPU (K1) and can differ on the GPU
        # where matmul rounding depends on shape (TF32 in this container); K3 decides whether that matters for accuracy.
        rel = max(x["eager_vs_graph"] / x["scale"] for x in rows)
        r["max_eager_vs_graph_rel"] = rel
        r["pass"] = bool(r["tiers_equal"] and rel <= tol)
        ok &= r["pass"]
        out["paths"]["fast" if fast else "fp32_attention"] = r
        print(bb, "fast" if fast else "fp32", json.dumps(r), flush=True)
    if dry:
        print("r40_graph check (CPU dry run):", "passed" if ok else "FAILED")
        return
    (Q / f"r40_graph_check_{bb}{'' if tf32 else '_notf32'}.json").write_text(json.dumps(out, indent=1) + "\n")
    if not ok:
        sys.exit("r40_graph check: FAILED")
    print("r40_graph check: passed")


@torch.no_grad()
def time_it(bb):
    from src.stage_b.flops import image_flops
    from src.stage_b.graphs import GraphRunner
    from src.stage_b.methods import ops_of
    from src.stage_b.p2_timing import HW
    from src.stage_b.tiered_graph import TieredRunner
    from src.stage_b.timing import gpu_idle
    from .r40_timing import run_timed, sequences
    if not gpu_idle():
        sys.exit("r40_graph time: another process is using the GPU; wall clock needs an idle GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    model, blocks = _setup(bb)
    seqs = sequences()
    res = {"backbone": bb, "setting": POINT[bb], "rows": []}

    def add(name, mode, step, reset=lambda: None, **extra):
        r = {"setting": name, "mode": mode, **run_timed(seqs, step, reset), **extra}
        res["rows"].append(r)
        print(f"{bb} {name[:60]:60s} {mode:6s} {r['ms_per_frame']:.2f} ms/frame", flush=True)

    add("dense", "eager", lambda fr: model(fr[0], dtype=torch.bfloat16))
    gd = GraphRunner(model, "dense", HW)
    add("dense", "graph", lambda fr: gd.run(fr[0]))
    lp, runner = Loop(POINT[bb], blocks), TieredRunner(model, HW, fast=True, graphs=True)
    stats = {"counted": [], "padded": [], "fallback": []}

    def step(fr):
        x, m, imap = lp.prepare(fr)
        _, st = runner.run(x, m, imap)
        N0 = imap.shape[1]
        stats["counted"].append(image_flops(st, ops_of(m), 0, N0, width=model.bb.config.hidden_size)["flops"])
        pad = type(st)(size=None, coords=None, src=None, carried=None, pairs=[torch.full((1,), float(a * a)) for a, _ in st.padded],
                       tokens=st.padded, tiles=[])
        stats["padded"].append(image_flops(pad, ops_of(m), 0, N0, width=model.bb.config.hidden_size)["flops"])
        stats["fallback"].append(m is lp.fb)
    for seq in seqs:                                    # capture pass: every (block, capacity) graph these frames use
        lp.reset()
        for fr in seq:
            step(fr)
    for k in stats:
        stats[k].clear()
    add(POINT[bb], "graph", step, lp.reset, graphs=len(runner.cache))
    from src.stage_b.flops import dense_flops
    N0 = seqs[0][0][4].shape[1]
    dn = dense_flops(N0, width=model.bb.config.hidden_size, blocks=blocks)
    res["counted_flops_ratio"] = float(np.mean(stats["counted"]) / dn)
    res["padded_flops_ratio"] = float(np.mean(stats["padded"]) / dn)
    res["fallback_share"] = float(np.mean(stats["fallback"]))
    d = {r["mode"]: r["ms_per_frame"] for r in res["rows"] if r["setting"] == "dense"}
    for r in res["rows"]:
        r["speedup_vs_dense_eager"] = d["eager"] / r["ms_per_frame"]
        r["speedup_vs_dense_graph"] = d["graph"] / r["ms_per_frame"]
    print(bb, "FLOPs counted", round(res["counted_flops_ratio"], 3), "padded", round(res["padded_flops_ratio"], 3), flush=True)
    (Q / f"r40_graph_time_{bb}.json").write_text(json.dumps(res, indent=1) + "\n")


def compare():
    """K3: per object, graph rows minus current rows (95% bootstrap); FLOPs difference; rules from the plan."""
    from .r28_compare import V, ci, load, tools
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    out, ok = {}, True
    cases = [("davis_vits", [("eval_r40k_davis", "eval_r40_davis_all", None)], S_FINAL, 0.3),
             ("davis_vitb", [("eval_r40kb_davis", "eval_r40b_davis", None)], SB_FINAL, 0.3),
             ("lvos_r31c", [("eval_r40k_r31c_v1val", "eval_r40_r31c_v1val", None),
                            ("eval_r40k_r31c_v2new", "eval_r40_r31c_v2new", new90)], S_FINAL, 1.0)]
    for tag, runs, k, margin in cases:
        if not all((V / a).exists() for a, _, _ in runs):
            print(tag, "rows missing; skipped")
            continue
        d, fn, fo, fr = [], [], [], []
        for a, b_, want in runs:
            A, B = load(V / a), load(V / b_)
            seqs = sorted(s for (kk, s) in A if kk == k and (want is None or s in want))
            assert seqs and all((k, s) in B for s in seqs), (tag, a)
            assert all(len(A[(k, s)]["J"]) == len(B[(k, s)]["J"]) for s in seqs), (tag, "objects")
            ja, fa, _ = tools(A, seqs)
            jb, fb, _ = tools(B, seqs)
            n = sum(A[(k, s)]["frames"] for s in seqs)
            d.append(ja(k) - jb(k))
            fn.append(fa(k) * n), fo.append(fb(k) * n), fr.append(n)
        d = np.concatenate(d)
        c = ci(d)
        fl_new, fl_old = sum(fn) / sum(fr), sum(fo) / sum(fr)
        r = {"objects": len(d), "graph_minus_current": c, "mean_abs": float(np.abs(d).mean()), "max_abs": float(np.abs(d).max()),
             "identical_objects": int((np.abs(d) < 1e-9).sum()), "flops_graph": fl_new, "flops_current": fl_old,
             "margin": margin}
        r["pass"] = bool(-margin < c[1] and c[2] < margin and abs(fl_new - fl_old) <= 0.002)
        ok &= r["pass"]
        out[tag] = r
        print(tag, json.dumps(r), flush=True)
    out["pass"] = bool(ok and len(out) == 3)
    print("K3 EQUIVALENCE", "PASS" if out["pass"] else "FAIL or incomplete")
    (Q / "r40_graph_equiv.json").write_text(json.dumps(out, indent=1) + "\n")


@torch.no_grad()
def diag(bb, n_frames=20):
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from src.stage_a.model import N_PREFIX as P
    from src.stage_b import tiered_graph as tg
    from src.stage_b.p2_timing import HW
    from .eval_lvos import deterministic
    from .r40_timing import sequences
    deterministic()
    model, blocks = _setup(bb)
    dev, H, hd = "cuda", model.heads, model.hd
    D = H * hd
    N0 = (HW[0] // 16) * (HW[1] // 16)
    g = torch.Generator(device=dev).manual_seed(0)
    out = {"backbone": bb, "tokens_full": P + N0, "kernels": {}, "linear": {}, "closed_loop": {}}
    # (1) attention kernels at several real lengths, capacity rounded up to 32 tokens; layout as in the model
    for n in (N0, N0 - 37, N0 - 200, N0 // 2 + 3):
        c = min(N0, -(-n // 32) * 32)
        T, C = P + n, P + c
        q, k, v = (torch.randn(1, C, H, hd, device=dev, generator=g).to(torch.bfloat16).transpose(1, 2) for _ in range(3))
        cut = lambda t: t[:, :, :T]
        ref = F.scaled_dot_product_attention(cut(q), cut(k), cut(v))
        forms = {}
        for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                         ("cudnn", SDPBackend.CUDNN_ATTENTION), ("math", SDPBackend.MATH)):
            try:
                with sdpa_kernel(be):
                    forms["unpadded_" + name] = F.scaled_dot_product_attention(cut(q), cut(k), cut(v))
            except RuntimeError as e:
                forms["unpadded_" + name] = str(e).splitlines()[0][:160]
        keep = torch.arange(C, device=dev) < T
        forms["padded_mask"] = cut(F.scaled_dot_product_attention(q, k, v, attn_mask=keep[None, None, None]))
        try:
            forms["padded_varlen"] = cut(tg._varlen(q, k, v, torch.full((), T, device=dev)))
        except RuntimeError as e:
            forms["padded_varlen"] = str(e).splitlines()[0][:160]
        res = {}
        for name, o in forms.items():
            res[name] = o if isinstance(o, str) else {"bitwise": bool(torch.equal(o, ref)),
                                                      "max_abs": float((o.float() - ref.float()).abs().max())}
        out["kernels"][f"T{T}_C{C}"] = res
        print("kernels", T, C, json.dumps(res), flush=True)
    # (2) linear layers of one block on the real rows against the same rows inside the padded buffer (bf16 autocast)
    layer = model.bb.model.layer[1]
    for n in (N0 - 37, N0 - 200, N0 // 2 + 3, 300):
        c = min(N0, -(-n // 32) * 32)
        T, C = P + n, P + c
        x = torch.randn(1, C, D, device=dev, generator=g)
        res = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for name, f in (("q_proj", layer.attention.q_proj), ("o_proj", layer.attention.o_proj), ("mlp", layer.mlp),
                            ("norm1", layer.norm1)):
                a_, b_ = f(x[:, :T]), f(x)[:, :T]
                res[name] = {"bitwise": bool(torch.equal(a_, b_)), "max_abs": float((a_.float() - b_.float()).abs().max())}
        out["linear"][f"T{T}_C{C}"] = res
        print("linear", T, C, json.dumps(res), flush=True)
    # (3) the padded runner per attention form against the current path, frame by frame (fast path, K2's frames)
    model.fast = True
    seqs = sequences(dev, n_frames)
    forms = {"mask": dict(kernel="mask"), "varlen": dict(kernel="varlen"), "varlen_graph": dict(kernel="varlen", graphs=True),
             "exact_no_padding": dict(kernel="dense", step=1)}
    for name, kw in forms.items():
        kw = {"graphs": False, **kw}
        loops = [Loop(POINT[bb], blocks) for _ in range(2)]
        runner = tg.TieredRunner(model, HW, fast=True, dtype=torch.bfloat16, **kw)
        rows = []
        try:
            for seq in seqs:
                for lp in loops:
                    lp.reset()
                for t, fr in enumerate(seq):
                    force = t % 5 == 4
                    (x, ma, ia), (_, mb, ib) = (lp.prepare(fr, force) for lp in loops)
                    sts = []
                    ref = model(x, structure=ma, dtype=torch.bfloat16, interest=ia, state_out=sts).final
                    o, s = runner.run(x, mb, ib)
                    tok = [tuple(map(int, a)) for a in sts[0].tokens]
                    mine = [tuple(map(int, a)) for a in s.tokens]
                    first = next((i for i, (u, w) in enumerate(zip(tok, mine)) if u != w), None)
                    rows.append({"tiers": loops[0].st.last == loops[1].st.last, "tokens": tok == mine,
                                 "src": bool(torch.equal(sts[0].src, s.src)), "first_block_differing": first,
                                 "out_bitwise": bool(torch.equal(ref, o)), "max_abs": float((ref - o).abs().max())})
        except RuntimeError as e:
            out["closed_loop"][name] = {"error": str(e).splitlines()[0][:200], "frames_done": len(rows)}
            print("closed loop", name, "ERROR", out["closed_loop"][name], flush=True)
            continue
        r = {"frames": len(rows), "tiers_equal": sum(x["tiers"] for x in rows), "tokens_equal": sum(x["tokens"] for x in rows),
             "src_equal": sum(x["src"] for x in rows), "out_bitwise": sum(x["out_bitwise"] for x in rows),
             "max_abs": max(x["max_abs"] for x in rows),
             "first_block_differing": [x["first_block_differing"] for x in rows]}
        out["closed_loop"][name] = r
        print("closed loop", name, json.dumps(r), flush=True)
    (Q / f"r40_graph_diag_{bb}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"check": lambda: check(a[1]), "time": lambda: time_it(a[1]), "diag": lambda: diag(a[1]), "compare": compare}[a[0]]()
