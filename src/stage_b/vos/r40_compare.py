"""R40 expectation focus map: gate and validation comparisons (rules fixed before any run: `reports/stage_b/queue/R40_expectation_plan.md`
section C). Paired J&F differences over objects, 1,000 bootstrap resamples (as the R-25 protocol).
`gate`: R26 head, LVOS V2 train; the new rows (`eval_r40_gate`, dense skipped) against DG1's rows of the same head and
kernels (`diag_dg1`: ALGM 0.98, ours). Pass: (1) gate subset: new − ours, lower bound above 0; (2) gate subset: new −
ALGM 0.98, lower bound above −3.0 at lower FLOPs; (3) all 420 videos: as (2).
`val`: the R-25 protocol: heads R31c, h2, h3; LVOS V1 val (50 videos) and the 90 V2 valid videos not in V1 val; per
object, the paired difference against ALGM 0.98 averaged over heads; non-inferiority if the lower bound is above −3.0
at lower head-averaged FLOPs, superiority if above 0.
`full`: plan section D (full resolution in the map against the baseline's rules).
`tome`: plan section E (the ToMe arm against plain ToMe r64, same protocol).
`d2val`: plan section G (the high tier under ALGM's rules at the final setting, validation protocol).
`tome_gate`, `tome2`: plan section I (the ToMe arm's second version: gate, then validation).
`davis_all`, `vitb`: plan section J (short clips with every baseline, ViT-S and ViT-B; ViT-B long videos).
`tome3_gate`, `tome3`: plan section M (the ToMe arm's third version: gate, validation).
`final`, `final_short`: plan section L4 (the validation, short clips and ViT-B long videos on the final implementation).
`vitl`: plan section N (R34, ViT-L: short clips and long videos, one head).
`vitb3`: plan section P (ViT-B heads r41_b1 to b3, all 140 LVOS V2 valid videos; section C's rules against ALGM 0.95
on our scorer, the official metric beside it; computed by `r40_official._long`, plus loop statistics).
Usage: python -m src.stage_b.vos.r40_compare <flops_sign [runs]|davis_final [run] [tag]|gate [run dir] [out json]|val|val_detail|full|tome|tome_detail|d2val|tome_gate|tome2|tome2_detail|davis_all|vitb|vitb3|vitl|davis>"""
import json
import sys
from pathlib import Path

import numpy as np

from .r28_compare import V, ci, load, tools

Q = Path("reports/stage_b/queue")
CLOSED_LOOP = "ftome_l0.7_t0.01_floor64_dil1_noprop_cls64~tome_r64_from1_noprop"
REF = "algm_t0.98"


def _same_objects(a, ka, b, kb, seqs):
    """Paired over objects: both runs must score the same objects in the same order on every video."""
    for s in seqs:
        assert a[(ka, s)]["objects"] == b[(kb, s)]["objects"], (s, ka, kb)


def _exp_stats(by, k, seqs):
    rows = [by[(k, s)] for s in seqs]
    fr = sum(r["frames"] for r in rows)
    return {x: sum(r.get(x, 0) * r["frames"] for r in rows) / fr for x in ("exp_high_share", "exp_mid_share")} | {
        x: sum(r.get(x, 0) for r in rows) for x in ("exp_global", "exp_lost_fallbacks", "fallback_frames", "frames")}


def gate(run_dir="eval_r40_gate", out_json="r40_gate.json"):
    """Second gate run (plan section F, after B4): `gate eval_r40_gate2 r40_gate2.json`."""
    run, ref = load(V / run_dir), load(V / "diag_dg1")
    sub = set((Q / "r40_gate_subset.txt").read_text().split())
    out = {}
    for k in sorted({k for (k, s) in run}):
        res = {}
        for tag, want in (("subset", sub), ("all", None)):
            seqs = sorted(s for (kk, s) in run if kk == k and (want is None or s in want))
            if want is None and len(seqs) < 420:
                res[tag] = {"videos": len(seqs), "complete": False}
                continue
            _same_objects(run, k, ref, REF, seqs)
            jn, fn, _ = tools(run, seqs)
            jr, fr, _ = tools(ref, seqs)
            a, o = jn(k) - jr(REF), jn(k) - jr(CLOSED_LOOP)
            res[tag] = {"videos": len(seqs), "objects": len(a), "jf": float(jn(k).mean()), "flops": fn(k),
                        "algm": {"jf": float(jr(REF).mean()), "flops": fr(REF)}, "closed_loop": {"jf": float(jr(CLOSED_LOOP).mean()), "flops": fr(CLOSED_LOOP)},
                        "minus_algm": ci(a), "minus_closed_loop": ci(o), "minus_dense": ci(jn(k) - jr("dense")),
                        "stats": _exp_stats(run, k, seqs)}
        s, a = res["subset"], res.get("all", {})
        c1 = s["minus_closed_loop"][1] > 0
        c2 = s["minus_algm"][1] > -3.0 and s["flops"] < s["algm"]["flops"]
        c3 = a.get("complete", True) and "minus_algm" in a and a["minus_algm"][1] > -3.0 and a["flops"] < a["algm"]["flops"]
        res["criteria"] = {"subset_beats_ours": bool(c1), "subset_noninferior_algm": bool(c2), "all_noninferior_algm": bool(c3),
                           "pass": bool(c1 and c2 and c3)}
        out[k] = res
        print(json.dumps({k: res}, indent=1))
    (Q / out_json).write_text(json.dumps(out, indent=1) + "\n")
    return out


HEADS = {"r31c": ("eval_r31c_lvos_det", "eval_r31c_v2val"), "r38_h2": ("eval_r38_h2_lvos", "eval_r38_h2_v2val"),
         "r38_h3": ("eval_r38_h3_lvos", "eval_r38_h3_v2val")}


TOME = "tome_r64_from1_noprop"
TOME_HEADS = {h: (f"eval_tome_{h}_v1val", f"eval_tome_{h}_v2new") for h in HEADS}


def val(new_prefix="eval_r40", heads=None, REF=REF, out_json="r40_val.json"):
    """The R-25 protocol against ALGM 0.98; `tome`: plan section E, the ToMe arm against plain ToMe r64."""
    heads = heads or HEADS
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    out, per_head = {}, {}
    keys = None
    for h, (r1, r2) in heads.items():
        n1, n2 = load(V / f"{new_prefix}_{h}_v1val"), load(V / f"{new_prefix}_{h}_v2new")
        b1, b2 = load(V / r1), load(V / r2)
        ks = sorted({k for (k, s) in n1})
        keys = ks if keys is None else keys
        assert ks == keys, (h, ks, keys)
        s1 = sorted(s for (k, s) in n1 if k == ks[0])
        s2 = sorted(s for (k, s) in n2 if k == ks[0] and s in new90)
        assert len(s1) == 50 and len(s2) == 90, (h, len(s1), len(s2))
        per_head[h] = {}
        for k in ks:
            _same_objects(n1, k, b1, REF, s1)
            _same_objects(n2, k, b2, REF, s2)
            j1, f1, _ = tools(n1, s1); j2, f2, _ = tools(n2, s2)
            g1, e1, _ = tools(b1, s1); g2, e2, _ = tools(b2, s2)
            fr1 = sum(n1[(k, s)]["frames"] for s in s1); fr2 = sum(n2[(k, s)]["frames"] for s in s2)
            per_head[h][k] = {"diff": np.concatenate([j1(k) - g1(REF), j2(k) - g2(REF)]),
                              "jf": np.concatenate([j1(k), j2(k)]), "algm": np.concatenate([g1(REF), g2(REF)]),
                              "flops": (f1(k) * fr1 + f2(k) * fr2) / (fr1 + fr2),
                              "algm_flops": (e1(REF) * fr1 + e2(REF) * fr2) / (fr1 + fr2)}
    for k in keys:
        d = np.mean([per_head[h][k]["diff"] for h in heads], 0)
        fl = float(np.mean([per_head[h][k]["flops"] for h in heads]))
        fa = float(np.mean([per_head[h][k]["algm_flops"] for h in heads]))
        c = ci(d)
        out[k] = {"objects": len(d), "minus_algm_head_avg": c, "flops": fl, "algm_flops": fa,
                  "per_head": {h: {"minus_algm": ci(per_head[h][k]["diff"]), "jf": float(per_head[h][k]["jf"].mean()),
                                   "algm_jf": float(per_head[h][k]["algm"].mean()), "flops": per_head[h][k]["flops"],
                                   "algm_flops": per_head[h][k]["algm_flops"]} for h in heads}, "reference": REF,
                  "noninferior": bool(c[1] > -3.0 and fl < fa), "superior": bool(c[1] > 0 and fl < fa)}
        print(json.dumps({k: out[k]}, indent=1))
    (Q / out_json).write_text(json.dumps(out, indent=1) + "\n")
    return out


def val_detail(new_prefix="eval_r40", heads=None, REF=REF, out_json="r40_val_detail.json"):
    """Reported with the validation, no rule: per dataset (V1 val, the 90 new V2 valid) and overall, the head-averaged
    per-object difference against the reference and against dense (dense rows from the ALGM reference runs of the same
    head, video and kernels), with FLOPs and loop statistics. `tome_detail`: the same for the ToMe arm."""
    heads = heads or HEADS
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    acc = {}
    for h, (r1, r2) in heads.items():
        for tag, nd, bd, dd, want in (("v1val", f"{new_prefix}_{h}_v1val", r1, HEADS[h][0], None),
                                      ("v2new", f"{new_prefix}_{h}_v2new", r2, HEADS[h][1], new90)):
            n, b = load(V / nd), load(V / bd)
            dn = b if dd == bd else load(V / dd)
            for k in sorted({k for (k, s) in n}):
                seqs = sorted(s for (kk, s) in n if kk == k and (want is None or s in want))
                _same_objects(n, k, b, REF, seqs)
                _same_objects(n, k, dn, "dense", seqs)
                jn, fn, _ = tools(n, seqs)
                jb, fb, _ = tools(b, seqs)
                jd, _, _ = tools(dn, seqs)
                fr = sum(n[(k, s)]["frames"] for s in seqs)
                acc.setdefault(k, {}).setdefault(tag, {})[h] = {
                    "ref": jn(k) - jb(REF), "dense": jn(k) - jd("dense"), "jf": jn(k), "ref_jf": jb(REF), "dense_jf": jd("dense"),
                    "flops": fn(k), "ref_flops": fb(REF), "frames": fr, "stats": _exp_stats(n, k, seqs)}
    out = {}
    for k, by in acc.items():
        out[k] = {}
        for tag in ("v1val", "v2new", "all"):
            parts = ("v1val", "v2new") if tag == "all" else (tag,)
            cat = lambda h, x: np.concatenate([by[p][h][x] for p in parts])
            wf = lambda h, x: sum(by[p][h][x] * by[p][h]["frames"] for p in parts) / sum(by[p][h]["frames"] for p in parts)
            st = lambda x: sum(by[p][h]["stats"][x] for p in parts for h in heads)
            out[k][tag] = {"objects": len(cat(list(heads)[0], "ref")),
                           "jf": float(np.mean([cat(h, "jf").mean() for h in heads])),
                           "ref_jf": float(np.mean([cat(h, "ref_jf").mean() for h in heads])),
                           "dense_jf": float(np.mean([cat(h, "dense_jf").mean() for h in heads])),
                           "minus_ref": ci(np.mean([cat(h, "ref") for h in heads], 0)),
                           "minus_dense": ci(np.mean([cat(h, "dense") for h in heads], 0)),
                           "flops": float(np.mean([wf(h, "flops") for h in heads])),
                           "ref_flops": float(np.mean([wf(h, "ref_flops") for h in heads])),
                           "global_frame_share": st("exp_global") / st("frames"),
                           "fallback_frame_share": st("fallback_frames") / st("frames"),
                           "mid_share": sum(by[p][h]["stats"]["exp_mid_share"] * by[p][h]["stats"]["frames"]
                                            for p in parts for h in heads) / st("frames")}
    print(json.dumps(out, indent=1))
    (Q / out_json).write_text(json.dumps(out, indent=1) + "\n")
    return out


E_FINAL = "exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1"
S_FINAL = f"titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98@{E_FINAL}"
D2_FINAL = f"titome_t0.98_l0.7_floor64_dil1_cls64_ha1~algm_t0.98@{E_FINAL}"


def d2val():
    """Plan section G (owner question after the validation): the high tier under ALGM's rules at the final setting
    (`eval_r40d2_*`) against the expectation focus map (`eval_r40_*`, the validation rows) and against ALGM 0.98, validation protocol
    (head-averaged per object). Goes to the owner if the difference to the expectation focus map has a lower bound above −1.0 at FLOPs
    at least 10% lower; otherwise full resolution in the map stands. No design change on the validation split alone."""
    a = val("eval_r40d2", {h: (f"eval_r40_{h}_v1val", f"eval_r40_{h}_v2new") for h in HEADS}, S_FINAL,
            "r40_d2val_vs_method.json")[D2_FINAL]
    b = val("eval_r40d2", HEADS, REF, "r40_d2val_vs_algm.json")[D2_FINAL]
    out = {"setting": D2_FINAL, "objects": a["objects"], "flops": a["flops"],
           "minus_method": a["minus_algm_head_avg"], "method_flops": a["algm_flops"],
           "minus_method_per_head": {h: v["minus_algm"] for h, v in a["per_head"].items()},
           "minus_algm": b["minus_algm_head_avg"], "algm_flops": b["algm_flops"],
           "minus_algm_per_head": {h: v["minus_algm"] for h, v in b["per_head"].items()},
           "noninferior_algm": b["noninferior"], "superior_algm": b["superior"]}
    out["to_owner"] = bool(out["minus_method"][1] > -1.0 and out["flops"] <= 0.9 * out["method_flops"])
    print(json.dumps(out, indent=1))
    (Q / "r40_d2val.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


TT2 = f"ttome_r64_l0.7_floor64_dil1_cls64_tr1~tome_r64_from1_noprop@{E_FINAL}"
TT3 = f"ttome_r64_l0.7_floor64_dil1_cls64_tr1~tome_r64_from1_noprop@membase@{E_FINAL}-mb"   # plan M


def tome_gate(run_dir="eval_r40f_gate", out_json="r40_tome_gate.json", new=None, tome_dir=None):
    """Plan section I: the ToMe arm's second version on the gate (R26 head, LVOS V2 train) against plain ToMe r64 (rows of
    the same jobs) and ours (DG1, its ToMe-fallback form). Pass: (1) subset: new − ours above 0; (2) subset: new − plain
    ToMe above −3.0 at lower FLOPs; (3) all 420: as (2)."""
    TT = new or TT2                                      # plan M: the third version, plain ToMe rows from section I's run
    run, ref = load(V / run_dir), load(V / "diag_dg1")
    tr = load(V / tome_dir) if tome_dir else run
    sub = set((Q / "r40_gate_subset.txt").read_text().split())
    res = {}
    for tag, want in (("subset", sub), ("all", None)):
        seqs = sorted(s for (k, s) in run if k == TT and (want is None or s in want))
        if want is None and len(seqs) < 420:
            res[tag] = {"videos": len(seqs), "complete": False}
            continue
        _same_objects(run, TT, tr, TOME, seqs)
        _same_objects(run, TT, ref, CLOSED_LOOP, seqs)
        jn, fn, _ = tools(run, seqs)
        jt, ft, _ = tools(tr, seqs)
        jr, fr, _ = tools(ref, seqs)
        res[tag] = {"videos": len(seqs), "objects": len(jn(TT)), "jf": float(jn(TT).mean()), "flops": fn(TT),
                    "tome": {"jf": float(jt(TOME).mean()), "flops": ft(TOME)}, "closed_loop": {"jf": float(jr(CLOSED_LOOP).mean()), "flops": fr(CLOSED_LOOP)},
                    "minus_tome": ci(jn(TT) - jt(TOME)), "minus_closed_loop": ci(jn(TT) - jr(CLOSED_LOOP)),
                    "minus_dense": ci(jn(TT) - jr("dense")), "stats": _exp_stats(run, TT, seqs)}
        if new:
            res[tag]["switched"] = {k: sum(run[(TT, s)].get(k, 0) for s in seqs) for k in ("mbase_frames", "membase_frames")}
    s, a = res["subset"], res.get("all", {})
    c1 = s["minus_closed_loop"][1] > 0
    c2 = s["minus_tome"][1] > -3.0 and s["flops"] < s["tome"]["flops"]
    c3 = a.get("complete", True) and "minus_tome" in a and a["minus_tome"][1] > -3.0 and a["flops"] < a["tome"]["flops"]
    res["criteria"] = {"subset_beats_ours": bool(c1), "subset_noninferior_tome": bool(c2), "all_noninferior_tome": bool(c3),
                       "pass": bool(c1 and c2 and c3)}
    print(json.dumps({TT: res}, indent=1))
    (Q / out_json).write_text(json.dumps({TT: res}, indent=1) + "\n")
    print("TOME GATE", "PASS" if res["criteria"]["pass"] else "FAIL")
    return res


def tome3_gate():
    """Plan section M: the ToMe arm's third version on section I's gate (`eval_r40g_gate`; plain ToMe rows of
    `eval_r40f_gate`, same head, videos and kernels)."""
    return tome_gate("eval_r40g_gate", "r40_tome3_gate.json", TT3, "eval_r40f_gate")


def tome3():
    """Plan section M: validation of the third version (section E's protocol), then its detail."""
    out = val("eval_r40g", TOME_HEADS, TOME, "r40_tome3_val.json")
    val_detail("eval_r40g", TOME_HEADS, TOME, "r40_tome3_val_detail.json")
    return out


def tome2():
    """Plan section I: validation of the ToMe arm's second version (`eval_r40f_*`) against plain ToMe r64 (section E's rows)."""
    return val("eval_r40f", TOME_HEADS, TOME, "r40_tome2_val.json")


def tome2_detail():
    return val_detail("eval_r40f", TOME_HEADS, TOME, "r40_tome2_val_detail.json")


SB_FINAL = f"titome_t0.95_l0.7_floor128_dil2_cls64~algm_t0.95@{E_FINAL}"


def _sl_final():
    """Plan section N (R34): the expectation focus map at ViT-L, ALGM at the threshold R25's rule picks on DAVIS (`r34_algm.txt`;
    None before the pick), budget floor 128 and margin 1 from R4's ViT-L choice."""
    p = Q / "r34_algm.txt"
    if not p.exists():
        return None
    t = p.read_text().strip().removeprefix("algm_t")
    return f"titome_t{t}_l0.7_floor128_dil1_cls64~algm_t{t}@{E_FINAL}"


SL_FINAL = _sl_final()


def davis_all():
    """Plan section J1 (reported): short clips inside Cutie with every baseline, one run per backbone (ViT-S R26 head,
    ViT-B R25 head; fast path, deterministic kernels): each setting's J&F, FLOPs and change from dense; the expectation focus map
    against every other setting (paired over objects)."""
    out = {}
    for tag, run, new in (("vits", "eval_r40_davis_all", S_FINAL), ("vitb", "eval_r40b_davis", SB_FINAL)):
        if not (V / run).exists():
            continue
        b = load(V / run)
        seqs = sorted(s for (k, s) in b if k == "dense")
        ks = sorted({k for (k, s) in b} - {"dense"})
        assert all((k, s) in b for k in ks for s in seqs), (tag, "incomplete")
        assert all(len(b[(k, s)]["J"]) == len(b[("dense", s)]["J"]) for k in ks for s in seqs), (tag, "objects")
        jf, fl, _ = tools(b, seqs)
        res = {"videos": len(seqs), "objects": len(jf("dense")), "dense": float(jf("dense").mean()), "settings": {}}
        for k in ks:
            res["settings"][k] = {"jf": float(jf(k).mean()), "flops": fl(k), "minus_dense": ci(jf(k) - jf("dense"))}
            if k != new:
                res["settings"][k]["method_minus"] = ci(jf(new) - jf(k))
        out[tag] = res
    print(json.dumps(out, indent=1))
    (Q / "r40_davis_all.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def davis_final(run="eval_r40n", out_tag="final_thr"):
    """Plan section L4c (reported): Tables 3 and 4 with the expectation focus map's row from the final implementation (`<run>_davis`,
    `<run>b_davis`) in place of J1's eager row; every other setting from J1's runs, paired over objects as `davis_all`
    (`r40_<out_tag>_davis_all.json`)."""
    out = {}
    for tag, rd, ref, new in (("vits", f"{run}_davis", "eval_r40_davis_all", S_FINAL),
                              ("vitb", f"{run}b_davis", "eval_r40b_davis", SB_FINAL)):
        if not (V / rd).exists():
            continue
        a, b = load(V / rd), load(V / ref)
        seqs = sorted(s for (k, s) in b if k == "dense")
        assert all((new, s) in a for s in seqs), (tag, "incomplete")
        assert all(len(a[(new, s)]["J"]) == len(b[("dense", s)]["J"]) for s in seqs), (tag, "objects")
        ja, fa, _ = tools(a, seqs)
        jf, fl, _ = tools(b, seqs)
        res = {"videos": len(seqs), "objects": len(jf("dense")), "dense": float(jf("dense").mean()), "run": rd,
               "method": {"jf": float(ja(new).mean()), "flops": fa(new), "minus_dense": ci(ja(new) - jf("dense"))},
               "settings": {}}
        for k in sorted({k for (k, s) in b} - {"dense", new}):
            res["settings"][k] = {"jf": float(jf(k).mean()), "flops": fl(k), "minus_dense": ci(jf(k) - jf("dense")),
                                  "method_minus": ci(ja(new) - jf(k))}
        out[tag] = res
    print(json.dumps(out, indent=1))
    (Q / f"r40_{out_tag}_davis_all.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def vitb():
    """Plan section J2 (reported, one head): ViT-B long videos (R25 head; LVOS V1 val and the 90 new V2 valid): the
    expectation focus map for ViT-B against plain ALGM 0.95 and dense, per split and pooled, per object."""
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    parts, out = {}, {"setting": SB_FINAL, "reference": "algm_t0.95", "head": "r25"}
    for tag, run, want in (("v1val", "eval_r40b_v1val", None), ("v2new", "eval_r40b_v2new", new90)):
        b = load(V / run)
        seqs = sorted(s for (k, s) in b if k == SB_FINAL and (want is None or s in want))
        assert len(seqs) == (50 if want is None else 90), (tag, len(seqs))
        for k in ("algm_t0.95", "dense"):
            _same_objects(b, SB_FINAL, b, k, seqs)
        jf, fl, _ = tools(b, seqs)
        fr = sum(b[(SB_FINAL, s)]["frames"] for s in seqs)
        parts[tag] = {"new": jf(SB_FINAL), "ref": jf("algm_t0.95"), "dense": jf("dense"), "frames": fr,
                      "flops": fl(SB_FINAL), "ref_flops": fl("algm_t0.95"), "stats": _exp_stats(b, SB_FINAL, seqs)}
    for tag in ("v1val", "v2new", "all"):
        ps = [parts[p] for p in (("v1val", "v2new") if tag == "all" else (tag,))]
        cat = lambda x: np.concatenate([p[x] for p in ps])
        wf = lambda x: sum(p[x] * p["frames"] for p in ps) / sum(p["frames"] for p in ps)
        fr = sum(p["stats"]["frames"] for p in ps)
        out[tag] = {"objects": len(cat("new")), "jf": float(cat("new").mean()), "ref_jf": float(cat("ref").mean()),
                    "dense_jf": float(cat("dense").mean()), "minus_ref": ci(cat("new") - cat("ref")),
                    "minus_dense": ci(cat("new") - cat("dense")), "ref_minus_dense": ci(cat("ref") - cat("dense")),
                    "flops": wf("flops"), "ref_flops": wf("ref_flops"),
                    "global_frame_share": sum(p["stats"]["exp_global"] for p in ps) / fr,
                    "fallback_frame_share": sum(p["stats"]["fallback_frames"] for p in ps) / fr}
    print(json.dumps(out, indent=1))
    (Q / "r40_vitb.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def vitl():
    """Plan section N (reported, one head, R34 head): ViT-L, on the rows of section O's amendment (all 140 LVOS V2 valid
    videos in one run per method, masks saved). Short clips: every R34 setting (`eval_oml_davis_all`) and the expectation focus map's
    row on the final implementation (`eval_oml_davis_r40`) against dense, the expectation focus map against each setting (paired over
    objects). Long videos: the expectation focus map (`eval_oml_r40`) against the picked plain ALGM (`r34_algm.txt`) and dense
    (`eval_oml_base`), per split (LVOS V1 val, the 90 new) and pooled, per object (`r40_vitl.json`)."""
    ref, new = (Q / "r34_algm.txt").read_text().strip(), SL_FINAL
    out = {"setting": new, "reference": ref, "head": "r34"}
    a, b = load(V / "eval_oml_davis_r40"), load(V / "eval_oml_davis_all")
    seqs = sorted(s for (k, s) in b if k == "dense")
    assert len(seqs) == 30 and all((new, s) in a for s in seqs), "davis incomplete"
    assert all(len(a[(new, s)]["J"]) == len(b[("dense", s)]["J"]) for s in seqs), "objects"
    ja, fa, _ = tools(a, seqs)
    jf, fl, _ = tools(b, seqs)
    out["davis"] = {"videos": len(seqs), "objects": len(jf("dense")), "dense": float(jf("dense").mean()),
                    "method": {"jf": float(ja(new).mean()), "flops": fa(new), "minus_dense": ci(ja(new) - jf("dense"))},
                    "settings": {k: {"jf": float(jf(k).mean()), "flops": fl(k), "minus_dense": ci(jf(k) - jf("dense")),
                                     "method_minus": ci(ja(new) - jf(k))} for k in sorted({k for (k, s) in b} - {"dense"})}}
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    a, b = load(V / "eval_oml_r40"), load(V / "eval_oml_base")
    allseq = sorted(s for (k, s) in a if k == new)
    assert len(allseq) == 140, len(allseq)
    parts = {}
    for tag, seqs in (("v1val", [s for s in allseq if s not in new90]), ("v2new", [s for s in allseq if s in new90])):
        assert len(seqs) == (50 if tag == "v1val" else 90), (tag, len(seqs))
        for k in (ref, "dense"):
            _same_objects(a, new, b, k, seqs)
        ja, fa, _ = tools(a, seqs)
        jb, fb_, _ = tools(b, seqs)
        parts[tag] = {"new": ja(new), "ref": jb(ref), "dense": jb("dense"), "frames": sum(a[(new, s)]["frames"] for s in seqs),
                      "flops": fa(new), "ref_flops": fb_(ref), "stats": _exp_stats(a, new, seqs)}
    for tag in ("v1val", "v2new", "all"):
        ps = [parts[p] for p in (("v1val", "v2new") if tag == "all" else (tag,))]
        cat = lambda x: np.concatenate([p[x] for p in ps])
        wf = lambda x: sum(p[x] * p["frames"] for p in ps) / sum(p["frames"] for p in ps)
        fr = sum(p["stats"]["frames"] for p in ps)
        out[tag] = {"objects": len(cat("new")), "jf": float(cat("new").mean()), "ref_jf": float(cat("ref").mean()),
                    "dense_jf": float(cat("dense").mean()), "minus_ref": ci(cat("new") - cat("ref")),
                    "minus_dense": ci(cat("new") - cat("dense")), "ref_minus_dense": ci(cat("ref") - cat("dense")),
                    "flops": wf("flops"), "ref_flops": wf("ref_flops"),
                    "global_frame_share": sum(p["stats"]["exp_global"] for p in ps) / fr,
                    "fallback_frame_share": sum(p["stats"]["fallback_frames"] for p in ps) / fr}
    print(json.dumps(out, indent=1))
    (Q / "r40_vitl.json").write_text(json.dumps(out, indent=1) + "\n")
    return out

def vitb3():
    """Plan section P: the three-head protocol at ViT-B (`r40_vitb3.json`). The comparison is `r40_official.vitb3`
    (head-averaged per-object differences, both scorers, section C's rules on ours); added here: fallback and
    global-motion frame shares and the tier shares over the three heads."""
    from .r40_official import vitb3 as off                              # local: r40_official imports this module
    out = off()
    st = [_exp_stats(b, SB_FINAL, sorted(s for (k, s) in b if k == SB_FINAL))
          for b in (load(V / f"eval_omb_b{h}_r40") for h in (1, 2, 3))]
    fr = sum(x["frames"] for x in st)
    out["fallback_frame_share"] = sum(x["fallback_frames"] for x in st) / fr
    out["global_frame_share"] = sum(x["exp_global"] for x in st) / fr
    out["mid_share"] = sum(x["exp_mid_share"] * x["frames"] for x in st) / fr
    out["high_share"] = sum(x["exp_high_share"] * x["frames"] for x in st) / fr
    print(json.dumps(out, indent=1))
    (Q / "r40_vitb3.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def iframe_gate():
    """Plan section Q (development): each I-frame variant (`eval_r40q_<v>`, R26 head, reference implementation) against
    the method's rows on the same videos and kernels (`eval_r40_gate2`): the long videos (`r40_q_long.txt`), the sample
    (`r40_q_sample.txt`), FLOPs on all 104 against plain ALGM 0.98 (`diag_dg1`). Writes `r40_q_gate.json`."""
    long_, samp = ((Q / f).read_text().split() for f in ("r40_q_long.txt", "r40_q_sample.txt"))
    ref, algm = load(V / "eval_r40_gate2"), load(V / "diag_dg1")
    out, cands = {}, []
    for v in ("key20", "key30", "key10", "lostkey", "fbkey20", "fbkey10", "lostfb"):
        k = f"{S_FINAL}@{v}"
        if not (V / f"eval_r40q_{v}").exists():
            continue
        n = load(V / f"eval_r40q_{v}")
        res = {}
        for tag, seqs in (("long", sorted(long_)), ("sample", sorted(samp))):
            _same_objects(n, k, ref, S_FINAL, seqs)
            jn, fn, _ = tools(n, seqs)
            jr, fr, _ = tools(ref, seqs)
            res[tag] = {"objects": len(jn(k)), "minus_method": ci(jn(k) - jr(S_FINAL)), "flops": fn(k), "method_flops": fr(S_FINAL)}
        allv = sorted(long_ + samp)
        _, fn, _ = tools(n, allv)
        _, fa, _ = tools(algm, allv)
        res["flops_all"], res["algm_flops_all"] = fn(k), fa(REF)
        res["iframes"] = sum(n[(k, s)].get("iframes", 0) for s in allv)
        res["candidate"] = bool(res["long"]["minus_method"][0] >= 1.0 and res["sample"]["minus_method"][1] > -1.0
                                and res["flops_all"] < res["algm_flops_all"])
        out[v] = res
        if res["candidate"]:
            cands.append((-res["long"]["minus_method"][0], res["flops_all"], v))
    out["chosen"] = sorted(cands)[0][2] if cands else None
    pairs = {"periodic (20)": ("key20", "fbkey20"), "periodic (10)": ("key10", "fbkey10"),
             "loss-triggered": ("lostkey", "lostfb")}                 # does the fallback suffice?
    out["dense_vs_fallback"] = {name: {"dense_long": out[d]["long"]["minus_method"], "fallback_long": out[f]["long"]["minus_method"],
                                       "dense_flops": out[d]["flops_all"], "fallback_flops": out[f]["flops_all"]}
                                for name, (d, f) in pairs.items() if d in out and f in out}
    print(json.dumps(out, indent=1))
    print(f"Q GATE: {out['chosen'] or 'none'}")
    (Q / "r40_q_gate.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def iframe_val():
    """Plan section Q (held-out, reported ablation): the chosen variant (`eval_r40q_<head>_*`) against the method's
    reference rows (`eval_r40_<head>_*`) and ALGM 0.98, head-averaged per object; all objects, objects in videos longer
    than 600 frames, the 90 new videos. Writes `r40_q_val.json`."""
    rg = Q / "r40_r_gate.json"
    v = json.loads(rg.read_text())["final_choice"] if rg.exists() else json.loads((Q / "r40_q_gate.json").read_text())["chosen"]
    k = f"{S_FINAL}@{v}"
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    per = {}
    for h, (r1, r2) in HEADS.items():
        for split, bdir, want in (("v1val", r1, None), ("v2new", r2, new90)):
            n, m, b = load(V / f"eval_r40q_{h}_{split}"), load(V / f"eval_r40_{h}_{split}"), load(V / bdir)
            seqs = sorted(s for (kk, s) in n if kk == k and (want is None or s in want))
            _same_objects(n, k, m, S_FINAL, seqs)
            _same_objects(n, k, b, REF, seqs)
            for s in seqs:
                d = per.setdefault(s, {"frames": n[(k, s)]["frames"], "vm": [], "va": []})
                jf = lambda by, kk: 50 * (np.array(by[(kk, s)]["J"]) + np.array(by[(kk, s)]["F"]))
                d["vm"].append(jf(n, k) - jf(m, S_FINAL)); d["va"].append(jf(n, k) - jf(b, REF))
    out = {"variant": v}
    for tag, pick in (("all", lambda s: True), ("long", lambda s: per[s]["frames"] > 600), ("new90", lambda s: s in new90)):
        ss = [s for s in sorted(per) if pick(s)]
        vm = np.concatenate([np.mean(per[s]["vm"], 0) for s in ss]); va = np.concatenate([np.mean(per[s]["va"], 0) for s in ss])
        out[tag] = {"objects": len(vm), "minus_method": ci(vm), "minus_algm": ci(va)}
    print(json.dumps(out, indent=1))
    (Q / "r40_q_val.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def d1_heldout():
    """Audits 2026-11-23: the development control D1 (`palgm_t0.98_dil1~algm_t0.98`: the previous prediction's map at
    full resolution, plain ALGM 0.98 elsewhere and as the fallback) on the 140 held-out videos, three ViT-S heads
    (`eval_omd1_<head>`), against plain ALGM 0.98 and the method on the masks-saved rows of the same heads
    (`eval_om_<head>_base`, `_r40`); head-averaged per object, also on the 90 new videos. Writes `r40_d1_heldout.json`."""
    D1 = "palgm_t0.98_dil1~algm_t0.98"
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    per = {}
    for h in HEADS:
        d, b, m = load(V / f"eval_omd1_{h}"), load(V / f"eval_om_{h}_base"), load(V / f"eval_om_{h}_r40")
        seqs = sorted(s for (k, s) in d if k == D1)
        assert len(seqs) == 140, (h, len(seqs))
        _same_objects(d, D1, b, REF, seqs)
        _same_objects(d, D1, m, S_FINAL, seqs)
        for s in seqs:
            jf = lambda by, k: 50 * (np.array(by[(k, s)]["J"]) + np.array(by[(k, s)]["F"]))
            e = per.setdefault(s, {"d_algm": [], "d_method": [], "m_algm": [], "fl": [], "flm": [], "fla": [], "fr": d[(D1, s)]["frames"]})
            e["d_algm"].append(jf(d, D1) - jf(b, REF)); e["d_method"].append(jf(d, D1) - jf(m, S_FINAL))
            e["m_algm"].append(jf(m, S_FINAL) - jf(b, REF))
            e["fl"].append(d[(D1, s)]["flops_ratio"]); e["flm"].append(m[(S_FINAL, s)]["flops_ratio"]); e["fla"].append(b[(REF, s)]["flops_ratio"])
    out = {}
    for tag, pick in (("all", lambda s: True), ("new90", lambda s: s in new90), ("long", lambda s: per[s]["fr"] > 600)):
        ss = [s for s in sorted(per) if pick(s)]
        cat = lambda x: np.concatenate([np.mean(per[s][x], 0) for s in ss])
        w = np.array([per[s]["fr"] for s in ss], float)
        fw = lambda x: float(np.sum([np.mean(per[s][x]) * per[s]["fr"] for s in ss]) / w.sum())
        out[tag] = {"objects": len(cat("d_algm")), "d1_minus_algm": ci(cat("d_algm")), "d1_minus_method": ci(cat("d_method")),
                    "method_minus_algm": ci(cat("m_algm")), "d1_flops": fw("fl"), "method_flops": fw("flm"), "algm_flops": fw("fla")}
    print(json.dumps(out, indent=1))
    (Q / "r40_d1_heldout.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def iframe_mem_gate():
    """Plan section R (development): memory handling on the dense I-frames of the base schedule B (section Q), each
    variant against B's rows and the method's rows on section Q's videos. Writes `r40_r_gate.json` with `final_choice`
    (R's chosen variant, else Q's chosen variant, else None) for the held-out step."""
    qg = json.loads((Q / "r40_q_gate.json").read_text())
    dense = [v for v in ("key10", "key20", "key30", "lostkey") if v in qg]
    base = qg["chosen"] if qg["chosen"] in dense else max(dense, key=lambda v: qg[v]["long"]["minus_method"][0])
    long_, samp = ((Q / f).read_text().split() for f in ("r40_q_long.txt", "r40_q_sample.txt"))
    kb = f"{S_FINAL}@{base}"
    bq, ref, algm = load(V / f"eval_r40q_{base}"), load(V / "eval_r40_gate2"), load(V / "diag_dg1")
    out, cands = {"base": base}, []
    for mv in ("iwrite", "irefresh", "iperm"):
        k = f"{kb}@{mv}"
        if not (V / f"eval_r40r_{mv}").exists():
            continue
        n = load(V / f"eval_r40r_{mv}")
        res = {}
        for tag, seqs in (("long", sorted(long_)), ("sample", sorted(samp))):
            _same_objects(n, k, bq, kb, seqs)
            _same_objects(n, k, ref, S_FINAL, seqs)
            jn, fn_, _ = tools(n, seqs)
            jb, _, _ = tools(bq, seqs)
            jr, _, _ = tools(ref, seqs)
            res[tag] = {"objects": len(jn(k)), "minus_base": ci(jn(k) - jb(kb)), "minus_method": ci(jn(k) - jr(S_FINAL)), "flops": fn_(k)}
        allv = sorted(long_ + samp)
        _, fn_, _ = tools(n, allv)
        _, fa, _ = tools(algm, allv)
        res["flops_all"], res["algm_flops_all"] = fn_(k), fa(REF)
        res["mem_events"] = sum(n[(k, s)].get("mem_events", 0) for s in allv)
        res["candidate"] = bool(res["long"]["minus_base"][0] >= 1.0 and res["sample"]["minus_method"][1] > -1.0
                                and res["flops_all"] < res["algm_flops_all"])
        out[mv] = res
        if res["candidate"]:
            cands.append((-res["long"]["minus_base"][0], res["flops_all"], mv))
    out["chosen"] = f"{base}@{sorted(cands)[0][2]}" if cands else None
    out["final_choice"] = out["chosen"] or qg["chosen"]
    print(json.dumps(out, indent=1))
    print(f"R GATE: {out['chosen'] or 'none'}; held-out: {out['final_choice'] or 'none'}")
    (Q / "r40_r_gate.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def surprise_gate():
    """Plan section T (development): surprise regions at full resolution (`-fh`), escalation on two consecutive frames
    (`-esc2`), hold for 3 or 5 frames (`-hold3`, `-hold5`) and `-esc2-hold3`, each against the method's rows on section
    Q's videos. Writes `r40_t_gate.json` with `chosen`."""
    long_, samp = ((Q / f).read_text().split() for f in ("r40_q_long.txt", "r40_q_sample.txt"))
    fast = set((Q / "r40_gate_subset.txt").read_text().split())
    ref, algm = load(V / "eval_r40_gate2"), load(V / "diag_dg1")
    out, cands = {}, []
    for v in ("fh", "esc2", "hold3", "hold5", "esc2-hold3"):
        d = V / f"eval_r40t_{v}"
        if not d.exists():
            continue
        k = f"{S_FINAL}-{v}"
        n = load(d)
        res = {}
        for tag, seqs in (("long", sorted(long_)), ("sample", sorted(samp)), ("fast", sorted(fast & set(long_ + samp)))):
            _same_objects(n, k, ref, S_FINAL, seqs)
            jn, fn_, _ = tools(n, seqs)
            jr, fr, _ = tools(ref, seqs)
            res[tag] = {"objects": len(jn(k)), "minus_method": ci(jn(k) - jr(S_FINAL)), "flops": fn_(k), "method_flops": fr(S_FINAL)}
        allv = sorted(long_ + samp)
        _, fn_, _ = tools(n, allv)
        _, fa, _ = tools(algm, allv)
        res["flops_all"], res["algm_flops_all"] = fn_(k), fa(REF)
        res["candidate"] = bool(res["long"]["minus_method"][0] >= 1.0 and res["sample"]["minus_method"][1] > -1.0
                                and res["flops_all"] < res["algm_flops_all"])
        out[v] = res
        if res["candidate"]:
            cands.append((-res["long"]["minus_method"][0], res["flops_all"], v))
    out["chosen"] = sorted(cands)[0][2] if cands else None
    print(json.dumps(out, indent=1))
    print(f"T GATE: {out['chosen'] or 'none'}")
    (Q / "r40_t_gate.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


RELAX = {"S1": ("titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98", E_FINAL.replace("-ring", "")),
         "S2": ("titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98", E_FINAL.replace("-srch", "")),
         "S3": ("titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98", E_FINAL.replace("-e0.06", "-e100")),
         "S4": ("titome_t0.95_l0.7_floor64_dil1_cls64~algm_t0.95", E_FINAL),
         "S5": ("titome_t0.98_l0.7_floor32_dil1_cls64~algm_t0.98", E_FINAL)}


def relax_gate():
    """Plan section S, stage 1 (development): each relaxation of the checks (`eval_r40s_<S>` without a schedule,
    `eval_r40s_<S>_fbkey20` with a fallback frame every 20 frames) against the unrelaxed method's rows
    (`eval_r40_gate2`) and the unrelaxed `@fbkey20` rows (`eval_r40q_fbkey20`). Candidate: with the schedule, above
    -0.5 J&F against the method on the fast stratum and on the long videos, lower bound above -1.0 on the sample, FLOPs
    at least 3% below the method's on all videos, and the schedule's run minus the control at least +0.5 on the fast
    stratum or the long videos (attribution). Writes `r40_s_gate.json`."""
    long_, samp = ((Q / f).read_text().split() for f in ("r40_q_long.txt", "r40_q_sample.txt"))
    fast = sorted(set((Q / "r40_gate_subset.txt").read_text().split()) & set(long_ + samp))
    ref, unrel = load(V / "eval_r40_gate2"), load(V / "eval_r40q_fbkey20")
    strata = (("long", sorted(long_)), ("sample", sorted(samp)), ("fast", fast), ("all", sorted(long_ + samp)))
    out = {}
    for r, (m, e) in RELAX.items():
        arms = {}
        for arm, suffix in (("control", ""), ("fbkey20", "@fbkey20")):
            d = V / f"eval_r40s_{r}{'_fbkey20' if suffix else ''}"
            if not d.exists() or len(list(d.glob("sequences/*.json"))) < 104:
                continue
            k, n = f"{m}@{e}{suffix}", load(d)
            res = {}
            for tag, seqs in strata:
                _same_objects(n, k, ref, S_FINAL, seqs)
                jn, fn_, _ = tools(n, seqs)
                jr, fr, _ = tools(ref, seqs)
                res[tag] = {"objects": len(jn(k)), "minus_method": ci(jn(k) - jr(S_FINAL)), "flops": fn_(k), "method_flops": fr(S_FINAL)}
                if suffix:
                    ju, _, _ = tools(unrel, seqs)
                    res[tag]["minus_unrelaxed_fbkey20"] = ci(jn(k) - ju(f"{S_FINAL}@fbkey20"))
            arms[arm] = res
        if "fbkey20" in arms:
            a, c = arms["fbkey20"], arms.get("control")
            att = {t: ci_diff(load(V / f"eval_r40s_{r}_fbkey20"), f"{m}@{e}@fbkey20", load(V / f"eval_r40s_{r}"), f"{m}@{e}", dict(strata)[t]) for t in ("long", "fast")} if c else None
            arms["attribution"] = att
            arms["candidate"] = bool(
                a["fast"]["minus_method"][0] > -0.5 and a["long"]["minus_method"][0] > -0.5
                and a["sample"]["minus_method"][1] > -1.0 and a["all"]["flops"] <= 0.97 * a["all"]["method_flops"]
                and att is not None and max(att["long"][0], att["fast"][0]) >= 0.5)
        out[r] = arms
    out["candidates"] = [r for r in RELAX if out[r].get("candidate")]
    print(json.dumps(out, indent=1))
    print("S GATE: " + (", ".join(out["candidates"]) or "none"))
    (Q / "r40_s_gate.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def ci_diff(a, ka, b, kb, seqs):
    """Paired per-object difference of two row sets with different settings (same objects), bootstrap interval."""
    _same_objects(a, ka, b, kb, seqs)
    ja, _, _ = tools(a, seqs)
    jb, _, _ = tools(b, seqs)
    return ci(ja(ka) - jb(kb))


def thr_gate():
    """Plan section U (development): the surprise threshold 0.12, 0.25, 0.5 (`eval_r40u_<thr>`) and off (`eval_r40s_S3`,
    the control arm of section S) against the method's rows. Writes `r40_u_gate.json` with `chosen`."""
    long_, samp = ((Q / f).read_text().split() for f in ("r40_q_long.txt", "r40_q_sample.txt"))
    fast = sorted(set((Q / "r40_gate_subset.txt").read_text().split()) & set(long_ + samp))
    ref = load(V / "eval_r40_gate2")
    M = "titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98"
    arms = {"e0.12": ("eval_r40u_e0.12", E_FINAL.replace("-e0.06", "-e0.12")), "e0.25": ("eval_r40u_e0.25", E_FINAL.replace("-e0.06", "-e0.25")),
            "e0.5": ("eval_r40u_e0.5", E_FINAL.replace("-e0.06", "-e0.5")), "off": ("eval_r40s_S3", E_FINAL.replace("-e0.06", "-e100"))}
    out, cands = {}, []
    for tag, (d, e) in arms.items():
        if not (V / d).exists() or len(list((V / d).glob("sequences/*.json"))) < 104:
            continue
        k, n = f"{M}@{e}", load(V / d)
        res = {}
        for st, seqs in (("all", sorted(long_ + samp)), ("long", sorted(long_)), ("sample", sorted(samp)), ("fast", fast)):
            _same_objects(n, k, ref, S_FINAL, seqs)
            jn, fn_, _ = tools(n, seqs)
            jr, fr, _ = tools(ref, seqs)
            res[st] = {"objects": len(jn(k)), "minus_method": ci(jn(k) - jr(S_FINAL)), "flops": fn_(k), "method_flops": fr(S_FINAL)}
        res["fallback_frames"] = sum(n[(k, s)]["fallback_frames"] for s in long_ + samp)
        res["candidate"] = bool(res["all"]["minus_method"][1] > -1.0 and res["long"]["minus_method"][0] > -0.5
                                and res["all"]["flops"] <= 0.9 * res["all"]["method_flops"])
        out[tag] = res
        if res["candidate"]:
            cands.append((res["all"]["flops"], tag))
    out["chosen"] = sorted(cands)[0][1] if cands else None
    out["held_out_choice"] = out["chosen"]
    thr = {"e0.12": "-e0.12", "e0.25": "-e0.25", "e0.5": "-e0.5", "off": "-e100"}.get(out["chosen"])
    dc = V / "eval_r40u_combo"
    if thr and dc.exists() and len(list(dc.glob("sequences/*.json"))) >= 104:      # the combination: no ring + that threshold
        k, n = f"{M}@{E_FINAL.replace('-ring', '').replace('-e0.06', thr)}", load(dc)
        res = {}
        for st, seqs in (("all", sorted(long_ + samp)), ("long", sorted(long_)), ("sample", sorted(samp)), ("fast", fast)):
            _same_objects(n, k, ref, S_FINAL, seqs)
            jn, fn_, _ = tools(n, seqs)
            jr, fr, _ = tools(ref, seqs)
            res[st] = {"objects": len(jn(k)), "minus_method": ci(jn(k) - jr(S_FINAL)), "flops": fn_(k), "method_flops": fr(S_FINAL)}
        res["candidate"] = bool(res["all"]["minus_method"][1] > -1.0 and res["long"]["minus_method"][0] > -0.5
                                and res["all"]["flops"] <= 0.9 * res["all"]["method_flops"])
        out["combo"] = res
        if res["candidate"] and res["all"]["flops"] < out[out["chosen"]]["all"]["flops"]:
            out["held_out_choice"] = "combo"
    print(json.dumps(out, indent=1))
    print(f"U GATE: {out['chosen'] or 'none'}; held-out: {out['held_out_choice'] or 'none'}")
    (Q / "r40_u_gate.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def _fw(per, ss, key):
    """FLOPs ratio of `key` (head-averaged per video), weighted by the videos' frames (Section 3.7: frame-weighted across videos)."""
    return float(sum(np.mean(per[s][key]) * per[s]["frames"] for s in ss) / sum(per[s]["frames"] for s in ss))


def iperm_controls():
    """Plan section V (held-out controls for section R's finding): the method with `key20@iperm` against plain ALGM 0.98
    and dense Cutie given the same schedule (`dense@key20@iperm`, `algm_t0.98@key20@iperm`; every 20th frame is written to
    Cutie's permanent memory), head-averaged per object, with each setting's own gain from the schedule. All objects, objects
    in videos longer than 600 frames, the 90 new videos. Writes `r40_v_controls.json`."""
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    kv, kd, ka = f"{S_FINAL}@key20@iperm", "dense@key20@iperm", f"{REF}@key20@iperm"
    per = {}
    for h, (r1, r2) in HEADS.items():
        for split, bdir, want in (("v1val", r1, None), ("v2new", r2, new90)):
            n, m, b = load(V / f"eval_r40q_{h}_{split}"), load(V / f"eval_r40_{h}_{split}"), load(V / bdir)
            cd_, ca_ = load(V / f"eval_r40v_dense_{h}_{split}"), load(V / f"eval_r40v_algm_{h}_{split}")
            seqs = sorted(s for (kk, s) in n if kk == kv and (want is None or s in want))
            for s in seqs:
                assert n[(kv, s)]["objects"] == cd_[(kd, s)]["objects"] == ca_[(ka, s)]["objects"], (h, s)
                jf = lambda by, kk: 50 * (np.array(by[(kk, s)]["J"]) + np.array(by[(kk, s)]["F"]))
                d = per.setdefault(s, {"frames": n[(kv, s)]["frames"], "v": [], "m": [], "a": [], "d": [], "va": [], "vd": [], "fv": [], "fva": [], "fm": []})
                d["v"].append(jf(n, kv)); d["m"].append(jf(m, S_FINAL)); d["a"].append(jf(b, REF)); d["d"].append(jf(b, "dense"))
                d["va"].append(jf(ca_, ka)); d["vd"].append(jf(cd_, kd))
                d["fv"].append(n[(kv, s)]["flops_ratio"]); d["fva"].append(ca_[(ka, s)]["flops_ratio"]); d["fm"].append(m[(S_FINAL, s)]["flops_ratio"])
    out = {}
    for tag, pick in (("all", lambda s: True), ("long", lambda s: per[s]["frames"] > 600), ("new90", lambda s: s in new90)):
        ss = [s for s in sorted(per) if pick(s)]
        g = lambda key: np.concatenate([np.mean(per[s][key], 0) for s in ss])
        out[tag] = {"objects": len(g("v")),
                    "jf": {k: float(g(k).mean()) for k in ("v", "m", "a", "d", "va", "vd")},
                    "method_iperm_minus_algm_iperm": ci(g("v") - g("va")), "method_iperm_minus_dense_iperm": ci(g("v") - g("vd")),
                    "gain_from_schedule": {"method": ci(g("v") - g("m")), "algm": ci(g("va") - g("a")), "dense": ci(g("vd") - g("d"))},
                    "flops": {"method_iperm": _fw(per, ss, "fv"),
                              "algm_iperm": _fw(per, ss, "fva"),
                              "method": _fw(per, ss, "fm")}}
    print(json.dumps(out, indent=1))
    (Q / "r40_v_controls.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def thr_heldout():
    """Plan section W (held-out): the surprise threshold `e0.5` and off (`e100`) (`eval_r40w_<tag>_<head>_{v1val,v2new}`), no schedule,
    against the method's reference rows (`eval_r40_<head>_*`) and ALGM 0.98, head-averaged per object; all objects, videos over 600 frames,
    the 90 new videos. Writes `r40_w_val.json`."""
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    M = "titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98"
    out = {}
    for tag, e in (("e0.5", E_FINAL.replace("-e0.06", "-e0.5")), ("off", E_FINAL.replace("-e0.06", "-e100"))):
        k, per = f"{M}@{e}", {}
        for h, (r1, r2) in HEADS.items():
            for split, bdir, want in (("v1val", r1, None), ("v2new", r2, new90)):
                n, m, b = load(V / f"eval_r40w_{tag}_{h}_{split}"), load(V / f"eval_r40_{h}_{split}"), load(V / bdir)
                seqs = sorted(s for (kk, s) in n if kk == k and (want is None or s in want))
                _same_objects(n, k, m, S_FINAL, seqs)
                _same_objects(n, k, b, REF, seqs)
                for s in seqs:
                    jf = lambda by, kk: 50 * (np.array(by[(kk, s)]["J"]) + np.array(by[(kk, s)]["F"]))
                    d = per.setdefault(s, {"frames": n[(k, s)]["frames"], "vm": [], "va": [], "fv": [], "fm": [], "fa": []})
                    d["vm"].append(jf(n, k) - jf(m, S_FINAL)); d["va"].append(jf(n, k) - jf(b, REF))
                    d["fv"].append(n[(k, s)]["flops_ratio"]); d["fm"].append(m[(S_FINAL, s)]["flops_ratio"]); d["fa"].append(b[(REF, s)]["flops_ratio"])
        res = {}
        for st, pick in (("all", lambda s: True), ("long", lambda s: per[s]["frames"] > 600), ("new90", lambda s: s in new90)):
            ss = [s for s in sorted(per) if pick(s)]
            vm = np.concatenate([np.mean(per[s]["vm"], 0) for s in ss]); va = np.concatenate([np.mean(per[s]["va"], 0) for s in ss])
            res[st] = {"objects": len(vm), "minus_method": ci(vm), "minus_algm": ci(va),
                       "flops": _fw(per, ss, "fv"), "method_flops": _fw(per, ss, "fm"),
                       "algm_flops": _fw(per, ss, "fa")}
        a = res["all"]
        res["confirmed"] = bool(a["minus_method"][1] > -1.0 and a["flops"] <= 0.9 * a["method_flops"] and a["minus_algm"][1] > 0)
        out[tag] = res
    print(json.dumps(out, indent=1))
    (Q / "r40_w_val.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def rest_replication():
    """Plan section X: the permanent-memory schedule (`key20@iperm`) on the 316 development videos not used to choose it
    (`eval_r40x_<method|dense|algm>`) against the existing rows of the same videos (`eval_r40_gate2`: the method; `diag_dg1`: dense and
    ALGM 0.98). Per object: each setting's gain from the schedule, and at equal schedule method minus ALGM and method minus dense; all
    objects, videos over 600 frames, the others. Writes `r40_x_rest.json`."""
    rest = sorted((Q / "r40_x_rest.txt").read_text().split())
    rm, rb = load(V / "eval_r40_gate2"), load(V / "diag_dg1")
    arms = {"method": (load(V / "eval_r40x_method"), f"{S_FINAL}@key20@iperm", rm, S_FINAL),
            "dense": (load(V / "eval_r40x_dense"), "dense@key20@iperm", rb, "dense"),
            "algm": (load(V / "eval_r40x_algm"), f"{REF}@key20@iperm", rb, REF)}
    jf = lambda by, k, s: 50 * (np.array(by[(k, s)]["J"]) + np.array(by[(k, s)]["F"]))
    frames = {s: rm[(S_FINAL, s)]["frames"] for s in rest}
    out = {}
    for tag, pick in (("all", lambda s: True), ("over300", lambda s: frames[s] > 300), ("under300", lambda s: frames[s] <= 300),
                      ("over600", lambda s: frames[s] > 600)):
        ss = [s for s in rest if pick(s)]
        if not ss:                                                  # the 44 videos over 600 frames are all in the development 104
            continue
        cat = lambda by, k: np.concatenate([jf(by, k, s) for s in ss])
        res = {"videos": len(ss), "objects": int(sum(len(rm[(S_FINAL, s)]["J"]) for s in ss))}
        sched, base = {}, {}
        for a, (n, k, b, kb) in arms.items():
            for s in ss:
                assert n[(k, s)]["objects"] == b[(kb, s)]["objects"], (a, s)
            sched[a], base[a] = cat(n, k), cat(b, kb)
            res[a] = {"default": float(base[a].mean()), "schedule": float(sched[a].mean()), "gain": ci(sched[a] - base[a]),
                      "flops_schedule": float(sum(n[(k, s)]["flops_ratio"] * frames[s] for s in ss) / sum(frames[s] for s in ss)),
                      "flops_default": float(sum(b[(kb, s)]["flops_ratio"] * frames[s] for s in ss) / sum(frames[s] for s in ss))}
        res["method_minus_algm_at_schedule"] = ci(sched["method"] - sched["algm"])
        res["method_minus_dense_at_schedule"] = ci(sched["method"] - sched["dense"])
        out[tag] = res
    out["replicates"] = bool(out["all"]["dense"]["gain"][1] > 0 and out["all"]["method"]["gain"][1] > 0)
    print(json.dumps(out, indent=1))
    print("X REPLICATES:", out["replicates"])
    (Q / "r40_x_rest.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def combo_gate():
    """Plan section Y (development): combinations of the late findings against the method's rows (`eval_r40_gate2`), each with its own
    candidate rule (see the plan). Writes `r40_y_gate.json` with `chosen`."""
    long_, samp = ((Q / f).read_text().split() for f in ("r40_q_long.txt", "r40_q_sample.txt"))
    fast = sorted(set((Q / "r40_gate_subset.txt").read_text().split()) & set(long_ + samp))
    ref = load(V / "eval_r40_gate2")
    M = "titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98"
    E5 = E_FINAL.replace("-e0.06", "-e0.5")
    arms = {"Y1": f"{M}@{E_FINAL}@perm20", "Y5": f"{M}@{E_FINAL}@perm40", "Y2": f"{M}@{E5}@perm20", "Y4": f"{M}@{E5}@key20@iperm", "Y3": f"{M}@{E5}-hold3"}
    out, cands = {}, []
    for tag, k in arms.items():
        d = V / f"eval_r40y_{tag}"
        if not d.exists() or len(list(d.glob("sequences/*.json"))) < 104:
            continue
        n = load(d)
        res = {}
        for st, seqs in (("all", sorted(long_ + samp)), ("long", sorted(long_)), ("sample", sorted(samp)), ("fast", fast)):
            _same_objects(n, k, ref, S_FINAL, seqs)
            jn, fn_, _ = tools(n, seqs)
            jr, fr, _ = tools(ref, seqs)
            res[st] = {"objects": len(jn(k)), "minus_method": ci(jn(k) - jr(S_FINAL)), "flops": fn_(k), "method_flops": fr(S_FINAL)}
        res["perm_writes"] = int(sum(n[(k, s)].get("mem_events", 0) for s in long_ + samp))
        fl = res["all"]["flops"] / res["all"]["method_flops"]
        if tag in ("Y1", "Y5"):
            ok = res["long"]["minus_method"][0] >= 1.0 and res["sample"]["minus_method"][1] > -1.0 and fl <= 1.02
        elif tag in ("Y2", "Y4"):
            ok = res["long"]["minus_method"][0] >= 1.0 and res["sample"]["minus_method"][1] > -1.0 and fl <= 0.9
        else:
            ok = res["fast"]["minus_method"][0] >= 1.0 and res["all"]["minus_method"][1] > -1.0 and fl <= 1.0
        res["flops_vs_method"], res["candidate"] = fl, bool(ok)
        out[tag] = res
        if ok:
            cands.append((-res["long"]["minus_method"][0], res["all"]["flops"], tag))
    out["chosen"] = sorted(cands)[0][2] if cands else None
    print(json.dumps(out, indent=1))
    print(f"Y GATE: {out['chosen'] or 'none'}")
    (Q / "r40_y_gate.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def length_gain():
    """Held-out gain by video length (three ViT-S heads, per object, head-averaged): the method against ALGM 0.98 and against dense without
    a schedule, and with the permanent-memory schedule `key20@iperm` on every setting (method, ALGM 0.98, dense): the schedule's gain, the
    method minus ALGM and minus dense at equal schedule. Bins: at most 300, 301 to 600, over 600 frames. Writes `r40_length_gain.json`."""
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    kv, kd, ka = f"{S_FINAL}@key20@iperm", "dense@key20@iperm", f"{REF}@key20@iperm"
    per = {}
    for h, (r1, r2) in HEADS.items():
        for split, bdir, want in (("v1val", r1, None), ("v2new", r2, new90)):
            n, m, b = load(V / f"eval_r40q_{h}_{split}"), load(V / f"eval_r40_{h}_{split}"), load(V / bdir)
            cd_, ca_ = load(V / f"eval_r40v_dense_{h}_{split}"), load(V / f"eval_r40v_algm_{h}_{split}")
            for s in sorted(s for (kk, s) in n if kk == kv and (want is None or s in want)):
                jf = lambda by, kk: 50 * (np.array(by[(kk, s)]["J"]) + np.array(by[(kk, s)]["F"]))
                d = per.setdefault(s, {"frames": n[(kv, s)]["frames"], **{k: [] for k in ("v", "m", "a", "d", "va", "vd")}})
                d["v"].append(jf(n, kv)); d["m"].append(jf(m, S_FINAL)); d["a"].append(jf(b, REF)); d["d"].append(jf(b, "dense"))
                d["va"].append(jf(ca_, ka)); d["vd"].append(jf(cd_, kd))
    out = {}
    for tag, pick in (("<=300", lambda f: f <= 300), ("301-600", lambda f: 300 < f <= 600), (">600", lambda f: f > 600)):
        ss = [s for s in sorted(per) if pick(per[s]["frames"])]
        g = lambda key: np.concatenate([np.mean(per[s][key], 0) for s in ss])
        out[tag] = {"videos": len(ss), "objects": len(g("v")),
                    "method_minus_algm": ci(g("m") - g("a")), "method_minus_dense": ci(g("m") - g("d")),
                    "schedule_gain": {"method": ci(g("v") - g("m")), "algm": ci(g("va") - g("a")), "dense": ci(g("vd") - g("d"))},
                    "method_minus_algm_at_schedule": ci(g("v") - g("va")), "method_minus_dense_at_schedule": ci(g("v") - g("vd"))}
    print(json.dumps(out, indent=1))
    (Q / "r40_length_gain.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def oracle_headroom():
    """Map-quality headroom (development, 104 videos, R26 head): the method with the expectation's object states and the evaluator's map taken from
    the ground truth of the two previous frames (`@gtobj`, a perfect tracker; a diagnostic, not a method) against the method's rows and dense
    (`diag_dg1`). Writes `r40_oracle_headroom.json`."""
    long_, samp = ((Q / f).read_text().split() for f in ("r40_q_long.txt", "r40_q_sample.txt"))
    fast = sorted(set((Q / "r40_gate_subset.txt").read_text().split()) & set(long_ + samp))
    ref, dns, n = load(V / "eval_r40_gate2"), load(V / "diag_dg1"), load(V / "eval_r40y_gtobj")
    k = f"{S_FINAL}@gtobj"
    out = {}
    for st, seqs in (("all", sorted(long_ + samp)), ("long", sorted(long_)), ("sample", sorted(samp)), ("fast", fast)):
        _same_objects(n, k, ref, S_FINAL, seqs)
        _same_objects(n, k, dns, "dense", seqs)
        jn, fn_, _ = tools(n, seqs)
        jr, fr, _ = tools(ref, seqs)
        jd, _, _ = tools(dns, seqs)
        out[st] = {"objects": len(jn(k)), "jf": float(jn(k).mean()), "method_jf": float(jr(S_FINAL).mean()), "dense_jf": float(jd("dense").mean()),
                   "minus_method": ci(jn(k) - jr(S_FINAL)), "minus_dense": ci(jn(k) - jd("dense")), "flops": fn_(k), "method_flops": fr(S_FINAL)}
    print(json.dumps(out, indent=1))
    (Q / "r40_oracle_headroom.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def y4_heldout():
    """Plan section Y (held-out step for the gate's choice Y4 = method with `e0.5`, `key20@iperm`; `eval_r40y_y4_<head>_{v1val,v2new}`): head-averaged
    per object against the method with the same schedule (`eval_r40q_*`, section V), the plain method, ALGM 0.98 and dense Cutie with the same
    schedule (`eval_r40v_*`) and dense Cutie (reference rows); all objects, videos over 600 frames, the 90 new videos. Rule fixed before the
    analysis: confirmed (as an operating point, not a change of method) if the lower bound of Y4 minus the method with the schedule is above -1.0 J&F
    on all objects and its FLOPs are at least 10% below that method's. Writes `r40_y4_heldout.json`."""
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    E5 = E_FINAL.replace("-e0.06", "-e0.5")
    k4, kv, kd, ka = f"{S_FINAL.split('@')[0]}@{E5}@key20@iperm", f"{S_FINAL}@key20@iperm", "dense@key20@iperm", f"{REF}@key20@iperm"
    per = {}
    for h, (r1, r2) in HEADS.items():
        for split, bdir, want in (("v1val", r1, None), ("v2new", r2, new90)):
            y, q, m, b = (load(V / f"eval_r40y_y4_{h}_{split}"), load(V / f"eval_r40q_{h}_{split}"), load(V / f"eval_r40_{h}_{split}"), load(V / bdir))
            cd_, ca_ = load(V / f"eval_r40v_dense_{h}_{split}"), load(V / f"eval_r40v_algm_{h}_{split}")
            seqs = sorted(s for (kk, s) in y if kk == k4 and (want is None or s in want))
            for s in seqs:
                assert y[(k4, s)]["objects"] == q[(kv, s)]["objects"], (h, s)
                jf = lambda by, kk: 50 * (np.array(by[(kk, s)]["J"]) + np.array(by[(kk, s)]["F"]))
                d = per.setdefault(s, {"frames": y[(k4, s)]["frames"], "y": [], "q": [], "m": [], "a": [], "d": [], "va": [], "vd": [], "fy": [], "fq": [], "fm": []})
                d["y"].append(jf(y, k4)); d["q"].append(jf(q, kv)); d["m"].append(jf(m, S_FINAL)); d["a"].append(jf(b, REF)); d["d"].append(jf(b, "dense"))
                d["va"].append(jf(ca_, ka)); d["vd"].append(jf(cd_, kd))
                d["fy"].append(y[(k4, s)]["flops_ratio"]); d["fq"].append(q[(kv, s)]["flops_ratio"]); d["fm"].append(m[(S_FINAL, s)]["flops_ratio"])
    out = {}
    for tag, pick in (("all", lambda s: True), ("long", lambda s: per[s]["frames"] > 600), ("new90", lambda s: s in new90)):
        ss = [s for s in sorted(per) if pick(s)]
        g = lambda key: np.concatenate([np.mean(per[s][key], 0) for s in ss])
        out[tag] = {"objects": len(g("y")), "jf": {k: float(g(k).mean()) for k in ("y", "q", "m", "a", "d", "va", "vd")},
                    "minus_method_schedule": ci(g("y") - g("q")), "minus_method": ci(g("y") - g("m")), "minus_dense": ci(g("y") - g("d")),
                    "minus_algm_schedule": ci(g("y") - g("va")), "minus_dense_schedule": ci(g("y") - g("vd")),
                    "flops": {"y4": _fw(per, ss, "fy"), "method_schedule": _fw(per, ss, "fq"),
                              "method": _fw(per, ss, "fm")}}
    a = out["all"]
    out["confirmed"] = bool(a["minus_method_schedule"][1] > -1.0 and a["flops"]["y4"] <= 0.9 * a["flops"]["method_schedule"])
    print(json.dumps(out, indent=1))
    (Q / "r40_y4_heldout.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def davis_more():
    """Plan section Z1 (reported): G2TM and DTEM inside Cutie on DAVIS 2017 val, the paper's head and fast protocol (`eval_r40_davis_more`, its own dense
    rows), each setting's J&F, FLOPs, change from dense, and the method's rows from `eval_r40_davis_all` on the same objects (method minus setting,
    paired over objects). Writes `r40_davis_more.json`."""
    b, a = load(V / "eval_r40_davis_more"), load(V / "eval_r40_davis_all")
    seqs = sorted(s for (k, s) in b if k == "dense")
    ks = sorted({k for (k, s) in b} - {"dense"})
    assert all((k, s) in b for k in ks for s in seqs), "incomplete"
    jf, fl, _ = tools(b, seqs)
    ja, fa, _ = tools(a, seqs)
    assert len(ja(S_FINAL)) == len(jf("dense")), "objects"
    res = {"videos": len(seqs), "objects": len(jf("dense")), "dense": float(jf("dense").mean()), "method_jf": float(ja(S_FINAL).mean()),
           "method_flops": fa(S_FINAL), "settings": {}}
    for k in ks:
        res["settings"][k] = {"jf": float(jf(k).mean()), "flops": fl(k), "minus_dense": ci(jf(k) - jf("dense")), "method_minus": ci(ja(S_FINAL) - jf(k))}
    print(json.dumps(res, indent=1))
    (Q / "r40_davis_more.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def extras_heldout():
    """Held-out comparison rows added for the paper (three ViT-S heads, 140 videos, reference path, no schedule): G2TM (`g2tm_t0.94_at2`,
    `g2tm_t0.98_at2`) and ALGM at lower thresholds (`algm_t0.95`, `algm_t0.96`; the FLOPs of the method), `eval_r40z_<tag>_<head>_{v1val,v2new}`,
    against dense, plain ALGM 0.98 and the method's reference rows, head-averaged per object; all objects, videos over 600 frames, the 90 new
    videos. Writes `r40_extras_heldout.json`."""
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    out = {}
    for tag, k in (("g2tm94", "g2tm_t0.94_at2"), ("g2tm98", "g2tm_t0.98_at2"), ("algm95", "algm_t0.95"), ("algm96", "algm_t0.96")):
        if not all((V / f"eval_r40z_{tag}_{h}_v2new").exists() for h in HEADS):
            continue
        per = {}
        for h, (r1, r2) in HEADS.items():
            for split, bdir, want in (("v1val", r1, None), ("v2new", r2, new90)):
                n, m, b = load(V / f"eval_r40z_{tag}_{h}_{split}"), load(V / f"eval_r40_{h}_{split}"), load(V / bdir)
                seqs = sorted(s for (kk, s) in n if kk == k and (want is None or s in want))
                for s in seqs:
                    jf = lambda by, kk: 50 * (np.array(by[(kk, s)]["J"]) + np.array(by[(kk, s)]["F"]))
                    d = per.setdefault(s, {"frames": n[(k, s)]["frames"], "x": [], "m": [], "a": [], "d": [], "fx": [], "fm": []})
                    d["x"].append(jf(n, k)); d["m"].append(jf(m, S_FINAL)); d["a"].append(jf(b, REF)); d["d"].append(jf(b, "dense"))
                    d["fx"].append(n[(k, s)]["flops_ratio"]); d["fm"].append(m[(S_FINAL, s)]["flops_ratio"])
        res = {}
        for st, pick in (("all", lambda s: True), ("long", lambda s: per[s]["frames"] > 600), ("new90", lambda s: s in new90)):
            ss = [s for s in sorted(per) if pick(s)]
            g = lambda key: np.concatenate([np.mean(per[s][key], 0) for s in ss])
            res[st] = {"objects": len(g("x")), "jf": float(g("x").mean()), "minus_dense": ci(g("x") - g("d")), "minus_algm98": ci(g("x") - g("a")),
                       "minus_method": ci(g("x") - g("m")), "flops": _fw(per, ss, "fx"),
                       "method_flops": _fw(per, ss, "fm")}
        out[tag] = res
    print(json.dumps(out, indent=1))
    (Q / "r40_extras_heldout.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def tome_detail():
    return val_detail("eval_r40e", TOME_HEADS, TOME, "r40_tome_val_detail.json")


def final(run="eval_r40l", out_tag="final"):
    """Plan section L4 (owner decision after K3): the validation protocol on the final timed implementation
    (`eval_r40l_<head>_*`, frame runner, VOS_GRAPH=2): section C's rules against ALGM 0.98 (`r40_final_val.json`), the
    detail (`r40_final_val_detail.json`), and per head the final rows against the eager rows of the same setting
    (`eval_r40_<head>_*`), a second draw of the same method (reported, R-37). `run`, `out_tag`: another run of the
    implementation (R-38: `eval_r40m`, `final_emu`)."""
    out = val(run, out_json=f"r40_{out_tag}_val.json")
    val_detail(run, out_json=f"r40_{out_tag}_val_detail.json")
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    draws = {}
    for h in HEADS:
        d = []
        for split, want in (("v1val", None), ("v2new", new90)):
            a, b = load(V / f"{run}_{h}_{split}"), load(V / f"eval_r40_{h}_{split}")
            seqs = sorted(s for (k, s) in a if k == S_FINAL and (want is None or s in want))
            _same_objects(a, S_FINAL, b, S_FINAL, seqs)
            ja, _, _ = tools(a, seqs)
            jb, _, _ = tools(b, seqs)
            d.append(ja(S_FINAL) - jb(S_FINAL))
        d = np.concatenate(d)
        draws[h] = {"final_minus_eager": ci(d), "mean_abs": float(np.abs(d).mean())}
    (Q / f"r40_{out_tag}_draws.json").write_text(json.dumps(draws, indent=1) + "\n")
    print(json.dumps(draws, indent=1))
    return out


def final_short(run="eval_r40l", out_tag="final"):
    """Plan section L4, reported: short clips (DAVIS val) and ViT-B long videos on the final implementation. DAVIS: the
    final rows (`eval_r40l_davis`, `eval_r40lb_davis`) against dense and the eager expectation focus map rows of J1. ViT-B long
    videos (`eval_r40lb_v1val`, `_v2new`, R25 head) against plain ALGM 0.95 and dense from J2's runs (`eval_r40b_*`)."""
    out = {}
    for tag, rd, ref, new in (("vits", f"{run}_davis", "eval_r40_davis_all", S_FINAL),
                              ("vitb", f"{run}b_davis", "eval_r40b_davis", SB_FINAL)):
        if not (V / rd).exists():
            continue
        a, b = load(V / rd), load(V / ref)
        seqs = sorted(s for (k, s) in a if k == new)
        jf, fl, _ = tools(a, seqs)
        jr, flr, _ = tools(b, seqs)
        out[tag] = {"videos": len(seqs), "objects": len(jf(new)), "jf": float(jf(new).mean()), "flops": fl(new),
                    "minus_dense": ci(jf(new) - jr("dense")), "final_minus_eager": ci(jf(new) - jr(new)),
                    "eager_flops": flr(new)}
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    if (V / f"{run}b_v1val").exists() and (V / f"{run}b_v2new").exists():
        parts = []
        for split, want in (("v1val", None), ("v2new", new90)):
            a, b = load(V / f"{run}b_{split}"), load(V / f"eval_r40b_{split}")
            seqs = sorted(s for (k, s) in a if k == SB_FINAL and (want is None or s in want))
            for k in ("algm_t0.95", "dense"):
                _same_objects(a, SB_FINAL, b, k, seqs)
            ja, fa, _ = tools(a, seqs)
            jb, fb_, _ = tools(b, seqs)
            fr = sum(a[(SB_FINAL, s)]["frames"] for s in seqs)
            parts.append((ja(SB_FINAL), jb("algm_t0.95"), jb("dense"), jb(SB_FINAL), fa(SB_FINAL) * fr, fb_("algm_t0.95") * fr, fr))
        cat = lambda i: np.concatenate([p[i] for p in parts])
        frs = sum(p[6] for p in parts)
        out["vitb_long"] = {"objects": len(cat(0)), "jf": float(cat(0).mean()), "minus_algm095": ci(cat(0) - cat(1)),
                            "minus_dense": ci(cat(0) - cat(2)), "final_minus_eager": ci(cat(0) - cat(3)),
                            "flops": sum(p[4] for p in parts) / frs, "algm_flops": sum(p[5] for p in parts) / frs}
    print(json.dumps(out, indent=1))
    (Q / f"r40_{out_tag}_short.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def davis():
    """Short-clip check (R26 head, DAVIS val): each setting against dense; the expectation focus map against ALGM 0.98 and ours
    (paired over objects). Reported; no rule attached."""
    b = load(V / "eval_r40_davis")
    seqs = sorted(s for (k, s) in b if k == "dense")
    jf, fl, _ = tools(b, seqs)
    out = {"videos": len(seqs), "objects": len(jf("dense")), "dense": float(jf("dense").mean())}
    for k in sorted({k for (k, s) in b} - {"dense"}):
        out[k] = {"jf": float(jf(k).mean()), "flops": fl(k), "minus_dense": ci(jf(k) - jf("dense"))}
        if "@exp-" in k:
            out[k] |= {"minus_algm": ci(jf(k) - jf(REF)), "minus_closed_loop": ci(jf(k) - jf(CLOSED_LOOP)),
                       "global_frames": sum(b[(k, s)].get("exp_global", 0) for s in seqs),
                       "frames": sum(b[(k, s)]["frames"] for s in seqs)}
    print(json.dumps(out, indent=1))
    (Q / "r40_davis.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def tome():
    """Plan section E: `ttome_...@exp-...` (eval_r40e_*) against plain ToMe r64 (eval_tome_*), R-25 protocol."""
    return val("eval_r40e", TOME_HEADS, TOME, "r40_tome_val.json")


def tome_fix():
    """Review flag R-41 (audits 2026-11-23): the first and third ToMe versions rerun after the fallback fix (fallback and
    baseline frames now get no map), R-25 protocol against plain ToMe r64 (`eval_r40e2_*`, `eval_r40g2_*`)."""
    return {"v1": val("eval_r40e2", TOME_HEADS, TOME, "r40_tome_fix_val.json"),
            "v3": val("eval_r40g2", TOME_HEADS, TOME, "r40_tome3_fix_val.json")}


def full():
    """Plan section D (owner question): D1 protected ALGM (map at full resolution) against plain ALGM 0.98 (DG1 rows);
    D2 the expectation focus map with the high tier also merged by ALGM's rules against the expectation focus map (gate rows). D2 goes to the owner
    if its lower bound is above −1.0 at FLOPs at least 10% lower."""
    run, ref, gate_run = load(V / "eval_r40_full"), load(V / "diag_dg1"), load(V / "eval_r40_gate")
    S = "titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98@exp-pix-mot-rl-r0-e0.24-ring-srch"
    D1, D2 = "palgm_t0.98_dil1~algm_t0.98", "titome_t0.98_l0.7_floor64_dil1_cls64_ha1" + S[len("titome_t0.98_l0.7_floor64_dil1_cls64"):]
    out = {}
    for tag, k, other, ok_ref, refk in (("D1", D1, ref, REF, REF), ("D2", D2, gate_run, S, S)):
        seqs = sorted(s for (kk, s) in run if kk == k)
        sub = set((Q / "r40_gate_subset.txt").read_text().split())
        _same_objects(run, k, other, refk, seqs)
        jn, fn, _ = tools(run, seqs)
        jr, fr, _ = tools(other, seqs)
        d = jn(k) - jr(refk)
        ss = [s for s in seqs if s in sub]
        jns, fns, _ = tools(run, ss)
        jrs, frs, _ = tools(other, ss)
        out[tag] = {"setting": k, "against": refk, "videos": len(seqs), "objects": len(d), "jf": float(jn(k).mean()),
                    "flops": fn(k), "ref_jf": float(jr(refk).mean()), "ref_flops": fr(refk), "difference": ci(d),
                    "subset": {"videos": len(ss), "difference": ci(jns(k) - jrs(refk)), "flops": fns(k), "ref_flops": frs(refk)}}
    c = out["D2"]
    c["to_owner"] = bool(c["videos"] == 420 and c["difference"][1] > -1.0 and c["flops"] <= 0.9 * c["ref_flops"])
    print(json.dumps(out, indent=1))
    (Q / "r40_full.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def _expect_products(root, seq, width):
    """Counted FLOPs of the expectation products on one frame of a long video (`expect_dev.ExpectDev`: the similarity
    2 N^2 48 and the floor mismatch (2 tol + 1)^2 N 48 8 at tol 1), over dense at the evaluator's patch grid."""
    from PIL import Image
    from src.stage_b.flops import dense_flops
    from .eval_lvos import SIZE
    w, h = Image.open(next(iter(sorted((root / "JPEGImages" / seq).glob("*.jpg"))))).size
    sc = SIZE / min(h, w)
    N = -(-round(h * sc) // 16) * -(-round(w * sc) // 16)
    return (2 * N * N * 48 + 9 * N * 48 * 8) / dense_flops(N, width=width)


def flops_sign(*runs):
    """R-38, reported: per video, the counted FLOPs of an implementation's rows (`final` and `final_short` runs, default
    `eval_r40l eval_r40m eval_r40n eval_r40o` where present) against the eager rows of the same setting: videos above, below and
    equal, videos with equal fallback frames, and the frame-weighted mean difference (`r40_flops_sign.json`). Also, as
    `above_added`, `below_added`: the same with the expectation products added back on the row's fallback frames (the
    frame runner does not compute them there; the eager path computes and counts them; nothing later reads them in the
    final setting)."""
    from .eval_lvos import LVOS, LVOS2
    new90 = set((Q / "v2valid_new.txt").read_text().split())
    roots = {"v1val": LVOS / "valid", "v2new": LVOS2 / "valid"}
    parts = [("davis_vits", "{}_davis", "eval_r40_davis_all", S_FINAL, None, None, 384),
             ("davis_vitb", "{}b_davis", "eval_r40b_davis", SB_FINAL, None, None, 768)]
    parts += [(f"{h}_{sp}", "{}_" + f"{h}_{sp}", f"eval_r40_{h}_{sp}", S_FINAL, w, roots[sp], 384)
              for h in HEADS for sp, w in (("v1val", None), ("v2new", new90))]
    parts += [(f"vitb_{sp}", "{}b_" + sp, f"eval_r40b_{sp}", SB_FINAL, w, roots[sp], 768) for sp, w in (("v1val", None), ("v2new", new90))]
    out, prod = {}, {}
    for run in runs or ("eval_r40l", "eval_r40m", "eval_r40n", "eval_r40o"):
        res = {}
        for tag, rd, ref, k, want, root, width in parts:
            if not (V / rd.format(run)).exists():
                continue
            a, b = load(V / rd.format(run)), load(V / ref)
            seqs = sorted(s for (kk, s) in a if kk == k and (want is None or s in want) and (k, s) in b)
            d = [a[(k, s)]["flops_ratio"] - b[(k, s)]["flops_ratio"] for s in seqs]
            fr = [a[(k, s)]["frames"] for s in seqs]
            add = []
            for s, x in zip(seqs, d):
                nfb = a[(k, s)].get("fallback_frames", 0) if root is not None else 0   # DAVIS: no fallback frames
                if nfb and (root, s, width) not in prod:
                    prod[(root, s, width)] = _expect_products(root, s, width)
                add.append(x + (nfb * prod[(root, s, width)] / a[(k, s)]["frames"] if nfb else 0.0))
            res[tag] = {"videos": len(seqs), "above": sum(x > 1e-6 for x in d), "below": sum(x < -1e-6 for x in d),
                        "equal": sum(abs(x) <= 1e-6 for x in d),
                        "fallback_equal": sum(a[(k, s)].get("fallback_frames") == b[(k, s)].get("fallback_frames") for s in seqs),
                        "mean_difference": float(sum(x * f for x, f in zip(d, fr)) / max(sum(fr), 1)),
                        "above_added": sum(x > 1e-6 for x in add), "below_added": sum(x < -1e-6 for x in add)}
        if res:
            out[run] = res
    print(json.dumps(out, indent=1))
    (Q / "r40_flops_sign.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    {"flops_sign": flops_sign, "davis_final": davis_final, "final": final, "final_short": final_short, "tome3_gate": tome3_gate, "tome3": tome3, "gate": gate, "val": val, "val_detail": val_detail, "tome_detail": tome_detail, "d2val": d2val, "tome_gate": tome_gate, "tome2": tome2, "tome2_detail": tome2_detail, "davis_all": davis_all, "vitb": vitb, "vitb3": vitb3, "iframe_gate": iframe_gate, "d1_heldout": d1_heldout, "iframe_mem_gate": iframe_mem_gate, "surprise_gate": surprise_gate, "relax_gate": relax_gate, "thr_gate": thr_gate, "iperm_controls": iperm_controls, "thr_heldout": thr_heldout, "rest_replication": rest_replication, "combo_gate": combo_gate, "length_gain": length_gain, "oracle_headroom": oracle_headroom, "extras_heldout": extras_heldout, "y4_heldout": y4_heldout, "davis_more": davis_more, "tome_fix": tome_fix, "iframe_val": iframe_val, "vitl": vitl, "full": full, "tome": tome, "davis": davis}[sys.argv[1]](*sys.argv[2:])
