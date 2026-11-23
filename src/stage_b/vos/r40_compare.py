"""R40 redesign: gate and validation comparisons (rules fixed before any run: `reports/stage_b/queue/R40_expectation_plan.md`
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
OURS = "ftome_l0.7_t0.01_floor64_dil1_noprop_cls64~tome_r64_from1_noprop"
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
            a, o = jn(k) - jr(REF), jn(k) - jr(OURS)
            res[tag] = {"videos": len(seqs), "objects": len(a), "jf": float(jn(k).mean()), "flops": fn(k),
                        "algm": {"jf": float(jr(REF).mean()), "flops": fr(REF)}, "ours": {"jf": float(jr(OURS).mean()), "flops": fr(OURS)},
                        "minus_algm": ci(a), "minus_ours": ci(o), "minus_dense": ci(jn(k) - jr("dense")),
                        "stats": _exp_stats(run, k, seqs)}
        s, a = res["subset"], res.get("all", {})
        c1 = s["minus_ours"][1] > 0
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
    (`eval_r40d2_*`) against the redesign (`eval_r40_*`, the validation rows) and against ALGM 0.98, validation protocol
    (head-averaged per object). Goes to the owner if the difference to the redesign has a lower bound above −1.0 at FLOPs
    at least 10% lower; otherwise full resolution in the map stands. No design change on the validation split alone."""
    a = val("eval_r40d2", {h: (f"eval_r40_{h}_v1val", f"eval_r40_{h}_v2new") for h in HEADS}, S_FINAL,
            "r40_d2val_vs_redesign.json")[D2_FINAL]
    b = val("eval_r40d2", HEADS, REF, "r40_d2val_vs_algm.json")[D2_FINAL]
    out = {"setting": D2_FINAL, "objects": a["objects"], "flops": a["flops"],
           "minus_redesign": a["minus_algm_head_avg"], "redesign_flops": a["algm_flops"],
           "minus_redesign_per_head": {h: v["minus_algm"] for h, v in a["per_head"].items()},
           "minus_algm": b["minus_algm_head_avg"], "algm_flops": b["algm_flops"],
           "minus_algm_per_head": {h: v["minus_algm"] for h, v in b["per_head"].items()},
           "noninferior_algm": b["noninferior"], "superior_algm": b["superior"]}
    out["to_owner"] = bool(out["minus_redesign"][1] > -1.0 and out["flops"] <= 0.9 * out["redesign_flops"])
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
        _same_objects(run, TT, ref, OURS, seqs)
        jn, fn, _ = tools(run, seqs)
        jt, ft, _ = tools(tr, seqs)
        jr, fr, _ = tools(ref, seqs)
        res[tag] = {"videos": len(seqs), "objects": len(jn(TT)), "jf": float(jn(TT).mean()), "flops": fn(TT),
                    "tome": {"jf": float(jt(TOME).mean()), "flops": ft(TOME)}, "ours": {"jf": float(jr(OURS).mean()), "flops": fr(OURS)},
                    "minus_tome": ci(jn(TT) - jt(TOME)), "minus_ours": ci(jn(TT) - jr(OURS)),
                    "minus_dense": ci(jn(TT) - jr("dense")), "stats": _exp_stats(run, TT, seqs)}
        if new:
            res[tag]["switched"] = {k: sum(run[(TT, s)].get(k, 0) for s in seqs) for k in ("mbase_frames", "membase_frames")}
    s, a = res["subset"], res.get("all", {})
    c1 = s["minus_ours"][1] > 0
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
    """Plan section N (R34): the redesign at ViT-L, ALGM at the threshold R25's rule picks on DAVIS (`r34_algm.txt`;
    None before the pick), budget floor 128 and margin 1 from R4's ViT-L choice."""
    p = Q / "r34_algm.txt"
    if not p.exists():
        return None
    t = p.read_text().strip().removeprefix("algm_t")
    return f"titome_t{t}_l0.7_floor128_dil1_cls64~algm_t{t}@{E_FINAL}"


SL_FINAL = _sl_final()


def davis_all():
    """Plan section J1 (reported): short clips inside Cutie with every baseline, one run per backbone (ViT-S R26 head,
    ViT-B R25 head; fast path, deterministic kernels): each setting's J&F, FLOPs and change from dense; the redesign
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
                res["settings"][k]["redesign_minus"] = ci(jf(new) - jf(k))
        out[tag] = res
    print(json.dumps(out, indent=1))
    (Q / "r40_davis_all.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def davis_final(run="eval_r40n", out_tag="final_thr"):
    """Plan section L4c (reported): Tables 3 and 4 with the redesign's row from the final implementation (`<run>_davis`,
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
               "redesign": {"jf": float(ja(new).mean()), "flops": fa(new), "minus_dense": ci(ja(new) - jf("dense"))},
               "settings": {}}
        for k in sorted({k for (k, s) in b} - {"dense", new}):
            res["settings"][k] = {"jf": float(jf(k).mean()), "flops": fl(k), "minus_dense": ci(jf(k) - jf("dense")),
                                  "redesign_minus": ci(ja(new) - jf(k))}
        out[tag] = res
    print(json.dumps(out, indent=1))
    (Q / f"r40_{out_tag}_davis_all.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def vitb():
    """Plan section J2 (reported, one head): ViT-B long videos (R25 head; LVOS V1 val and the 90 new V2 valid): the
    redesign for ViT-B against plain ALGM 0.95 and dense, per split and pooled, per object."""
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
    videos in one run per method, masks saved). Short clips: every R34 setting (`eval_oml_davis_all`) and the redesign's
    row on the final implementation (`eval_oml_davis_r40`) against dense, the redesign against each setting (paired over
    objects). Long videos: the redesign (`eval_oml_r40`) against the picked plain ALGM (`r34_algm.txt`) and dense
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
                    "redesign": {"jf": float(ja(new).mean()), "flops": fa(new), "minus_dense": ci(ja(new) - jf("dense"))},
                    "settings": {k: {"jf": float(jf(k).mean()), "flops": fl(k), "minus_dense": ci(jf(k) - jf("dense")),
                                     "redesign_minus": ci(ja(new) - jf(k))} for k in sorted({k for (k, s) in b} - {"dense"})}}
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
    final rows (`eval_r40l_davis`, `eval_r40lb_davis`) against dense and the eager redesign rows of J1. ViT-B long
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
    """Short-clip check (R26 head, DAVIS val): each setting against dense; the redesign against ALGM 0.98 and ours
    (paired over objects). Reported; no rule attached."""
    b = load(V / "eval_r40_davis")
    seqs = sorted(s for (k, s) in b if k == "dense")
    jf, fl, _ = tools(b, seqs)
    out = {"videos": len(seqs), "objects": len(jf("dense")), "dense": float(jf("dense").mean())}
    for k in sorted({k for (k, s) in b} - {"dense"}):
        out[k] = {"jf": float(jf(k).mean()), "flops": fl(k), "minus_dense": ci(jf(k) - jf("dense"))}
        if "@exp-" in k:
            out[k] |= {"minus_algm": ci(jf(k) - jf(REF)), "minus_ours": ci(jf(k) - jf(OURS)),
                       "global_frames": sum(b[(k, s)].get("exp_global", 0) for s in seqs),
                       "frames": sum(b[(k, s)]["frames"] for s in seqs)}
    print(json.dumps(out, indent=1))
    (Q / "r40_davis.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def tome():
    """Plan section E: `ttome_...@exp-...` (eval_r40e_*) against plain ToMe r64 (eval_tome_*), R-25 protocol."""
    return val("eval_r40e", TOME_HEADS, TOME, "r40_tome_val.json")


def full():
    """Plan section D (owner question): D1 protected ALGM (map at full resolution) against plain ALGM 0.98 (DG1 rows);
    D2 the redesign with the high tier also merged by ALGM's rules against the redesign (gate rows). D2 goes to the owner
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
    {"flops_sign": flops_sign, "davis_final": davis_final, "final": final, "final_short": final_short, "tome3_gate": tome3_gate, "tome3": tome3, "gate": gate, "val": val, "val_detail": val_detail, "tome_detail": tome_detail, "d2val": d2val, "tome_gate": tome_gate, "tome2": tome2, "tome2_detail": tome2_detail, "davis_all": davis_all, "vitb": vitb, "vitb3": vitb3, "vitl": vitl, "full": full, "tome": tome, "davis": davis}[sys.argv[1]](*sys.argv[2:])
