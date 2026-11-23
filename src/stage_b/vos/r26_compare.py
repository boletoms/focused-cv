"""R26 (stronger ViT-S tracker from the official Cutie weights). DAVIS val inside Cutie, R10 protocol and settings: J&F
change from dense per setting, ours against protected ALGM interpolated at equal FLOPs (thresholds 0.77 / 0.83 / 0.91,
R26 rows), and each R26 row against the same R10 row (new head; dense rows differ). Long videos (`lvos`): LVOS val,
reading fixed before the run: (1) dense against R12's 48.97 (paired over objects, identical videos); (2) the closed loop
and the surprise setting against ALGM 0.98 by R14's criterion (beat it, or within 0.5 J&F at lower FLOPs); ground-truth
maps as the R23 reference. Bootstrap 1,000 resamples over objects.
Usage: python -m src.stage_b.vos.r26_compare [lvos]"""
import json
import sys
from pathlib import Path

import numpy as np

V = Path("/work/stage_b/vos")
OURS = "ftome_l0.7_t0.01_floor64_dil1_noprop_cls64"
PALGM = ("palgm_t0.77_dil1", "palgm_t0.83_dil1", "palgm_t0.91_dil1")


def load(run):
    return {(r["setting"], r["sequence"]): r for f in sorted((run / "sequences").glob("*.json"))
            for r in json.loads(f.read_text())}


def tools(by, seqs):
    jf = lambda k: np.concatenate([50 * (np.array(by[(k, s)]["J"]) + np.array(by[(k, s)]["F"])) for s in seqs])
    fl = lambda k: sum(by[(k, s)]["flops_ratio"] * by[(k, s)]["frames"] for s in seqs) / sum(by[(k, s)]["frames"] for s in seqs)
    return jf, fl


def ci(x, rng):
    return [float(x.mean())] + [float(v) for v in np.percentile(
        [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)], [2.5, 97.5])]


def davis():
    a, b = load(V / "eval_r10"), load(V / "eval_r26_cutie")
    seqs = sorted(s for (k, s) in b if k == "dense")
    assert len(seqs) == 30
    ja, fa = tools(a, seqs)
    jb, fb = tools(b, seqs)
    d = jb("dense")
    assert len(d) == 61
    curve = sorted((fb(k), jb(k)) for k in PALGM)

    def at(x0):
        for (x1, y1), (x2, y2) in zip(curve, curve[1:]):
            if x1 <= x0 <= x2:
                return y1 + (y2 - y1) * (x0 - x1) / (x2 - x1)
        return None
    settings = sorted({k for (k, s) in b} - {"dense"})
    out = {"dense": {"jf": float(d.mean()), "minus_r10": ci(d - ja("dense"), np.random.default_rng(0))}}
    print(f"dense {d.mean():.2f}  - R10 dense {out['dense']['minus_r10']}")
    for k in settings:
        rng = np.random.default_rng(0)
        f, y = fb(k), at(fb(k))
        row = {"flops": f, "jf": float(jb(k).mean()), "change": ci(jb(k) - d, rng),
               "change_r10": float((ja(k) - ja("dense")).mean()), "flops_r10": fa(k)}
        if k.startswith("ftome") and y is not None:
            row["minus_protected_algm_at_equal_flops"] = ci(jb(k) - y, rng)
        out[k] = row
        print(f"{k:45s} flops {f:.3f}  change {[round(t, 2) for t in row['change']]}  (R10 {row['change_r10']:+.2f} at "
              f"{row['flops_r10']:.3f})" + (f"  vs pALGM {[round(t, 2) for t in row['minus_protected_algm_at_equal_flops']]}"
                                            if "minus_protected_algm_at_equal_flops" in row else ""))
    out["protected_algm_curve_flops"] = [c[0] for c in curve]
    for x, y in [(OURS, "palgm_t0.77_dil1"), (OURS.replace("l0.7", "l0.3"), "palgm_t0.77_dil1"), (OURS, "algm_t0.98")]:
        out[f"{x} - {y}"] = ci(jb(x) - jb(y), np.random.default_rng(0))
        print(f"{x[:10]} - {y}: {[round(t, 2) for t in out[f'{x} - {y}']]}")
    Path("reports/stage_b/queue/r26_davis.json").write_text(json.dumps(out, indent=1) + "\n")


def lvos():
    a, b = load(V / "eval_r12_lvos"), load(V / "eval_r26_lvos")
    seqs = sorted(s for (k, s) in b if k == "dense")
    ja, _ = tools(a, seqs)
    jb, fb = tools(b, seqs)
    d = jb("dense")
    track = np.where(d > 50)[0]                                         # objects R26 dense tracks (J&F > 50)
    out = {"objects": len(d), "tracked_objects": len(track), "frames": sum(b[("dense", s)]["frames"] for s in seqs), "dense": {"jf": float(d.mean()), "r12_dense": float(ja("dense").mean()),
                                        "minus_r12_dense": ci(d - ja("dense"), np.random.default_rng(0))}}
    print(f"dense {d.mean():.2f}  R12 dense {ja('dense').mean():.2f}  diff {[round(t, 2) for t in out['dense']['minus_r12_dense']]}")
    for x, y in [("algm_t0.98", "algm_t0.98"), (OURS + "~tome_r64_from1_noprop", OURS + "~tome_r64_from1_noprop")]:
        out[f"{x}: R26 - R12"] = ci(jb(x) - ja(y), np.random.default_rng(0))
    ref = "algm_t0.98"
    for k in sorted({k for (k, s) in b} - {"dense"}):
        rng = np.random.default_rng(0)
        row = {"flops": fb(k), "jf": float(jb(k).mean()), "change": ci(jb(k) - d, rng),
               "change_tracked": ci((jb(k) - d)[track], rng),
               "empty_map_frames": sum(b[(k, s)]["fallback_frames"] for s in seqs),
               "trigger_frames": sum(b[(k, s)].get("triggers", 0) for s in seqs)}
        if k != ref:
            diff = ci(jb(k) - jb(ref), rng)
            row["minus_algm098"] = diff
            row["minus_algm098_tracked"] = ci((jb(k) - jb(ref))[track], rng)
            row["r14_criterion"] = bool(diff[0] > 0 or (diff[0] >= -0.5 and fb(k) < fb(ref)))
        out[k] = row
        print(f"{k:60s} flops {row['flops']:.3f}  J&F {row['jf']:.2f}  change {[round(t, 2) for t in row['change']]}"
              + (f"  - ALGM0.98 {[round(t, 2) for t in row['minus_algm098']]}  R14 {row['r14_criterion']}" if k != ref else ""))
    Path("reports/stage_b/queue/r26_lvos.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    lvos() if sys.argv[1:] == ["lvos"] else davis()
