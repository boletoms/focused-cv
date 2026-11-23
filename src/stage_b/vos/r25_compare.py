"""R25 (Cutie retrained on the frozen DINOv3 ViT-B, small head from cutie-small-mega). DAVIS val inside Cutie: J&F change
from dense per setting, ours against protected ALGM interpolated at equal FLOPs (0.83 / 0.95, R25 rows), and dense
against R26's dense (same head recipe on ViT-S: a pure backbone change). Long videos (`lvos`): LVOS val, reading fixed
before the run: (1) dense against R12's 48.97 (and R26's dense), paired over objects; (2) the closed loop and the
surprise setting against plain ALGM (the threshold picked from DAVIS FLOPs, `algm_t0.95`) by R14's criterion (beat it,
or within 0.5 J&F at lower FLOPs); ground-truth maps as the R23 reference. FLOPs relative to ViT-B dense. Bootstrap
1,000 resamples over objects.
R34 (Cutie on ViT-L, same recipe and readings; owner 2026-11-20): `r34` / `r34 lvos` with ViT-L names; dense also
against R25's dense; the plain ALGM reference is the threshold the R34 chain picks (file `r34_algm.txt`).
Usage: python -m src.stage_b.vos.r25_compare [r34] [lvos]"""
import json
import sys
from pathlib import Path

import numpy as np

from .r26_compare import V, ci, load, tools

OURS = "ftome_l0.7_t0.01_floor128_dil2_noprop_cls64"
PALGM = ("palgm_t0.83_dil1", "palgm_t0.95_dil1")
REF = "algm_t0.95"                                                      # chain's pick (DAVIS FLOPs closest to 0.44)
TAG, PREV, PREV_LVOS = "r25", "eval_r26_cutie", "eval_r26_lvos"         # previous tracker: R26 (ViT-S)
if "r34" in sys.argv[1:]:
    OURS, TAG, PREV, PREV_LVOS = "ftome_l0.7_t0.01_floor128_dil1_noprop_cls64", "r34", "eval_r25_cutie", "eval_r25_lvos"
    REF = Path("reports/stage_b/queue/r34_algm.txt").read_text().strip() if Path("reports/stage_b/queue/r34_algm.txt").exists() else REF


def davis():
    a, b = load(V / PREV), load(V / f"eval_{TAG}_cutie")
    seqs = sorted(s for (k, s) in b if k == "dense")
    assert len(seqs) == 30
    ja, _ = tools(a, seqs)
    jb, fb = tools(b, seqs)
    d = jb("dense")
    assert len(d) == 61
    curve = sorted((fb(k), jb(k)) for k in PALGM)

    def at(x0):
        for (x1, y1), (x2, y2) in zip(curve, curve[1:]):
            if x1 <= x0 <= x2:
                return y1 + (y2 - y1) * (x0 - x1) / (x2 - x1)
        return None
    out = {"dense": {"jf": float(d.mean()), "minus_previous": ci(d - ja("dense"), np.random.default_rng(0)), "previous": PREV}}
    print(f"dense {d.mean():.2f}  - previous dense {[round(t, 2) for t in out['dense']['minus_previous']]}")
    for k in sorted({k for (k, s) in b} - {"dense"}):
        rng = np.random.default_rng(0)
        f, y = fb(k), at(fb(k))
        row = {"flops": f, "jf": float(jb(k).mean()), "change": ci(jb(k) - d, rng)}
        if k.startswith("ftome") and y is not None:
            row["minus_protected_algm_at_equal_flops"] = ci(jb(k) - y, rng)
        out[k] = row
        print(f"{k:45s} flops {f:.3f}  change {[round(t, 2) for t in row['change']]}"
              + (f"  vs pALGM {[round(t, 2) for t in row['minus_protected_algm_at_equal_flops']]}"
                 if "minus_protected_algm_at_equal_flops" in row else ""))
    out["protected_algm_curve_flops"] = [c[0] for c in curve]
    for x, y in [(OURS, "palgm_t0.83_dil1"), (OURS.replace("l0.7", "l0.3"), "palgm_t0.95_dil1"), (OURS, REF)]:
        out[f"{x} - {y}"] = ci(jb(x) - jb(y), np.random.default_rng(0))
        print(f"{x[:10]}{x[6:10]} - {y}: {[round(t, 2) for t in out[f'{x} - {y}']]}")
    Path(f"reports/stage_b/queue/{TAG}_davis.json").write_text(json.dumps(out, indent=1) + "\n")


def lvos():
    a, r26, b = load(V / "eval_r12_lvos"), load(V / PREV_LVOS), load(V / f"eval_{TAG}_lvos")
    seqs = sorted(s for (k, s) in b if k == "dense")
    ja, _ = tools(a, seqs)
    j26, f26 = tools(r26, seqs)
    jb, fb = tools(b, seqs)
    d = jb("dense")
    assert len(d) == len(ja("dense")) == len(j26("dense"))
    track = np.where(d > 50)[0]                                         # objects R25 dense tracks (J&F > 50)
    out = {"objects": len(d), "tracked_objects": len(track), "previous_tracked_objects": int((j26("dense") > 50).sum()),
           "dense": {"jf": float(d.mean()), "minus_r12_dense": ci(d - ja("dense"), np.random.default_rng(0)),
                     "minus_previous_dense": ci(d - j26("dense"), np.random.default_rng(0))}}
    print(f"dense {d.mean():.2f}  - R12 dense {[round(t, 2) for t in out['dense']['minus_r12_dense']]}"
          f"  - previous dense {[round(t, 2) for t in out['dense']['minus_previous_dense']]}  tracked {len(track)}")
    for k in sorted({k for (k, s) in b} - {"dense"}):
        rng = np.random.default_rng(0)
        row = {"flops": fb(k), "jf": float(jb(k).mean()), "change": ci(jb(k) - d, rng),
               "change_tracked": ci((jb(k) - d)[track], rng),
               "empty_map_frames": sum(b[(k, s)]["fallback_frames"] for s in seqs),
               "trigger_frames": sum(b[(k, s)].get("triggers", 0) for s in seqs)}
        if k != REF:
            diff = ci(jb(k) - jb(REF), rng)
            row["minus_ref"] = diff
            row["minus_ref_tracked"] = ci((jb(k) - jb(REF))[track], rng)
            row["r14_criterion"] = bool(diff[0] > 0 or (diff[0] >= -0.5 and fb(k) < fb(REF)))
        out[k] = row
        print(f"{k[-50:]:50s} flops {row['flops']:.3f}  J&F {row['jf']:.2f}  change {[round(t, 2) for t in row['change']]}"
              + (f"  - {REF} {[round(t, 2) for t in row['minus_ref']]}  R14 {row['r14_criterion']}" if k != REF else ""))
    out["reference"] = REF
    Path(f"reports/stage_b/queue/{TAG}_lvos.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    lvos() if "lvos" in sys.argv[1:] else davis()
