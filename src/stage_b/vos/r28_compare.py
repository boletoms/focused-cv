"""R28 stage C analysis (surprise in closed loop, R26 weights). `davis`: DAVIS val inside Cutie, each method with the
detector against itself without it (paired over the 61 objects), FLOPs and surprise counts. `lvos`: the LVOS val run of
the setting chosen on LVOS train, against the R26 rows (identical dense rows): ours without surprise, and ALGM 0.98 by
R14's criterion (beat it, or within 0.5 J&F at lower FLOPs); all objects and the objects R26 dense tracks.
Bootstrap 1,000 resamples over objects. Stage C2 passes its run directory and output name. The ViT-B run (R28b, after
C2) passes the R25 rows, our ViT-B setting and plain ALGM 0.95 as the reference.
Usage: python -m src.stage_b.vos.r28_compare <davis|lvos> [run dir name] [out json name] [ref run] [ours] [ref]"""
import json
import sys
from pathlib import Path

import numpy as np

V = Path("/work/stage_b/vos")
OURS = "ftome_l0.7_t0.01_floor64_dil1_noprop_cls64"


def load(run):
    return {(r["setting"], r["sequence"]): r for f in sorted((run / "sequences").glob("*.json"))
            for r in json.loads(f.read_text())}


def ci(x, seed=0):
    rng = np.random.default_rng(seed)
    return [float(x.mean())] + [float(v) for v in np.percentile(
        [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)], [2.5, 97.5])]


def tools(by, seqs):
    jf = lambda k: np.concatenate([50 * (np.array(by[(k, s)]["J"]) + np.array(by[(k, s)]["F"])) for s in seqs])
    fl = lambda k: sum(by[(k, s)]["flops_ratio"] * by[(k, s)]["frames"] for s in seqs) / sum(by[(k, s)]["frames"] for s in seqs)
    srp = lambda k: {x: sum(by[(k, s)].get(x, 0) for s in seqs) for x in ("srp_frames", "srp_global", "frames")} | {
        "srp_share": float(np.mean([by[(k, s)].get("srp_share", 0) for s in seqs]))}
    return jf, fl, srp


def davis(run="eval_r28_davis", out_name="r28_davis.json"):
    b = load(V / run)
    seqs = sorted(s for (k, s) in b if k == "dense")
    jf, fl, srp = tools(b, seqs)
    d = jf("dense")
    out = {"dense": float(d.mean())}
    for k in sorted({k for (k, s) in b} - {"dense"}):
        out[k] = {"flops": fl(k), "change": ci(jf(k) - d), "surprise": srp(k)}
        if "@srp_" in k:
            base = k.split("@")[0]
            out[k]["minus_without"] = ci(jf(k) - jf(base))
            out[k]["flops_added"] = fl(k) - fl(base)
        print(k[-60:], {x: (round(v, 3) if isinstance(v, float) else v) for x, v in out[k].items() if x != "surprise"},
              out[k]["surprise"])
    Path("reports/stage_b/queue", out_name).write_text(json.dumps(out, indent=1) + "\n")


def lvos(run="eval_r28_val", out_name="r28_lvos.json", ref_run="eval_r26_lvos", ours=OURS, ref="algm_t0.98"):
    r26, v = load(V / ref_run), load(V / run)
    seqs = sorted(s for (k, s) in v if k == "dense")
    ja, fa, _ = tools(r26, seqs)
    jv, fv, srp = tools(v, seqs)
    d = jv("dense")
    assert np.array_equal(d, ja("dense")), f"dense rows differ from {ref_run}"
    track = np.where(d > 50)[0]
    chosen = sorted({k for (k, s) in v} - {"dense"})
    out = {"objects": len(d), "tracked": len(track)}
    for k in chosen:
        row = {"flops": fv(k), "jf": float(jv(k).mean()), "change": ci(jv(k) - d), "surprise": srp(k)}
        for name, r in (("ours without surprise", ours + "~tome_r64_from1_noprop"), ("reference", ref)):
            diff = jv(k) - ja(r)
            row[f"minus {name}"] = {"all": ci(diff), "tracked": ci(diff[track]), "ref_flops": fa(r), "ref_setting": r}
        m = row["minus reference"]["all"][0]
        row["r14_criterion"] = bool(m > 0 or (m >= -0.5 and row["flops"] < fa(ref)))
        out[k] = row
        print(json.dumps({k: row}, indent=1))
    Path("reports/stage_b/queue", out_name).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    (davis if sys.argv[1] == "davis" else lvos)(*sys.argv[2:])
