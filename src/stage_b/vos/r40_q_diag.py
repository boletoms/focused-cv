"""Plan section S, diagnostics first: what the per-frame logs of `eval_r40diag_q` (method, `@fbkey20`, `@key20` on 34
development videos, `VOS_DIAG=1`) say about drift, recovery and tracking margin. Reads logs only. Writes
`reports/stage_b/queue/r40_q_diag.json` and prints a table. Questions, fixed before the run:
1. Margin: the share of an object's ground-truth pixels inside protected patches (`map_cov`) and the share of
   object-frames with `map_cov` < 0.9 (misses), by frames since the last I-frame (phase bins of 5 frames; phase is
   t mod 20 for every setting, so the method serves as the no-I-frame control) and by object speed (ground-truth
   centroid, patches per frame: slow < 0.5, mid 0.5 to 2, fast > 2). If I-frames raise coverage right after them and
   it decays with phase, drift is real and the schedule bounds it; if coverage is flat in phase, a tighter map is not
   made safe by this schedule.
2. Recovery: J (annotated frames) by phase; the same contrast.
3. Fallback accounting: frames with the fallback, dense, or an empty map per setting, and the frames where the method
   fell back but the variant did not.
The strata are the 14 fast-and-long videos, the other fast ones and the other long ones (`r40_q_diag.txt`)."""
import json
from pathlib import Path

import numpy as np

V = Path("/work/stage_b/vos") if Path("/work/stage_b/vos").exists() else Path.home() / "ml_work/stage_b/vos"
Q = Path("reports/stage_b/queue")
PERIOD, BINS = 20, (0, 5, 10, 15, 20)
SPEED = {"slow": (0, 0.5), "mid": (0.5, 2), "fast": (2, 1e9)}


def settings(d):
    names = json.loads((d / "settings.json").read_text())
    out = {}
    for i, n in names.items():
        tail = n.split("@")[-1]
        if tail.startswith(("key", "fbkey")):
            out[tail] = int(i)
        elif "exp-" in n:
            out["method"] = int(i)                                # the setting without a schedule
    return out


def stratum_of(v, fast, long_):
    return "fast_long" if v in fast and v in long_ else ("fast_other" if v in fast else "long_other")


def one(z):
    """Per object-frame arrays: phase, coverage, J, speed (patches per frame)."""
    t, cov, j = z["t"], z["map_cov"], z["j"]
    gc = z["gt_c"]                                                    # (T, O, 2) pixels, NaN if absent
    ph = np.tile((t % PERIOD)[:, None], (1, cov.shape[1]))
    sp = np.full_like(cov, np.nan, dtype=float)
    dt = np.diff(t).clip(min=1)
    for o in range(cov.shape[1]):
        step = np.linalg.norm(np.diff(gc[:, o], axis=0), axis=1) / 16.0 / dt
        sp[1:, o] = step
    return ph.ravel(), cov.ravel(), j.ravel(), sp.ravel()


def table(rows):
    out = {}
    for lab, sel in rows.items():
        ph, cov, j, sp = (np.concatenate(x) for x in zip(*sel)) if sel else (np.array([]),) * 4
        ok = ~np.isnan(cov) & ~np.isnan(sp)
        res = {"object_frames": int(ok.sum())}
        for a, b in zip(BINS[:-1], BINS[1:]):
            m = ok & (ph >= a) & (ph < b)
            mj = m & ~np.isnan(j)
            res[f"phase{a}-{b}"] = {"cov": float(cov[m].mean()) if m.any() else None,
                                    "miss": float((cov[m] < 0.9).mean()) if m.any() else None,
                                    "J": float(j[mj].mean()) if mj.any() else None, "n": int(m.sum())}
        for nm, (a, b) in SPEED.items():
            m = ok & (sp >= a) & (sp < b)
            res[f"speed_{nm}"] = {"cov": float(cov[m].mean()) if m.any() else None,
                                  "miss": float((cov[m] < 0.9).mean()) if m.any() else None, "n": int(m.sum())}
        out[lab] = res
    return out


def main(out_dir="eval_r40diag_q"):
    d = V / out_dir / "diag"
    fast, long_ = (set((Q / f).read_text().split()) for f in ("r40_gate_subset.txt", "r40_q_long.txt"))
    per, acct = {}, {}
    for vd in sorted(p for p in d.iterdir() if p.is_dir()):
        idx = settings(vd)
        z = {k: np.load(vd / f"{i}.npz") for k, i in idx.items()}
        st = stratum_of(vd.name, fast, long_)
        for k, zz in z.items():
            for lab in (k, f"{k}|{st}"):
                per.setdefault(lab, []).append(one(zz))
            a = acct.setdefault(k, {"frames": 0, "fallback": 0, "dense": 0, "empty": 0, "method_fb_not_here": 0})
            a["frames"] += len(zz["t"]); a["fallback"] += int(zz["fallback"].sum())
            a["dense"] += int(zz["dense"].sum()); a["empty"] += int(zz["empty"].sum())
            if k != "method" and "method" in z and len(z["method"]["t"]) == len(zz["t"]):
                a["method_fb_not_here"] += int((z["method"]["fallback"].astype(bool) & ~zz["fallback"].astype(bool)).sum())
    res = {"phase_and_speed": table(per), "accounting": acct}
    print(json.dumps(acct, indent=1))
    for lab, r in res["phase_and_speed"].items():
        print(lab, r["object_frames"], {k: (round(v["cov"], 3), round(v["miss"], 3)) for k, v in r.items() if k.startswith(("phase", "speed")) and v["cov"] is not None})
    (Q / "r40_q_diag.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
