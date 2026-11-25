"""R40 plan section H (review flag R-33): where the validation gain comes from. Logged rows (`VOS_DIAG=1`, deterministic
kernels) of dense, ALGM 0.98 and the expectation focus map, head R31c, LVOS V1 val and the 90 new V2 valid videos
(`eval_r40_attr_r31c_*`). Check first: the logged rows equal the validation rows, and the logged per-frame values give
the scored per-object J&F. Decomposition (exact): per object, the J&F difference is the sum over frame classes of
(share of its scored frames) x (mean per-frame difference in the class); classes from the expectation focus map's own state:
`before` the video's first fallback frame, `fallback` frames (every object lost, plain ALGM 0.98 runs), `after`
(non-fallback frames after a fallback episode). Also, descriptively (cells defined on outcomes), the cross-table of
per-frame loss (J below 0.1) under the expectation focus map and under ALGM 0.98. Reported, no rule. CPU.
Usage (ml-base:vos): python -m src.stage_b.vos.r40_attr          (section H, writes r40_attr.json)
                     python -m src.stage_b.vos.r40_attr test [n]  (the same on n DG1 videos, ours against ALGM; prints)"""
import json
import sys

import numpy as np

from .r28_compare import V, ci, load
from .r40_compare import CLOSED_LOOP, Q, REF, S_FINAL

RUNS = {"v1val": ("eval_r40_attr_r31c_v1val", "eval_r40_r31c_v1val", "eval_r31c_lvos_det"),
        "v2new": ("eval_r40_attr_r31c_v2new", "eval_r40_r31c_v2new", "eval_r31c_v2val")}
CLASSES = ("before", "fallback", "after")
CELLS = ("neither_lost", "only_ref_lost", "only_new_lost", "both_lost")
LOSS = 0.1
TOL = 1e-6


def _jf(row):
    """Scored per-object J&F (x100) of a row, by object id."""
    return {o: (j + f) / 2 * 100 for o, j, f in zip(row["objects"], row["J"], row["F"])}


def _max_diff(a, b):
    assert a.keys() == b.keys(), (sorted(a), sorted(b))
    return max(abs(a[o] - b[o]) for o in a)


def video(run, seq, new, ref, rows):
    """Per scored object of one video: frame count, class and cell shares and contributions (x100 J&F)."""
    d = V / run / "diag" / seq
    idx = {k: i for i, k in json.loads((d / "settings.json").read_text()).items()}
    z = {k: np.load(d / f"{idx[k]}.npz") for k in ("dense", ref, new)}
    j = {k: z[k]["j"] for k in z}
    jf = {k: (z[k]["j"] + z[k]["f"]) / 2 * 100 for k in z}                  # (T, K), NaN where not scored
    keep = [o for o in range(jf[new].shape[1]) if (~np.isnan(jf[new][:, o])).any()]
    ids = sorted(rows[(new, seq)]["objects"])
    assert len(ids) == len(keep), (seq, ids, keep)
    for k in z:                                                              # logged frames give the scored values
        got = {i: float(np.nanmean(jf[k][:, o])) for i, o in zip(ids, keep)}
        assert _max_diff(got, _jf(rows[(k, seq)])) < TOL, (seq, k)
    fb = z[new]["fallback"].astype(bool)
    first = int(np.argmax(fb)) if fb.any() else len(fb)
    cls = np.where(fb, 1, np.where(np.arange(len(fb)) < first, 0, 2))
    out = []
    for i, o in zip(ids, keep):
        ok = ~np.isnan(jf[new][:, o])
        assert all((ok == ~np.isnan(jf[k][:, o])).all() for k in z), (seq, i)
        n = int(ok.sum())
        dr, dd = jf[new][ok, o] - jf[ref][ok, o], jf[new][ok, o] - jf["dense"][ok, o]
        c = cls[ok]
        cell = 2 * (j[new][ok, o] < LOSS) + (j[ref][ok, o] < LOSS)
        out.append({"video": seq, "object": int(i), "frames": n,
                    "jf": {k: float(jf[k][ok, o].mean()) for k in ("dense", ref, new)},
                    "share": [float((c == q).mean()) for q in range(3)],
                    "n": [int((c == q).sum()) for q in range(3)],
                    "sum_ref": [float(dr[c == q].sum()) for q in range(3)],
                    "sum_dense": [float(dd[c == q].sum()) for q in range(3)],
                    "cell_n": [int((cell == q).sum()) for q in range(4)],
                    "cell_sum_ref": [float(dr[cell == q].sum()) for q in range(4)]})
    return out


def summarise(objs):
    """Object-mean difference and its exact split over classes and cells, with bootstrap intervals over objects."""
    n = np.array([o["frames"] for o in objs], float)
    S = {"objects": len(objs), "frames": int(n.sum())}
    tot_r = np.array([sum(o["sum_ref"]) for o in objs]) / n
    tot_d = np.array([sum(o["sum_dense"]) for o in objs]) / n
    S["minus_ref"], S["minus_dense"] = ci(tot_r), ci(tot_d)
    S["classes"] = {}
    for q, name in enumerate(CLASSES):
        nq = np.array([o["n"][q] for o in objs], float)
        sr = np.array([o["sum_ref"][q] for o in objs]); sd = np.array([o["sum_dense"][q] for o in objs])
        S["classes"][name] = {"frame_share": float(nq.sum() / n.sum()), "object_mean_share": float((nq / n).mean()),
                              "objects_with_frames": int((nq > 0).sum()),
                              "mean_frame_minus_ref": float(sr.sum() / max(nq.sum(), 1)),
                              "mean_frame_minus_dense": float(sd.sum() / max(nq.sum(), 1)),
                              "contribution_ref": ci(sr / n), "contribution_dense": ci(sd / n)}
    S["cells"] = {}
    for q, name in enumerate(CELLS):
        nq = np.array([o["cell_n"][q] for o in objs], float)
        sr = np.array([o["cell_sum_ref"][q] for o in objs])
        S["cells"][name] = {"frame_share": float(nq.sum() / n.sum()),
                            "mean_frame_minus_ref": float(sr.sum() / max(nq.sum(), 1)), "contribution_ref": ci(sr / n)}
    parts = sum(S["classes"][c]["contribution_ref"][0] for c in CLASSES)
    assert abs(parts - S["minus_ref"][0]) < 1e-6, (parts, S["minus_ref"][0])           # the split is exact
    return S


def main(mode="val", n_test="30"):
    if mode == "test":                                                       # DG1 logs: ours against ALGM 0.98
        rows = load(V / "diag_dg1")
        seqs = sorted(s for (k, s) in rows if k == CLOSED_LOOP)[: int(n_test)]
        objs = [o for s in seqs for o in video("diag_dg1", s, CLOSED_LOOP, REF, rows)]
        print(json.dumps(summarise(objs), indent=1))
        return
    out, every = {"setting": S_FINAL, "reference": REF, "head": "r31c", "loss_threshold": LOSS}, []
    for tag, (run, val_new, val_ref) in RUNS.items():
        rows, vn, vr = load(V / run), load(V / val_new), load(V / val_ref)
        seqs = sorted(s for (k, s) in rows if k == S_FINAL)
        assert len(seqs) == len(sorted(s for (k, s) in vn if k == S_FINAL)), (tag, len(seqs))
        check = {k: max(_max_diff(_jf(rows[(k, s)]), _jf(src[(k, s)])) for s in seqs)
                 for k, src in ((S_FINAL, vn), (REF, vr), ("dense", vr))}
        assert all(v < TOL for v in check.values()), (tag, check)              # logged rows = validation rows
        objs = [o for s in seqs for o in video(run, s, S_FINAL, REF, rows)]
        out[tag] = {"videos": len(seqs), "check_max_diff": check} | summarise(objs)
        every += objs
    out["all"] = summarise(every)
    print(json.dumps(out, indent=1))
    (Q / "r40_attr.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
