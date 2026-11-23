"""R40 plan sections O and P (reported): the reruns with saved masks under our scorer and the official toolkits
(`official_eval.py`). Per object the pairing of section C; FLOPs from the rerun rows.
`check`: the reruns against the stored rows under our scorer (share of objects with equal J and F, largest J&F
difference), per run pair.
`claim`: section C's comparison on the rerun rows (ViT-S heads R31c, h2, h3; against ALGM 0.98 and dense), under both
scorers, pooled, per split (V1 val, the 90 new) and by the toolkit's seen / unseen; the toolkit-style averages
(LVOS: the mean of seen and unseen J and F).
`vitb3`: plan section P (ViT-B heads r41_b1 to b3; against ALGM 0.95; section C's rules on our scorer).
`vitl`: plan section N on the rerun rows (one head; against the picked plain ALGM and dense).
`davis`: Tables 3 and 4 under both scorers (every setting against dense, the redesign against each), per backbone.
`all`: every part whose runs exist -> `reports/stage_b/queue/r40_official.json`, with Holm-adjusted two-sided bootstrap
p-values (1,000 resamples, as the intervals) for each family of secondary comparisons: per backbone the method against
every short-clip baseline; per protocol the per-head and per-split differences. The pre-registered primary comparison
(head-averaged, against the reference merger) is not adjusted.
Usage: python -m src.stage_b.vos.r40_official [check|claim|vitb3|vitl|davis|all]"""
import json
import sys
from pathlib import Path

import numpy as np

from .r28_compare import V, ci, load
from .r40_compare import HEADS, Q, S_FINAL, SB_FINAL, SL_FINAL

NEW90 = set((Q / "v2valid_new.txt").read_text().split())
V1 = {p.name for p in Path("/work/data/lvos/valid/JPEGImages").iterdir()}       # LVOS V1 val (50 videos)


def pval(x, seed=0):
    """Two-sided bootstrap p-value of a mean difference (the resamples of `ci`)."""
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)])
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))


def holm(ps):
    """{name: p} -> {name: Holm-adjusted p}."""
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def _off(run, k):
    d = json.loads((V / run / "official" / f"{k}.json").read_text())
    return {(s, int(o)): v for s, objs in d["sequences"].items() for o, v in objs.items()}, set(d["unseen"])


class Rows:
    """One setting of one run: per object (sequence, object id) in row order, our J&F, the official J&F, FLOPs."""

    def __init__(self, run, k, seqs=None):
        by = load(V / run)
        for r in by.values():                                           # DAVIS rows carry no ids: objects 1..n, as the
            r.setdefault("objects", list(range(1, len(r["J"]) + 1)))    # official files number them
        self.seqs = sorted(s for (kk, s) in by if kk == k and (seqs is None or s in seqs))
        self.keys = [(s, o) for s in self.seqs for o in by[(k, s)]["objects"]]
        self.ours = np.concatenate([50 * (np.array(by[(k, s)]["J"]) + np.array(by[(k, s)]["F"])) for s in self.seqs])
        self.fr = sum(by[(k, s)]["frames"] for s in self.seqs)
        self.flops = sum(by[(k, s)]["flops_ratio"] * by[(k, s)]["frames"] for s in self.seqs) / self.fr
        self.raw = {(s, o): (j, f) for s in self.seqs for o, j, f in zip(by[(k, s)]["objects"], by[(k, s)]["J"], by[(k, s)]["F"])}
        p = V / run / "official" / f"{k}.json"
        if p.exists():
            off, self.unseen = _off(run, k)
            self.off_jf = {key: off[key] for key in self.keys if key in off}
            self.off = np.array([50 * sum(off[key]) if key in off else np.nan for key in self.keys])
        else:
            self.off, self.off_jf, self.unseen = None, {}, set()

    def sel(self, which, mask):
        x = self.ours if which == "ours" else self.off
        return x[mask]

    def toolkit(self):
        """The toolkit's global numbers over these objects (LVOS: mean of seen and unseen J and F; DAVIS: J and F)."""
        if self.off is None:
            return None
        js = {u: [v[0] for (s, o), v in self.off_jf.items() if (s in self.unseen) == u] for u in (False, True)}
        fs = {u: [v[1] for (s, o), v in self.off_jf.items() if (s in self.unseen) == u] for u in (False, True)}
        out = {"J": float(np.nanmean(js[False] + js[True])) * 100, "F": float(np.nanmean(fs[False] + fs[True])) * 100}
        out["J&F"] = (out["J"] + out["F"]) / 2
        if js[True]:
            out |= {"J_seen": float(np.nanmean(js[False])) * 100, "F_seen": float(np.nanmean(fs[False])) * 100,
                    "J_unseen": float(np.nanmean(js[True])) * 100, "F_unseen": float(np.nanmean(fs[True])) * 100}
            out["J&F_lvos"] = (out["J_seen"] + out["F_seen"] + out["J_unseen"] + out["F_unseen"]) / 4
        return out


def _pair(a, b):
    assert a.keys == b.keys, "objects differ"
    return a, b


def _diffs(new, ref):
    """Per object: ours and official differences; objects the toolkit leaves out (no inner frame) are dropped from the
    official difference only (count reported)."""
    _pair(new, ref)
    d = {"ours": new.ours - ref.ours}
    if new.off is not None and ref.off is not None:
        x = new.off - ref.off
        d["official"], d["official_dropped"] = x[~np.isnan(x)], int(np.isnan(x).sum())
    return d


def _same(a, b):
    """Our scorer on the rerun against the stored rows: objects with equal J and F, largest J&F difference."""
    keys = [k for k in a.raw if k in b.raw]
    eq = sum(a.raw[k] == b.raw[k] for k in keys)
    big = max(abs(50 * (sum(a.raw[k]) - sum(b.raw[k]))) for k in keys) if keys else None
    return {"objects": len(keys), "equal": int(eq), "max_abs_jf_diff": big}


def check():
    out = {}
    for h, (r1, r2) in HEADS.items():
        for k in ("dense", "algm_t0.98"):
            if (V / f"eval_om_{h}_base").exists():
                out[f"{h} {k} V1"] = _same(Rows(f"eval_om_{h}_base", k, V1), Rows(r1, k, V1))
                out[f"{h} {k} new90"] = _same(Rows(f"eval_om_{h}_base", k, NEW90), Rows(r2, k, NEW90))
        if (V / f"eval_om_{h}_r40").exists():
            out[f"{h} redesign V1"] = _same(Rows(f"eval_om_{h}_r40", S_FINAL, V1), Rows(f"eval_r40o_{h}_v1val", S_FINAL))
            out[f"{h} redesign new90"] = _same(Rows(f"eval_om_{h}_r40", S_FINAL, NEW90), Rows(f"eval_r40o_{h}_v2new", S_FINAL, NEW90))
    for new, old in (("eval_om_davis_all", "eval_r40_davis_all"), ("eval_om_davis_r40", "eval_r40o_davis"),
                     ("eval_omb_davis_all", "eval_r40b_davis"), ("eval_omb_davis_r40", "eval_r40ob_davis"),
                     ("eval_oml_davis_all", "eval_r34_cutie")):
        if (V / new).exists():
            ks = sorted({k for (k, s) in load(V / new)} & {k for (k, s) in load(V / old)})
            for k in ks:
                out[f"{new} {k}"] = _same(Rows(new, k), Rows(old, k))
    if (V / "eval_oml_base").exists() and (V / "eval_r34_v1val").exists():
        for k in ("dense", (Q / "r34_algm.txt").read_text().strip()):
            out[f"vitl {k} V1"] = _same(Rows("eval_oml_base", k, V1), Rows("eval_r34_v1val", k, V1))
    return out


def _long(heads, new_k, ref_k, rules):
    """heads: {name: (base run, redesign run)}; head-averaged per-object differences (section C) under both scorers,
    pooled and per split / seen-unseen; per head; toolkit-style averages per head."""
    parts = {"all": None, "v1val": V1, "new90": NEW90}
    out = {"setting": new_k, "reference": ref_k, "heads": list(heads)}
    rows = {h: {k: Rows(r, k) for r, k in ((b, ref_k), (b, "dense"), (n, new_k))} for h, (b, n) in heads.items()}
    keys = rows[next(iter(heads))][new_k].keys
    assert all(r.keys == keys for h in heads for r in rows[h].values()), "objects differ between rows"
    unseen = rows[next(iter(heads))][new_k].unseen
    for part, want in list(parts.items()) + [("seen", "seen"), ("unseen", "unseen")]:
        if want == "seen":
            m = np.array([s not in unseen for s, o in keys])
        elif want == "unseen":
            m = np.array([s in unseen for s, o in keys])
        else:
            m = np.array([want is None or s in want for s, o in keys])     # V1 val, or the 90 new
        if not m.any():
            continue
        res = {"objects": int(m.sum())}
        for vs in (ref_k, "dense"):
            for sc in ("ours", "official"):
                if any(rows[h][new_k].off is None or rows[h][vs].off is None for h in heads) and sc == "official":
                    continue
                d = np.mean([(rows[h][new_k].sel(sc, m) - rows[h][vs].sel(sc, m)) for h in heads], 0)
                d = d[~np.isnan(d)]
                res[f"minus_{'ref' if vs == ref_k else 'dense'}_{sc}"] = ci(d)
        for sc in ("ours", "official"):
            if all(rows[h][new_k].off is not None for h in heads) or sc == "ours":
                res[f"jf_{sc}"] = float(np.nanmean([rows[h][new_k].sel(sc, m) for h in heads]))
                res[f"ref_jf_{sc}"] = float(np.nanmean([rows[h][ref_k].sel(sc, m) for h in heads]))
                res[f"dense_jf_{sc}"] = float(np.nanmean([rows[h]["dense"].sel(sc, m) for h in heads]))
        out[part] = res
    fl = float(np.mean([rows[h][new_k].flops for h in heads]))
    fa = float(np.mean([rows[h][ref_k].flops for h in heads]))
    out["flops"], out["ref_flops"] = fl, fa
    if rules:
        lo = out["all"]["minus_ref_ours"][1]
        out["noninferior"], out["superior"] = bool(lo > -3.0 and fl < fa), bool(lo > 0 and fl < fa)
    out["per_head"] = {h: {"minus_ref_ours": ci(rows[h][new_k].ours - rows[h][ref_k].ours),
                           "minus_ref_official": ci((rows[h][new_k].off - rows[h][ref_k].off)[~np.isnan(rows[h][new_k].off - rows[h][ref_k].off)])
                           if rows[h][new_k].off is not None and rows[h][ref_k].off is not None else None,
                           "flops": rows[h][new_k].flops, "ref_flops": rows[h][ref_k].flops,
                           "toolkit": {k: rows[h][k].toolkit() for k in (new_k, ref_k, "dense")}} for h in heads}
    fam = {}                                                            # secondary: per head and per split (ours)
    for h in heads:
        fam[f"head {h}"] = pval(rows[h][new_k].ours - rows[h][ref_k].ours)
    for part, want in (("v1val", V1), ("new90", NEW90)):
        m = np.array([s in want for s, o in keys])
        fam[f"split {part}"] = pval(np.mean([rows[h][new_k].ours[m] - rows[h][ref_k].ours[m] for h in heads], 0))
    out["secondary_p_ours"] = fam
    out["secondary_p_ours_holm"] = holm(fam)
    return out


def claim():
    return _long({h: (f"eval_om_{h}_base", f"eval_om_{h}_r40") for h in HEADS}, S_FINAL, "algm_t0.98", True)


def vitb3():
    return _long({f"r41_b{s}": (f"eval_omb_b{s}_base", f"eval_omb_b{s}_r40") for s in (1, 2, 3)}, SB_FINAL, "algm_t0.95", True)


def vitl():
    return _long({"r34": ("eval_oml_base", "eval_oml_r40")}, SL_FINAL, (Q / "r34_algm.txt").read_text().strip(), False)


def _davis_one(run_all, run_new, new_k):
    base = {k: Rows(run_all, k) for k in sorted({k for (k, s) in load(V / run_all)})}
    new = Rows(run_new, new_k)
    dn = base["dense"]
    res = {"objects": len(dn.keys), "dense": {"ours": float(dn.ours.mean()), "toolkit": dn.toolkit()}, "settings": {}}
    for k, r in list(base.items()) + [(new_k, new)]:
        if k == "dense":
            continue
        e = {"flops": r.flops, "jf_ours": float(r.ours.mean()), "minus_dense_ours": ci(_diffs(r, dn)["ours"]), "toolkit": r.toolkit()}
        d = _diffs(r, dn)
        if "official" in d:
            e["minus_dense_official"] = ci(d["official"])
        if k != new_k:
            d = _diffs(new, r)
            e["redesign_minus_ours"] = ci(d["ours"])
            e["p_ours"] = pval(d["ours"])
            if "official" in d:
                e["redesign_minus_official"] = ci(d["official"])
                e["p_official"] = pval(d["official"])
        res["settings"][k] = e
    for sc in ("ours", "official"):                                     # Holm within the backbone's table
        ps = {k: e[f"p_{sc}"] for k, e in res["settings"].items() if f"p_{sc}" in e}
        for k, q in holm(ps).items():
            res["settings"][k][f"p_{sc}_holm"] = q
    return res


def davis():
    out = {}
    for tag, a, n, k in (("vits", "eval_om_davis_all", "eval_om_davis_r40", S_FINAL),
                         ("vitb", "eval_omb_davis_all", "eval_omb_davis_r40", SB_FINAL),
                         ("vitl", "eval_oml_davis_all", "eval_oml_davis_r40", SL_FINAL)):
        if (V / a).exists() and (V / n).exists():
            out[tag] = _davis_one(a, n, k)
    return out


def all_():
    out = {"check": check(), "davis": davis()}
    for name, f, runs in (("claim", claim, [f"eval_om_{h}_r40" for h in HEADS]),
                          ("vitb3", vitb3, [f"eval_omb_b{s}_r40" for s in (1, 2, 3)]),
                          ("vitl", vitl, ["eval_oml_r40"])):
        if all((V / r).exists() for r in runs):
            out[name] = f()
    (Q / "r40_official.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    r = {"check": check, "claim": claim, "vitb3": vitb3, "vitl": vitl, "davis": davis, "all": all_}[a]()
    if a != "all":
        print(json.dumps(r, indent=1))
