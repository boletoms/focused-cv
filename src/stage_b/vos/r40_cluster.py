"""Video-level sensitivity of the long-video comparisons (audits 2026-11-23). Objects of one video share frames and
tracker state; here whole videos are resampled (10,000 draws, seed 0, within the split strata LVOS V1 val and the 90
new V2 videos). Two estimands: the object-weighted mean (the pre-specified one) and the mean that weights videos
equally. Rows: the claim (ViT-S, the reported timed rows `eval_r40o_*` against ALGM 0.98 and dense), ViT-B three heads
(`eval_omb_*`), ViT-L one head (`eval_oml_<run>`, default the single-mode rerun `r40s`). Writes `r40_video_cluster.json`.
Usage: python -m src.stage_b.vos.r40_cluster [vitl run suffix]"""
import json
import sys

import numpy as np

from .r28_compare import V, load
from .r40_compare import HEADS, Q, S_FINAL, SB_FINAL, SL_FINAL

NEW90 = set((Q / "v2valid_new.txt").read_text().split())
jf = lambda by, k, s: 50 * (np.array(by[(k, s)]["J"]) + np.array(by[(k, s)]["F"]))


def boot(pv, draws=10000, seed=0):
    names = sorted(pv)
    sums = np.array([pv[s].sum() for s in names]); cnt = np.array([len(pv[s]) for s in names]); means = sums / cnt
    strata = [np.array([i for i, s in enumerate(names) if (s in NEW90) == g]) for g in (False, True)]
    strata = [ix for ix in strata if len(ix)]
    rng = np.random.default_rng(seed)
    ow, ev = [], []
    for _ in range(draws):
        pick = np.concatenate([rng.choice(ix, len(ix)) for ix in strata])
        ow.append(sums[pick].sum() / cnt[pick].sum()); ev.append(means[pick].mean())
    q = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    return {"videos": len(names), "objects": int(cnt.sum()), "object_weighted": [float(sums.sum() / cnt.sum())] + q(ow),
            "equal_video": [float(means.mean())] + q(ev)}


def per_video(heads_rows, k_new, k_ref):
    return {s: np.mean([jf(n, k_new, s) - jf(b, k_ref, s) for n, b, _ in heads_rows], 0) for s in heads_rows[0][2]}


def claim_rows():
    hr = []
    for h, (r1, r2) in HEADS.items():
        n1, n2, b1, b2 = (load(V / f"eval_r40o_{h}_v1val"), load(V / f"eval_r40o_{h}_v2new"), load(V / r1), load(V / r2))
        s1 = sorted(s for (k, s) in n1 if k == S_FINAL); s2 = sorted(s for (k, s) in n2 if k == S_FINAL and s in NEW90)
        hr.append(((n1, b1, s1), (n2, b2, s2)))
    return hr


def main(vitl_run="r40s"):
    out = {}
    hr = claim_rows()
    for ref in ("algm_t0.98", "dense"):
        pv = {}
        for part in (0, 1):
            pv |= per_video([h[part] for h in hr], S_FINAL, ref)
        out[f"vits_minus_{ref}"] = boot(pv)
        out[f"vits_new90_minus_{ref}"] = boot({s: v for s, v in pv.items() if s in NEW90})
    full = lambda pn, pr, heads, kn: [(load(V / pn.format(h)), load(V / pr.format(h)),
                                       sorted(s for (k, s) in load(V / pn.format(h)) if k == kn)) for h in heads]
    hb = full("eval_omb_b{}_r40", "eval_omb_b{}_base", (1, 2, 3), SB_FINAL)
    hl = full(f"eval_oml_{vitl_run}" + "{}", "eval_oml_base{}", ("",), SL_FINAL)
    ref_l = (Q / "r34_algm.txt").read_text().strip()
    for tag, rows, kn, ref in (("vitb3", hb, SB_FINAL, "algm_t0.95"), ("vitl", hl, SL_FINAL, ref_l)):
        for r in (ref, "dense"):
            pv = per_video(rows, kn, r)
            out[f"{tag}_minus_{'ref' if r != 'dense' else 'dense'}"] = boot(pv)
            out[f"{tag}_new90_minus_{'ref' if r != 'dense' else 'dense'}"] = boot({s: v for s, v in pv.items() if s in NEW90})
    out["vitl_run"] = f"eval_oml_{vitl_run}"
    (Q / "r40_video_cluster.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, v["videos"], v["objects"], [round(x, 2) for x in v["object_weighted"]], [round(x, 2) for x in v["equal_video"]])
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
