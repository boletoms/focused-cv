"""What the safety nets of the expectation focus map catch (owner question 2026-11-24: change-gated instead of a full
ring; objects that appear for 1 to 5 frames). Reads the logs of `eval_r40diag_t` (`VOS_DIAG=1`; `cov_h`, `cov_hm`: the
share of an object's ground truth inside the frame's final high tier and inside high or middle; `map_cov`: inside the
evaluator's map dilated by one patch, i.e. before the ring, the search region and the surprise promotions). Writes
`reports/stage_b/queue/r40_tier_diag.json`. Per setting, on the frames where the expectation ran (tiers logged):
1. by object speed (patches per frame: slow < 0.5, mid 0.5 to 2, fast > 2): mean coverage and share of object-frames
   below 0.9 for the base map, the high tier and high or middle;
2. the first five frames after an object's ground truth reappears after an absence, and frames where the object is
   visible but the prediction is empty (lost): the same coverages and J."""
import json
from pathlib import Path

import numpy as np

V = Path("/work/stage_b/vos") if Path("/work/stage_b/vos").exists() else Path.home() / "ml_work/stage_b/vos"
Q = Path("reports/stage_b/queue")
SPEED = {"slow": (0, 0.5), "mid": (0.5, 2), "fast": (2, 1e9)}


def stats(cov):
    c = cov[~np.isnan(cov)]
    return {"n": int(c.size), "cov": float(c.mean()) if c.size else None, "below_0.9": float((c < 0.9).mean()) if c.size else None}


def per_object(z):
    t, gpx, ppx = z["t"], z["gt_px"], z["pred_px"]
    gc = z["gt_c"]
    for o in range(gpx.shape[1]):
        sp = np.full(len(t), np.nan)
        sp[1:] = np.linalg.norm(np.diff(gc[:, o], axis=0), axis=1) / 16.0 / np.diff(t).clip(min=1)
        pres = gpx[:, o] > 0
        since = np.full(len(t), 10 ** 6)                         # frames since the object's ground truth reappeared
        c, seen_absent = 10 ** 6, False
        for i in range(len(t)):
            if not pres[i]:
                seen_absent, c = True, 10 ** 6
            elif seen_absent and c == 10 ** 6:
                c = 0
            elif c < 10 ** 6:
                c += 1
            since[i] = c
        yield {"speed": sp, "pres": pres, "lost": pres & (ppx[:, o] == 0), "since": since,
               "base": z["map_cov"][:, o], "high": z["cov_h"][:, o], "hm": z["cov_hm"][:, o], "j": z["j"][:, o]}


def camera_fires(d):
    """Share of patches that fired per frame (`srp_grid`, over the grid's patches) by camera-shift size, and the share of
    frame-wide frames (fired share above 0.5): do false fires come with camera motion the translation model misses?"""
    bins = {"0": [], "1": [], "2-3": [], "4+": []}
    for vd in sorted(p for p in d.iterdir() if p.is_dir()):
        names = json.loads((vd / "settings.json").read_text())
        for i, n in names.items():
            if "exp-" not in n or len(n.split("@")) > 2:
                continue
            z = np.load(vd / f"{i}.npz")
            ran = z["cam_y"] != -999
            mag = np.abs(z["cam_y"]) + np.abs(z["cam_x"])
            share = np.unpackbits(z["srp_grid"], axis=1).sum(1) / max(int(np.unpackbits(z["map_grid"], axis=1).shape[1]), 1)
            for lab, sel in (("0", mag == 0), ("1", mag == 1), ("2-3", (mag >= 2) & (mag <= 3)), ("4+", mag >= 4)):
                bins[lab] += list(share[ran & sel])
    return {k: {"frames": len(v), "mean_fired_share": float(np.mean(v)) if v else None,
                "frame_wide": float(np.mean(np.array(v) > 0.5)) if v else None} for k, v in bins.items()}


def main(out_dir="eval_r40diag_t"):
    d = V / out_dir / "diag"
    acc = {}
    for vd in sorted(p for p in d.iterdir() if p.is_dir()):
        names = json.loads((vd / "settings.json").read_text())
        for i, n in names.items():
            if "exp-" not in n:
                continue
            tag = "@".join(x for x in n.split("@")[2:]) if len(n.split("@")) > 2 else "method"     # schedule suffixes
            for ob in per_object(np.load(vd / f"{i}.npz")):
                ran = ~np.isnan(ob["high"])                       # the expectation ran on this frame
                for lab, sel in [("all", ran & ob["pres"])] + \
                        [(f"speed_{k}", ran & ob["pres"] & (ob["speed"] >= a) & (ob["speed"] < b)) for k, (a, b) in SPEED.items()] + \
                        [("reappear_first5", ran & ob["pres"] & (ob["since"] < 5)), ("lost_frames", ran & ob["lost"])]:
                    a = acc.setdefault(tag, {}).setdefault(lab, {k: [] for k in ("base", "high", "hm", "j")})
                    for k in a:
                        a[k].append(ob[k][sel])
    res = {}
    for tag, labs in acc.items():
        res[tag] = {}
        for lab, a in labs.items():
            c = {k: np.concatenate(v) for k, v in a.items()}
            res[tag][lab] = {"base": stats(c["base"]), "high": stats(c["high"]), "high_or_middle": stats(c["hm"]),
                             "J": float(np.nanmean(c["j"])) if np.isfinite(c["j"]).any() else None}
    for tag, labs in list(res.items()):
        for lab, r in labs.items():
            f = lambda s: "-" if s["cov"] is None else f"{s['cov']:.3f} ({100 * s['below_0.9']:.0f}% < 0.9)"
            print(tag, lab, r["base"]["n"], "base", f(r["base"]), "| high", f(r["high"]), "| high+mid", f(r["high_or_middle"]), "| J", None if r["J"] is None else round(r["J"], 3))
    res["camera_fires"] = camera_fires(d)
    print("camera_fires", json.dumps(res["camera_fires"]))
    (Q / "r40_tier_diag.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
