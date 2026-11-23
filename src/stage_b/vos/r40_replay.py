"""R-38 (reported, no rule): the expectation logic of the frame runner against the current path on long videos.
Each video's ground-truth masks are replayed as the 'prediction' (objects from their first annotated frame, the last
annotation held between annotated frames) through the current path (`objects_from_predictions`, the motion map, the
wrappers via `tiered_graph.plan`, as `eval_lvos` without VOS_GRAPH) and through the frame runner's steps before the
encoder (`FrameRunner.run` with the encoder stubbed, as `eval_lvos` with VOS_GRAPH=2). CPU, fp32, so the comparison
is of the logic, not of rounding. Per video: frames, fallback frames, fallback decisions that differ, frames whose
tier maps differ, middle-tier shares of both, and the first differing frames with both paths' object states.
Usage: python -m src.stage_b.vos.r40_replay [workers] [out json] [videos...] (default: LVOS V1 val and the 90 new V2
valid videos, `reports/stage_b/queue/r40_replay_gt.json`). R40_REPLAY_SETTING=vitb replays the ViT-B setting (margin 2,
floor 128, ALGM 0.95; default out `r40_replay_gt_vitb.json`)."""
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, normalize, resize, to_tensor

from .eval_lvos import LVOS, LVOS2, SIZE
from .r40_compare import Q, S_FINAL, SB_FINAL

VITB = os.environ.get("R40_REPLAY_SETTING") == "vitb"
SETTING = SB_FINAL if VITB else S_FINAL

MAX_REPORT = 3
_model = None


def _videos():
    v1 = sorted(p.name for p in (LVOS / "valid" / "JPEGImages").iterdir())
    new90 = sorted((Q / "v2valid_new.txt").read_text().split())
    return [(LVOS / "valid", s) for s in v1] + [(LVOS2 / "valid", s) for s in new90]


def replay(root, seq, preds=None, keep=()):
    """`preds` {t: (h, w) uint8 ids}: a run's own predictions at the evaluation size (`VOS_SAVE_SMALL=1`) replace the
    replayed ground truth, so the tiers are the run's own (figures); `keep`: frames whose tier maps (reference path,
    patch grid) are returned under "tiers"."""
    return _replay(root, seq, preds, set(keep))


def _replay(root, seq, preds, keep):
    from src.stage_a.model import load_backbone
    from src.stage_b.expect import ExpectMap, ExpectState, objects_from_predictions
    from src.stage_b.frame_runner import FrameRunner
    from src.stage_b.model import EffModel
    from src.stage_b.p2_davis import shift_by_velocity
    from src.stage_b.tiered_graph import plan
    from src.stage_b.timing import from_name
    global _model
    torch.set_grad_enabled(False)
    if _model is None:
        _model = EffModel(load_backbone("cpu"))
    inner_name, rest = SETTING.split("~")
    fb_name = rest.split("@")[0]
    frames = sorted((root / "JPEGImages" / seq).glob("*.jpg"))
    annos = {p.stem: p for p in (root / "Annotations" / seq).glob("*.png")}
    gt = lambda t: np.array(Image.open(annos[frames[t].stem])) if frames[t].stem in annos else None
    first = {}
    for t in range(len(frames)):
        g = gt(t)
        if g is not None:
            for o in np.unique(g).tolist():
                if o and o not in first:
                    first[o] = t
    t0 = time.time()
    fr = prev = idx_prev = idx_prev2 = held = None
    cap, started, reports, tiers = {}, False, [], {}
    n = {"frames": 0, "fallback": 0, "fallback_differ": 0, "main": 0, "tiers_differ": 0}
    mid = [0.0, 0.0]
    for t, fp in enumerate(frames):
        new = [o for o, t0_ in first.items() if t0_ == t]
        if not started and not new:
            continue
        started = True
        img = resize(to_tensor(Image.open(fp).convert("RGB")), SIZE, InterpolationMode.BILINEAR, antialias=True)
        h, w = img.shape[-2:]
        H, W = -(-h // 16) * 16, -(-w // 16) * 16
        lh, lw = (H - h) // 2, (W - w) // 2
        x = F.pad(normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), (lw, W - w - lw, lh, H - h - lh))[None]
        if fr is None:
            fr = FrameRunner(_model, (H, W), SETTING, fast=False, dtype=torch.float32, graphs=False)
            fr.runner.launch_attn0 = lambda p: None

            def enc(pixels, spec, tiers, ops, st, caps):
                cap["tiers"] = tiers.clone()
                return None, st
            fr.runner.encode, fr.runner.overflow = enc, 0
            fr.reset()
            est, ost = ExpectState(), {}
            main = ExpectMap(from_name(inner_name, H // 16, 12), est, fr.cfg)
            fbm = ExpectMap(from_name(fb_name, H // 16, 12), est, fr.cfg, passive=True)
        g = gt(t)
        if g is not None:
            held = torch.from_numpy(g.astype(np.int64))
        if new:
            gn = torch.where(torch.isin(held, torch.tensor(new)), held, 0)
            mask = resize(gn[None], [h, w], InterpolationMode.NEAREST)[0]
            prev = (mask > 0) if prev is None else (prev | (mask > 0))
        b = prev                                                         # current path: the map with the motion
        if idx_prev is not None and idx_prev2 is not None:
            b = b | (shift_by_velocity(idx_prev, idx_prev2) > 0)
        ids = [o for o, t0_ in first.items() if t0_ < t]
        objs, all_lost = objects_from_predictions(idx_prev, idx_prev2, ids, ost, fr.cfg.motion) if idx_prev is not None else ([], False)
        est.objects = objs
        all_lost = all_lost and not new
        use_fb = not bool(b.any()) or all_lost
        m = torch.zeros(H, W)
        m[lh:lh + h, lw:lw + w] = b.float()
        st = plan(fbm if use_fb else main, x, F.avg_pool2d(m[None, None], 16)[0, 0].flatten()[None])
        last = dict(est.last)
        cap.clear()
        _, _, _, fb_dev = fr.run(x, ids, idx_prev, idx_prev2, prev, bool(new))   # runner: the map before the motion
        n["frames"] += 1
        n["fallback"] += use_fb
        if t in keep:                                                    # the reference path's tiers (None: fallback)
            tiers[t] = (None if use_fb else st.interest[0].reshape(H // 16, W // 16).numpy().copy(), (lh, lw, h, w))
        if fb_dev != use_fb:
            n["fallback_differ"] += 1
            if len(reports) < MAX_REPORT:
                reports.append({"t": t, "fallback_current": use_fb, "fallback_runner": fb_dev})
        elif not use_fb and not last.get("first"):
            n["main"] += 1
            te, td = st.interest[0], cap["tiers"]
            mid[0] += float(((te > 0) & (te < 1)).float().mean())
            mid[1] += float(((td > 0) & (td < 1)).float().mean())
            if not torch.equal(te, td):
                n["tiers_differ"] += 1
                if len(reports) < MAX_REPORT:
                    s = fr.dobj.s
                    reports.append({"t": t, "high_differ": int(((te >= 1) != (td >= 1)).sum()),
                                    "mid_differ": int((((te > 0) & (te < 1)) != ((td > 0) & (td < 1))).sum()),
                                    "current": [{"id": o["id"], "lost": bool(o["lost"]), "n": int(o["n"]), "speed": float(o["speed"]),
                                                 "v": [float(v) for v in o["v"]]} for o in objs],
                                    "runner": [{"id": o, "n": int(s["n"][k]), "v": [float(v) for v in s["v"][k]],
                                                "hasM": bool(s["hasM"][k])} for k, o in enumerate(fr.dobj.ids)]})
        pred = torch.where(torch.isin(held, torch.tensor([o for o, t0_ in first.items() if t0_ <= t])), held, 0)
        small = resize(pred[None], [h, w], InterpolationMode.NEAREST)[0] if preds is None else torch.from_numpy(preds[t].astype(np.int64))
        prev = small > 0
        idx_prev2, idx_prev = idx_prev, small
    return {"video": seq, "split": root.parent.name, "objects": len(first), **n,
            "mid_current": mid[0] / max(1, n["main"]), "mid_runner": mid[1] / max(1, n["main"]),
            "seconds": round(time.time() - t0, 1), "first_differences": reports, **({"tiers": tiers} if keep else {})}


def _job(a):
    torch.set_num_threads(4)
    return replay(*a)


def main(workers="6", out=str(Q / ("r40_replay_gt_vitb.json" if VITB else "r40_replay_gt.json")), *videos):
    vids = _videos()
    if videos:
        vids = [v for v in vids if v[1] in set(videos)]
    with get_context("spawn").Pool(int(workers)) as pool:
        rows = []
        for r in pool.imap_unordered(_job, vids):
            rows.append(r)
            print(f"{r['video']} {r['split']}: {r['frames']} frames, fallback {r['fallback']}, fallback differ "
                  f"{r['fallback_differ']}, tiers differ {r['tiers_differ']} of {r['main']}, mid {r['mid_current']:.4f} / "
                  f"{r['mid_runner']:.4f} ({r['seconds']} s)", flush=True)
    rows.sort(key=lambda r: (r["split"], r["video"]))
    tot = {k: sum(r[k] for r in rows) for k in ("frames", "fallback", "fallback_differ", "main", "tiers_differ")}
    summary = {"videos": len(rows), **tot, "videos_with_differences": sum(bool(r["fallback_differ"] or r["tiers_differ"]) for r in rows)}
    print(json.dumps(summary))
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
