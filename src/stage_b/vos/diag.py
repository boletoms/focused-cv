"""Long-video diagnostics (owner 2026-11-20, after the surprise review: "plan and prepare all the diagnostics needed to
pinpoint issues and their exact causes"; plan `reports/stage_b/queue/DIAGNOSTICS_PLAN.md`). Per-frame logging inside
`vos/eval_lvos.py`, switched on by the environment variable `VOS_DIAG=1`; it only reads tensors the evaluation already
has and never changes them, so every scored number stays identical (checked against earlier runs, plan section 4).

One file per video and setting, `<out>/diag/<video>/<setting index>.npz` (`<out>/diag/<video>/settings.json` maps
indices to setting names). Frames are those the evaluation processes (from the first annotated object on); objects are
the video's objects in id order; per object, values are logged on frames after its first appearance (as scoring) and
are NaN / 0 before. At the evaluation size (shortest side 480; patch grid of the encoder: padded symmetrically to
multiples of 16 as Cutie):
- per frame: `t` frame index, `flops`, `fallback` (the empty-map method ran), `dense` (no reduction ran: dense
  frame, I-frame or trigger), `empty` (empty map), `oracle` (the oracle fired), `mem` (Cutie wrote working memory),
  `map_grid` (packed bits: patches whose map coverage reaches the methods' threshold 0.01, before the one-patch
  margin), `srp_grid` (packed bits: patches the surprise detector protected; zeros without it);
- per frame and object: `j`, `f` (exactly the scored values), `gt_px`, `pred_px`, `inter_px` (pixels at the evaluation
  size), `gt_c`, `pred_c` (centroids (y, x) in pixels, NaN if empty), `map_cov` (share of the object's ground-truth
  pixels in protected patches: the map's patches dilated by one patch on the grid, as `dil1`, before size-class
  padding), `prob_gt` (mean object probability on its ground-truth pixels), `prob_out` (the object's highest
  patch-mean probability outside the protected patches), `prob_grid`
  (uint8, object probability averaged per patch, x255);
- `gt_grid` (uint8, per object ground-truth coverage per patch, x255), written only for the `dense` setting (the
  ground truth is the same for every setting)."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PATCH, THETA = 16, 0.01                                     # the methods' patch and interest threshold


def to_grid(x):
    """(h, w) float tensor -> (gh, gw) patch means on the encoder's padded grid (symmetric zero padding)."""
    h, w = x.shape
    ph, pw = (-h) % PATCH, (-w) % PATCH
    lh, lw = ph // 2, pw // 2
    return F.avg_pool2d(F.pad(x.float()[None, None], (lw, pw - lw, lh, ph - lh)), PATCH)[0, 0]


def _centroid(m):
    if not bool(m.any()):
        return (float("nan"), float("nan"))
    ys, xs = torch.nonzero(m, as_tuple=True)
    return (float(ys.float().mean()), float(xs.float().mean()))


class DiagLog:
    def __init__(self, out, seq, spec, index, objects, with_gt):
        self.path = Path(out) / "diag" / seq
        self.path.mkdir(parents=True, exist_ok=True)
        self.spec, self.index, self.objects, self.with_gt = spec, index, list(objects), with_gt
        self.rows = {k: [] for k in ("t", "flops", "fallback", "dense", "empty", "oracle", "mem", "map_grid", "srp_grid", "tier_h", "tier_m", "cam_y", "cam_x")}
        self.obj = {k: [] for k in ("j", "f", "gt_px", "pred_px", "inter_px", "gt_c", "pred_c", "map_cov", "prob_gt",
                                    "prob_out", "prob_grid", "gt_grid", "cov_h", "cov_hm")}

    def frame(self, t, first, interest_mask, prob, small, tmp2obj, gt_eval, flops, fallback, dense, empty,
              oracle, mem, srp_grid, jf_now, tiers=None, cam=None):
        """interest_mask (h, w) bool; prob (K + 1, h, w); small
        (h, w) object ids; tmp2obj {prob channel: object id}; gt_eval (h, w) object ids at the evaluation size or
        None; jf_now {object: (j, f)} scored on this frame; srp_grid (gh, gw) bool or None; tiers (high, middle) (gh, gw)
        bool grids of the expectation focus map's final tiers, or None (cov_h, cov_hm: the share of the object's ground
        truth inside the high tier and inside high or middle; NaN without tiers)."""
        mg = to_grid(interest_mask) >= THETA
        prot = F.max_pool2d(mg.float()[None, None], 3, 1, 1)[0, 0] > 0           # dil1 on the grid
        outside = ~prot
        self.rows["t"].append(t)
        self.rows["flops"].append(float(flops))
        for k, v in (("fallback", fallback), ("dense", dense), ("empty", empty), ("oracle", oracle), ("mem", mem)):
            self.rows[k].append(bool(v))
        self.rows["map_grid"].append(np.packbits(mg.cpu().numpy().ravel()))
        sg = srp_grid if srp_grid is not None else torch.zeros_like(mg)
        self.rows["srp_grid"].append(np.packbits(sg.cpu().numpy().ravel()))
        th, tm = tiers if tiers is not None else (torch.zeros_like(mg), torch.zeros_like(mg))
        self.rows["tier_h"].append(np.packbits(th.cpu().numpy().ravel()))
        self.rows["tier_m"].append(np.packbits(tm.cpu().numpy().ravel()))
        self.rows["cam_y"].append(cam[0] if cam is not None else -999)       # camera shift in patches (-999: none)
        self.rows["cam_x"].append(cam[1] if cam is not None else -999)
        chan = {oid: i for i, oid in tmp2obj.items()}
        vals = {k: [] for k in self.obj}
        for o in self.objects:
            scored = t > first[o]
            p = prob[chan[o]] if o in chan else torch.zeros_like(prob[0])
            pg = to_grid(p)
            pred = small == o
            gt = (gt_eval == o) if gt_eval is not None else torch.zeros_like(pred)
            n_gt = int(gt.sum())
            j, f = jf_now.get(o, (float("nan"), float("nan"))) if scored else (float("nan"), float("nan"))
            vals["j"].append(j)
            vals["f"].append(f)
            vals["gt_px"].append(n_gt)
            vals["pred_px"].append(int(pred.sum()))
            vals["inter_px"].append(int((pred & gt).sum()))
            vals["gt_c"].append(_centroid(gt))
            vals["pred_c"].append(_centroid(pred))
            gg = to_grid(gt.float())
            vals["map_cov"].append(float((gg * prot).sum() / gg.sum()) if n_gt else float("nan"))
            vals["cov_h"].append(float((gg * th).sum() / gg.sum()) if n_gt and tiers is not None else float("nan"))
            vals["cov_hm"].append(float((gg * (th | tm)).sum() / gg.sum()) if n_gt and tiers is not None else float("nan"))
            vals["prob_gt"].append(float(p[gt].mean()) if n_gt else float("nan"))
            vals["prob_out"].append(float(pg[outside].max()) if bool(outside.any()) else 0.0)
            vals["prob_grid"].append((pg.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())
            vals["gt_grid"].append((gg * 255).round().to(torch.uint8).cpu().numpy()
                                   if self.with_gt else None)
        for k, v in vals.items():
            self.obj[k].append(v)

    def save(self):
        out = {k: np.array(v) for k, v in self.rows.items() if k not in ("map_grid", "srp_grid", "tier_h", "tier_m")}
        out["map_grid"] = np.stack(self.rows["map_grid"]) if self.rows["map_grid"] else np.zeros((0, 0), np.uint8)
        out["srp_grid"] = np.stack(self.rows["srp_grid"]) if self.rows["srp_grid"] else np.zeros((0, 0), np.uint8)
        for k in ("tier_h", "tier_m"):
            out[k] = np.stack(self.rows[k]) if self.rows[k] else np.zeros((0, 0), np.uint8)
        for k, v in self.obj.items():
            if k == "gt_grid" and not self.with_gt:
                continue
            out[k] = np.array(v, dtype=np.uint8 if k.endswith("grid") else np.int64 if k.endswith("_px")
                              else np.float64 if k in ("j", "f") else np.float32)       # j, f exact (check 1)
        out["objects"] = np.array(self.objects)
        np.savez_compressed(self.path / f"{self.index}.npz", **out)
        idx = self.path / "settings.json"
        names = json.loads(idx.read_text()) if idx.exists() else {}
        names[str(self.index)] = self.spec
        idx.write_text(json.dumps(names, indent=1) + "\n")
