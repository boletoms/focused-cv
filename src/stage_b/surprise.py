"""R28 surprise: detect unexpected change in low-focus areas, so those regions are reprocessed at full resolution.

Owner's definition (2026-11-20): a surprise is any low-focus area that has changed beyond a threshold, i.e.
something unexpected happened there (a new object entering is one case; an object starting to move, disappearing, a
local appearance change are others); the region then needs full processing. Expected changes must not fire: camera
motion, noise, ongoing dynamic texture (water, foliage), the tracked target's own motion.

Detector (per video, stateful; works on the patch grid of one frame):
- Features: full-resolution per-patch features available before any merging (`feat`): `x0` the block-0 residual after
  attention (where every method's first merge happens), `emb` the patch embeddings, or `pix` 4x4-pooled RGB (baseline);
  centred on the frame's mean patch, layer-normalised, then unit length (`unit`). `pix8` (fine grid): the same on 8-pixel
  cells (four per patch, 2x2-pixel blocks); the detector runs on the cell grid (radius, min_size, global share in
  cells) and its detections are pooled to patches (a patch with any detected cell), then dilated by `dilate` patches,
  so a change covering a fraction of a patch is not diluted by the rest of the patch.
- Prediction error: e(i) = 1 - max cosine between patch i and the patches of the last `memory` frames within `radius`
  patches of i (content may move by up to `radius` patches per frame: camera and object motion are expected); `sub`
  also searches half-patch positions (memory interpolated), for motion by a fraction of a patch; `coh` > 0 predicts a
  patch only at the median motion of its well-matched neighbours within `coh` patches (give or take one step), so a
  change that only an unrelated look-alike nearby would explain still counts (`coherent_match`).
- Expectation: per patch, a running mean and variance of e (rate `ema`; variance starts at `prior`^2, so early frames
  are not trusted), carried along the best match in the previous frame where that match explains the patch (cosine
  >= 1 - tau; the statistics move with the content) and kept in place elsewhere; z(i) = (e - mean) / sqrt(var +
  floor^2). A global change leaves the statistics untouched.
- Candidate: low interest now and at the previous frame (a patch the map just left shows the background its target
  uncovered: expected), statistics warmed up (`warmup` observations), e > `tau` and z > `z` (z <= 0: no expectation
  test), not in the strip of new content that the global motion brings in at the border.
- Region: connected components (4-neighbour) of at least `min_size` patches, dilated by `dilate` patches; kept for
  `hold` frames after the last detection. If candidates cover more than `global_frac` of the out-of-focus patches, the
  change is global (scene cut, fast pan, lighting): the whole frame is reprocessed and the statistics are not updated.
`step` returns the patches to reprocess (h, w) and whether the change is global. FLOPs of the detector are reported."""
import math
from collections import deque

import torch
import torch.nn.functional as F

from src.stage_a.model import N_PREFIX

from .baselines import _fastsv_labels


def unit(f):
    """(h, w, D) -> features centred on the frame's mean patch, layer-normalised, unit length (float32). Centring
    matters: DINOv3's block-0 residual carries one channel about 10x larger than the rest on every patch, so without it
    any two patches have cosine about 0.9 and nothing looks changed."""
    f = f.float()
    f = f - f.reshape(-1, f.shape[-1]).mean(0)
    return F.normalize(F.layer_norm(f, (f.shape[-1],)), dim=-1)


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)            # ImageNet, as Cutie and the ViT


def pixel_features(pixels, cell=16):
    """Normalised frame (3, H, W), H and W multiples of 16 (padded with 0 before normalising, as Cutie) -> (H/cell,
    W/cell, 48): RGB in [0, 1] average-pooled over (cell/4)-pixel blocks, the 4 x 4 blocks of each cell concatenated
    (`pix`: cell 16, the patch; `pix8`: cell 8, four cells per patch)."""
    x = pixels.float() * torch.tensor(STD, device=pixels.device)[:, None, None] + \
        torch.tensor(MEAN, device=pixels.device)[:, None, None]
    gh, gw = x.shape[1] // cell, x.shape[2] // cell
    pix = F.avg_pool2d(x[None], cell // 4)[0]
    return pix.reshape(3, gh, 4, gw, 4).permute(1, 3, 0, 2, 4).reshape(gh, gw, 48)


def correlations(f, mem, radius):
    """C (O, K, h, w): cosine between patch i of f (h, w, D) and patch i + o of memory frame k (mem (K, h, w, D)), for
    the integer offsets o in [-radius, radius]^2 (row-major); -2 where i + o is outside the frame."""
    h, w, _ = f.shape
    m = mem.permute(0, 3, 1, 2)
    pad = F.pad(m, (radius,) * 4)
    valid = F.pad(torch.ones(1, 1, h, w, device=f.device), (radius,) * 4)[0, 0] > 0
    ft = f.permute(2, 0, 1)
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys, xs = radius + dy, radius + dx
            c = (pad[:, :, ys:ys + h, xs:xs + w] * ft[None]).sum(1)
            out.append(c.masked_fill(~valid[ys:ys + h, xs:xs + w][None], -2.0))
    return torch.stack(out)


def adjacent_dots(mem):
    """Dot products between neighbouring memory patches: right, down, diagonal, anti-diagonal (K, ., .)."""
    return ((mem[:, :, :-1] * mem[:, :, 1:]).sum(-1), (mem[:, :-1] * mem[:, 1:]).sum(-1),
            (mem[:, :-1, :-1] * mem[:, 1:, 1:]).sum(-1), (mem[:, :-1, 1:] * mem[:, 1:, :-1]).sum(-1))


def select_radius(C, radius_from, radius):
    """The offsets of C (computed for radius_from) within `radius`."""
    keep = [n for n, (dy, dx) in enumerate((dy, dx) for dy in range(-radius_from, radius_from + 1)
                                           for dx in range(-radius_from, radius_from + 1))
            if abs(dy) <= radius and abs(dx) <= radius]
    return C[keep]


def fine(C, dots, radius, sub):
    """Search positions: integer offsets, or (sub) half-patch steps. A half-patch position is the bilinear mean of 2 or 4
    neighbouring memory patches, renormalised, so its cosine is exact from C and the neighbours' dot products:
    sum of their cosines / norm of their sum. Returns (offsets (O, 2) in steps, steps per patch, cosines (O, K, h, w))."""
    O, K, h, w = C.shape
    ints = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    if not sub:
        return torch.tensor(ints, device=C.device, dtype=torch.float32), 1, C
    idx = {o: n for n, o in enumerate(ints)}
    rr, dd, gg, aa = dots

    def at(g, p, q):                                  # g (K, H', W') read at (y + p, x + q); nan outside
        out = torch.full((K, h, w), float("nan"), device=C.device)
        y0, y1, x0, x1 = max(0, -p), min(h, g.shape[1] - p), max(0, -q), min(w, g.shape[2] - q)
        if y0 < y1 and x0 < x1:
            out[:, y0:y1, x0:x1] = g[:, y0 + p:y1 + p, x0 + q:x1 + q]
        return out
    offs, res = [], []
    for a in range(-2 * radius, 2 * radius + 1):
        for b in range(-2 * radius, 2 * radius + 1):
            p, al = divmod(a, 2)
            q, be = divmod(b, 2)
            S = [(p + i, q + j) for i in range(al + 1) for j in range(be + 1)]
            cs = torch.stack([C[idx[o]] for o in S])
            num, bad = cs.sum(0), (cs < -1.5).any(0)
            if len(S) == 1:
                cos = num
            else:
                if al and be:
                    n2 = 4 + 2 * (at(rr, p, q) + at(rr, p + 1, q) + at(dd, p, q) + at(dd, p, q + 1) + at(gg, p, q)
                                  + at(aa, p, q))
                elif al:
                    n2 = 2 + 2 * at(dd, p, q)
                else:
                    n2 = 2 + 2 * at(rr, p, q)
                bad = bad | torch.isnan(n2)
                cos = num / torch.sqrt(torch.nan_to_num(n2, nan=1.0).clamp_min(1e-12))
            res.append(cos.masked_fill(bad, -2.0))
            offs.append((a, b))
    return torch.tensor(offs, device=C.device, dtype=torch.float32), 2, torch.stack(res)


def _windowed(offs, s, Cf):
    best = Cf.max(1).values.max(0).values
    best_last, arg = Cf[:, -1].max(0)
    return best, (offs[arg] / s).round().long(), best_last


def _coherent(offs, s, Cf, coh):
    h, w = Cf.shape[2:]
    c_all, c_last = Cf.max(1).values, Cf[:, -1]
    best_last, arg = c_last.max(0)
    own = offs[arg].permute(2, 0, 1)                                               # (2, h, w)
    ok = best_last >= best_last.median()
    vals = torch.where(ok[None], own, torch.full_like(own, float("nan")))
    q = 2 * coh + 1
    nb = F.unfold(F.pad(vals[None], (coh,) * 4, value=float("nan")), q)[0].view(2, q * q, h, w)
    med = torch.nanmedian(nb, dim=1).values
    glob = torch.nanmedian(vals.reshape(2, -1), dim=1).values if ok.any() else torch.zeros(2, device=Cf.device)
    med = torch.where(torch.isnan(med), glob[:, None, None].expand_as(med), med).round()
    near = ((offs[:, :, None, None] - med[None]).abs() <= 1).all(1)
    best = c_all.masked_fill(~near, -2.0).max(0).values
    return best, (med.permute(1, 2, 0) / s).round().long(), best_last


def windowed_match(f, mem, radius, sub=False):
    """Best cosine between each patch of f (h, w, D) and the patches of every frame in mem (K, h, w, D) within `radius`
    patches. sub: also at half-patch positions (memory features interpolated bilinearly between patch centres and
    renormalised), so motion by a fraction of a patch is matched too. Returns (best cosine over all frames, offset
    (dy, dx) in whole patches of the best match in mem[-1] (the latest), best cosine in mem[-1]); positions outside the
    frame never match."""
    return _windowed(*fine(correlations(f, mem, radius), adjacent_dots(mem) if sub else None, radius, sub))


def coherent_match(f, mem, radius, sub, coh):
    """As windowed_match, but a patch is predicted only from the motion of its surroundings: the median offset of the
    well-matched patches (latest frame; the better-matched half of the frame) within `coh` patches, give or take one
    search step, over all memory frames. A change that only some unrelated look-alike nearby explains (an object
    appearing among similar ones) stays a change. Returns (coherent best cosine, median offset in whole patches, best
    cosine in mem[-1])."""
    return _coherent(*fine(correlations(f, mem, radius), adjacent_dots(mem) if sub else None, radius, sub), coh)


def components(mask, min_size):
    """Keep the 4-connected components of `mask` (h, w) with at least `min_size` patches."""
    if min_size <= 1 or not mask.any():
        return mask
    if min_size == 2:                                 # exact: in a component of 2 or more = has a 4-neighbour in it
        k = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]], device=mask.device)[None, None]
        return mask & (F.conv2d(mask.float()[None, None], k, padding=1)[0, 0] > 0)
    h, w = mask.shape
    m = mask.flatten()[None]
    right = (m & torch.cat([m[:, 1:], torch.zeros_like(m[:, :1])], 1))
    right = right & (torch.arange(h * w, device=mask.device) % w != w - 1)[None]
    bottom = m & torch.cat([m[:, w:], torch.zeros_like(m[:, :w])], 1)
    lab = _fastsv_labels(right, bottom, w)[0]
    size = torch.bincount(lab[m[0]], minlength=h * w)
    return (mask.flatten() & (size[lab] >= min_size)).view(h, w)


def grow(mask, d):
    return mask if d <= 0 else F.max_pool2d(mask.float()[None, None], 2 * d + 1, 1, d)[0, 0] > 0


class SurpriseDetector:
    def __init__(self, feat="x0", radius=2, memory=4, tau=0.3, z=3.0, ema=0.1, floor=0.02, prior=0.1, warmup=3,
                 min_size=1, dilate=1, hold=5, global_frac=0.3, border=True, store=True, sub=False, coh=0):
        self.feat, self.radius, self.memory, self.tau, self.z, self.sub, self.coh = feat, radius, memory, tau, z, sub, coh
        self.ema, self.floor, self.prior, self.warmup = ema, floor, prior, warmup
        self.min_size, self.dilate = min_size, dilate
        self.hold, self.global_frac, self.border, self.store = hold, global_frac, border, store
        self.reset()

    @property
    def name(self):
        return (f"srp_{self.feat}_r{self.radius}{'s' if self.sub else ''}_k{self.memory}_t{self.tau:g}_z{self.z:g}_m{self.min_size}"
                f"_h{self.hold}_g{self.global_frac:g}" + (f"_c{self.coh}" if self.coh else ""))

    def reset(self):
        self.mem, self.mean, self.var, self.count, self.timer, self.free_prev = deque(maxlen=self.memory), None, None, None, None, None
        self.last = {}

    def match(self, f, mem, C=None, dots=None):
        """Matching result for step(); C (correlations at this radius) and dots may be precomputed and shared."""
        C = correlations(f, mem, self.radius) if C is None else C
        dots = (adjacent_dots(mem) if dots is None else dots) if self.sub else None
        fn = fine(C, dots, self.radius, self.sub)
        return _coherent(*fn, self.coh) if self.coh else _windowed(*fn)

    def flops(self, h, w, D):
        """Matching FLOPs for an (h, w) patch grid (pix8: four cells per patch)."""
        n = len(self.mem) if self.store else self.memory
        cells = 4 if self.feat == "pix8" else 1
        return 2 * ((4 if self.sub else 2) * self.radius + 1) ** 2 * n * h * w * cells * D

    def step(self, feat, free, match=None):
        """feat (h, w, D) features of this frame; free (h, w) bool, the out-of-focus patches; match: optional
        precomputed `windowed_match(unit(feat), memory, radius)` (detectors sharing features, radius and memory).
        Returns (reprocess (h, w) bool, global change bool). `pix8`: feat is on the cell grid (2h, 2w, D); free and
        the result are on the patch grid."""
        if self.feat != "pix8":
            return self._step(feat, free, match)
        free_c = free.repeat_interleave(2, 0).repeat_interleave(2, 1)
        dilate, self.dilate = self.dilate, 0
        try:
            out_c, glob = self._step(feat, free_c, match)
        finally:
            self.dilate = dilate
        pool = lambda m: F.max_pool2d(m.float()[None, None], 2)[0, 0] > 0
        det_c = self.last["det"]
        det = torch.ones_like(free) if glob else grow(pool(det_c), dilate)
        out = torch.ones_like(free) if glob else grow(pool(out_c), dilate)
        self.last.update({"det": det, "out": out, "det_cells": det_c})
        return out, glob

    def _step(self, feat, free, match=None):
        f = feat if match is not None else unit(feat)
        h, w, _ = f.shape
        dev = f.device
        out, glob = torch.zeros(h, w, dtype=torch.bool, device=dev), False
        cand = torch.zeros_like(out)
        prev = match is not None if not self.store else bool(self.mem) and self.mem[-1].shape == f.shape
        if prev and self.mean is not None and self.mean.shape == f.shape[:2]:
            best, off, best_last = match if match is not None else self.match(f, torch.stack(list(self.mem)))
            e = (1 - best).clamp(0, 2)
            good = best_last >= 1 - self.tau             # statistics follow the content where the latest frame explains
            gy, gx = torch.arange(h, device=dev)[:, None].expand(h, w), torch.arange(w, device=dev)[None, :].expand(h, w)
            ys = torch.where(good, (gy + off[..., 0]).clamp(0, h - 1), gy)            # it; elsewhere they stay put
            xs = torch.where(good, (gx + off[..., 1]).clamp(0, w - 1), gx)
            mean, var, count = self.mean[ys, xs], self.var[ys, xs], self.count[ys, xs]
            z = (e - mean) / torch.sqrt(var + self.floor ** 2)
            cand = free & self.free_prev & (count >= self.warmup) & (e > self.tau)
            if self.z > 0:
                cand = cand & (z > self.z)
            if self.border:                                                     # new content the global motion brings
                ok = e < self.tau
                if ok.sum() >= 8:
                    gy, gx = (int(off[..., i][ok].float().median().round()) for i in (0, 1))
                    yy, xx = torch.arange(h, device=dev)[:, None] + gy, torch.arange(w, device=dev)[None, :] + gx
                    cand = cand & (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            nfree = int(free.sum())
            glob = nfree > 0 and int(cand.sum()) > self.global_frac * nfree
            if not glob:                                                        # a global change is not the norm
                d = e - mean
                self.mean = torch.where(count > 0, mean + self.ema * d, e)
                self.var = torch.where(count > 0, (1 - self.ema) * (var + self.ema * d * d), var)
                self.count = count + 1
            self.last = {"e": e, "z": z, "cand": cand}
        else:
            self.mean = torch.zeros(h, w, device=dev)
            self.var = torch.full((h, w), self.prior ** 2, device=dev)
            self.count = torch.zeros(h, w, dtype=torch.long, device=dev)
            self.timer = torch.zeros(h, w, dtype=torch.long, device=dev)
        if self.store:
            self.mem.append(f)
        self.free_prev = free.clone()
        if glob:
            out = det = torch.ones_like(out)
        else:
            det = grow(components(cand, self.min_size), self.dilate)
            self.timer = torch.where(det, torch.full_like(self.timer, self.hold), (self.timer - 1).clamp_min(0))
            out = self.timer > 0
        self.last.update({"global": glob, "out": out, "det": det})
        return out, glob


def detector_from_name(name):
    """srp_{feat}_r{radius}[s]_k{memory}_t{tau}_z{z}_m{min_size}_h{hold}_g{global_frac} (as SurpriseDetector.name;
    `s`: half-patch search; optional `_c{q}`: coherent motion over q patches)."""
    f = name.split("_")
    assert f[0] == "srp" and len(f) in (9, 10), name
    return SurpriseDetector(feat=f[1], radius=int(f[2][1:].rstrip("s")), sub=f[2].endswith("s"), memory=int(f[3][1:]),
                            tau=float(f[4][1:]), z=float(f[5][1:]), min_size=int(f[6][1:]), hold=int(f[7][1:]),
                            global_frac=float(f[8][1:]), coh=int(f[9][1:]) if len(f) == 10 else 0)


class SurpriseMap:
    """Method wrapper: at block `at` (0: before the first merge of ours, protected ALGM and protected ToMe), the patch
    features go to the detector with the out-of-focus patches; the patches it returns are protected from merging from
    that block on (`protect_tokens`); a global change protects every patch (the frame runs at full resolution from
    block `at`). `inner` None: dense frame, the detector only observes (its memory and statistics stay current).
    Detector features: `x0` the block-`at` residual after attention, `emb` the block input, `pix` pooled pixels of the
    input frame (`pixel_features`; `pix8` on 8-pixel cells; before 2026-11-20 `pix` fell through to `x0`: review flag R-21). `fired` holds (reprocessed
    out-of-focus patches, global) of the last frame. The detector state lives across frames (one per video)."""
    per_image = True

    def __init__(self, inner, detector, at=0):
        self.inner, self.det, self.at = inner, detector, at
        self.theta = getattr(inner, "theta", 0.01) if inner is not None else 0.01
        self.needs_cls = getattr(inner, "needs_cls", True) if inner is not None else False
        self.proportional = getattr(inner, "proportional", False) if inner is not None else False
        self.name = f"{detector.name}:{inner.name if inner is not None else 'dense'}"
        self.ops, self.fired, self._emb = [], (0, False), None
        self.fired_mask = None

    def __getattr__(self, attr):                      # other hooks of the inner method (before_block, finish, ...)
        inner = self.__dict__.get("inner")
        if inner is None or attr.startswith("__"):
            raise AttributeError(attr)
        return getattr(inner, attr)

    def start(self, st):
        if self.inner is not None and hasattr(self.inner, "start"):
            self.inner.start(st)
            self.ops = self.inner.ops
        else:
            self.ops = []

    def prepare(self, x, st):
        if self.det.feat == "emb":
            self._emb = x[0, N_PREFIX:]
        if self.inner is not None and hasattr(self.inner, "prepare"):
            return self.inner.prepare(x, st)
        return x

    def after_attn(self, b, x, k, p, st):
        if b == self.at:
            x, k = self._surprise(b, x, k, st)
        return x if self.inner is None else self.inner.after_attn(b, x, k, p, st)

    def _surprise(self, b, x, k, st):
        from .baselines import protect_tokens
        gh, gw = st.grid
        src = st.src[0]                                   # original patch -> token (no merge yet in this frame)
        if self.det.feat == "emb":
            feat = self._emb.reshape(gh, gw, -1)          # captured before prepare reordered the tokens
        elif self.det.feat in ("pix", "pix8"):
            feat = pixel_features(st.pixels[0], 8 if self.det.feat == "pix8" else 16)
        elif self.det.feat == "x0":
            feat = x[0, N_PREFIX:][src].reshape(gh, gw, -1)
        else:
            raise ValueError(f"unknown surprise feature {self.det.feat}")
        if st.interest is None:
            high = torch.zeros(gh, gw, dtype=torch.bool, device=x.device)
        else:
            high = (st.interest[0][src] >= self.theta).reshape(gh, gw)
        if self.inner is None:                            # dense frame: the map as the methods see it (margin 1)
            high = grow(high, 1)
        free = ~high
        self.ops.append((b, "surprise_match", self.det.flops(gh, gw, feat.shape[-1])))
        out, glob = self.det.step(feat, free)
        sel = (out & free).flatten()
        self.fired = (int(sel.sum()), bool(glob))
        self.fired_mask = (out & free)                    # diagnostics only (`vos/diag.py`)
        if self.inner is None or not bool(sel.any()):
            return x, k
        return protect_tokens(self.inner, b, x, k, st, src[sel])


def protected_form(name):
    """The protected form of a plain fallback (identical to it while nothing is protected; tests in test_r1_baselines):
    tome_r{r}_from1_noprop -> ptome_r{r}; algm_t{t} -> palgm_t{t}."""
    f = name.split("_")
    if f[0] == "tome" and f[2:] == ["from1", "noprop"]:
        return f"ptome_{f[1]}"
    if f[0] == "algm" and len(f) == 2:
        return f"palgm_{f[1]}"
    raise ValueError(f"no protected form of {name}: surprise cannot act on this fallback")
