"""R40 plan section L (implementation only, method unchanged): the redesign's per-frame steps before the encoder on device
values, so that no step reads a value back to the host and each step can be captured in a CUDA graph.
- `moved_union`: `p2_davis.shift_by_velocity(cur, prev) > 0` (each object's mask moved by its centroid displacement,
  united with the current mask), over a fixed list of candidate ids.
- `DevObjects`: `expect.objects_from_predictions` with each object's state (tracked areas, last centroid and velocity,
  frames since the loss, last tracked mask) kept as device tensors, all objects at once.
- `dilate_map`, `promote_count`, `promote_map`: the inner method's start (`dilate_interest`, `promote` size classes)
  and the wrapper's binary map; the promotion's count is a host value read with the frame's one decision.
- `ExpectDev`: `expect.ExpectMap._step` at the adopted setting (`pix` features, the floored mismatch, score `e`, no
  prototypes): camera shift, block matching, search regions, rings, surprise and the frame-wide rule, with per-object
  branches as selections and variable growth radii as Chebyshev distances. Fallback frames only keep the previous
  frame's features (score `e` reads no mismatch history and `proto` 0 no prototypes, so the rest is unused there).
Differences from the current code are rounding-level only: centroids are exact (integer sums) where the current code
rounds a float32 mean, and sums over the patch grid include zero terms the current code skips (R-37)."""
import math

import torch
import torch.nn.functional as F

from .expect import _disps, floor_norm, patch_grid
from .surprise import MEAN, STD


def shift_dev(m, d):
    """m (K, H, W) bool, d (K, 2) long: out[k, p + d_k] = m[k, p], nothing wraps (`expect.shift` per object)."""
    K, H, W = m.shape
    yy = torch.arange(H, device=m.device).view(1, H, 1) - d[:, 0].view(K, 1, 1)
    xx = torch.arange(W, device=m.device).view(1, 1, W) - d[:, 1].view(K, 1, 1)
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    src = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).reshape(K, -1)
    return torch.gather(m.reshape(K, -1), 1, src).view(K, H, W) & ok


def grow_dev(m, r):
    """m (K, H, W) bool, r (K,) long >= 0: `expect.grow` (max over a (2r + 1) square) with a radius per object, as two
    Chebyshev passes (along rows, then columns). The radius is clamped to H + W, which already covers the grid, so the
    no-pixel distance `big` stays above it and an empty mask stays empty at any radius (R-38: a search region 2 + n
    after a long loss reached `big`)."""
    K, H, W = m.shape
    dev = m.device
    big = H + W + 1
    xs, ys = torch.arange(W, device=dev), torch.arange(H, device=dev)
    rr = r.clamp(max=H + W).view(K, 1, 1)
    dx = (xs[:, None] - xs[None, :]).abs()                                       # (x, x')
    row = torch.where(m[:, :, None, :], dx[None, None], big).amin(-1) <= rr      # (K, H, W)
    dy = (ys[:, None] - ys[None, :]).abs()                                       # (y, y')
    return torch.where(row[:, None, :, :], dy[None, :, :, None], big).amin(2) <= rr


def moved_union(cur, prev, ids):
    """(H, W) id maps at t − 1 and t − 2, ids (K,) long device: `shift_by_velocity(cur, prev) > 0`."""
    H, W = cur.shape
    if ids.numel() == 0:
        return torch.zeros(H, W, dtype=torch.bool, device=cur.device)
    mc = cur[None] == ids[:, None, None]                                         # (K, H, W)
    mp = prev[None] == ids[:, None, None]
    yy = torch.arange(H, device=cur.device).view(1, H, 1)
    xx = torch.arange(W, device=cur.device).view(1, 1, W)
    na, nb = mc.sum((1, 2)), mp.sum((1, 2))

    def mean(m, n):
        return torch.stack([(m * yy).sum((1, 2)), (m * xx).sum((1, 2))], 1).double() / n.clamp_min(1).double()[:, None]
    d = torch.round(mean(mc, na) - mean(mp, nb)).long()                         # round half to even, as the current code
    d = torch.where(((na > 0) & (nb > 0))[:, None], d, torch.zeros_like(d))      # an object missing in either: no move
    return (shift_dev(mc, d) | mc).any(0)


class DevObjects:
    """`objects_from_predictions(idx1, idx2, ids, ostate, motion)` with the per-object state on the device. One per
    video. `update` returns a dict of (K, ...) tensors (M, T, core, speed, lost, n, v, Mlast, hasM) and `all_lost`
    (device bool), in the order of `ids`."""
    NH = 10                                                                      # tracked areas kept (median of the last 10)

    def __init__(self, motion, theta=0.01):
        self.motion, self.theta = motion, theta
        self.ids, self.s, self.idt = [], None, None

    def _rows(self, ids, gh, gw, dev):
        """State rows for `ids` (new objects start empty), keeping the rows of known ids."""
        K = len(ids)
        new = {"areas": torch.zeros(K, self.NH, dtype=torch.float64, device=dev),
               "cnt": torch.zeros(K, dtype=torch.long, device=dev),
               "v": torch.zeros(K, 2, dtype=torch.float64, device=dev),
               "n": torch.zeros(K, dtype=torch.long, device=dev),
               "Mlast": torch.zeros(K, gh, gw, dtype=torch.bool, device=dev),
               "hasM": torch.zeros(K, dtype=torch.bool, device=dev)}
        for i, o in enumerate(ids):
            if self.s is not None and o in self.ids:
                j = self.ids.index(o)
                for k in new:
                    new[k][i] = self.s[k][j]
        self.ids, self.s = list(ids), new
        self.version = getattr(self, "version", 0) + 1                           # graphs that read the old rows are stale
        self.idt = torch.tensor(ids, dtype=torch.long, device=dev) if K else torch.zeros(0, dtype=torch.long, device=dev)

    def ensure(self, ids, shape, dev):
        """State rows for this frame's ids (outside any graph capture)."""
        gh, gw = -(-shape[0] // 16), -(-shape[1] // 16)
        if list(ids) != self.ids or self.s is None:
            self._rows(ids, gh, gw, dev)

    def state(self):
        return list(self.s.values()) if self.s else []

    def update(self, idx1, idx2, ids):
        h, w = idx1.shape
        dev = idx1.device
        H16, W16 = -(-h // 16) * 16, -(-w // 16) * 16
        lh, lw = (H16 - h) // 2, (W16 - w) // 2
        self.ensure(ids, (h, w), dev)
        s, o = self.s, self.idt
        K = len(ids)
        if K == 0:
            return {}, torch.zeros((), dtype=torch.bool, device=dev)
        yy = torch.arange(h, device=dev).view(1, h, 1)
        xx = torch.arange(w, device=dev).view(1, 1, w)

        def cent(m):                                                             # exact mean position, patch units
            n = m.sum((1, 2))
            sy, sx = (m * yy).sum((1, 2)).double(), (m * xx).sum((1, 2)).double()
            nd = n.clamp_min(1).double()
            return torch.stack([(sy / nd + lh - 7.5) / 16, (sx / nd + lw - 7.5) / 16], 1), n > 0
        m1 = idx1[None] == o[:, None, None]                                      # (K, h, w)
        a1 = m1.sum((1, 2))
        valid = torch.arange(self.NH, device=dev)[None] >= self.NH - s["cnt"][:, None]   # the last cnt areas
        srt = torch.where(valid, s["areas"], math.inf).sort(1).values
        ref = torch.where(s["cnt"] > 0, srt.gather(1, (s["cnt"] // 2)[:, None])[:, 0], torch.zeros_like(srt[:, 0]))
        lost = a1.double() < torch.clamp(0.1 * ref, min=16.0)
        z = F.pad(m1.float(), (lw, W16 - w - lw, lh, H16 - h - lh))
        fr = F.avg_pool2d(z[:, None], 16)[:, 0]                                 # (K, gh, gw)
        M, T = fr >= self.theta, fr >= 0.5
        T = torch.where(T.any((1, 2))[:, None, None], T, M)
        c1, ok1 = cent(m1)
        if idx2 is not None:
            c2, ok2 = cent(idx2[None] == o[:, None, None])
            dv = torch.where((ok1 & ok2)[:, None], c1 - c2, torch.zeros_like(c1))
        else:
            dv = torch.zeros_like(c1)
        core = (M | shift_dev(M, torch.round(dv).long())) if self.motion else M.clone()
        tr = ~lost                                                               # state in place (graph capture)
        s["areas"].copy_(torch.where(tr[:, None], torch.cat([s["areas"][:, 1:], a1.double()[:, None]], 1), s["areas"]))
        s["cnt"].copy_(torch.where(tr, (s["cnt"] + 1).clamp(max=self.NH), s["cnt"]))
        s["n"].copy_(torch.where(tr, torch.zeros_like(s["n"]), s["n"] + 1))
        s["v"].copy_(torch.where(tr[:, None], dv, s["v"]))
        s["Mlast"].copy_(torch.where(tr[:, None, None], M, s["Mlast"]))
        s["hasM"].copy_(s["hasM"] | tr)
        out = {"M": M, "T": T, "core": core, "speed": torch.hypot(dv[:, 0], dv[:, 1]), "lost": lost, "n": s["n"].clone(),
               "v": s["v"].clone(), "Mlast": s["Mlast"].clone(), "hasM": s["hasM"].clone()}
        return out, lost.all()


def dilate_map(interest, dilate, grid):
    """(1, N) patch coverage -> `dilate_interest` (the inner method's margin)."""
    gh, gw = grid
    if not dilate:
        return interest
    return F.max_pool2d(interest.view(1, 1, gh, gw), 2 * dilate + 1, 1, dilate).flatten(1)


def promote_count(n_high, size_class, N):
    """Host: patches the size classes add (`promote`: c − n), from the high count read with the frame's decision."""
    return min(N, -(-n_high // size_class) * size_class) - n_high if size_class else 0


def promote_map(interest, theta, size_class, grid, k):
    """`promote` with its count k (host; `promote_count`) and the wrapper's binary map (`ExpectMap.start`), (1, N)
    float. `topk(k)` as the current code: which of several equal-score patches it takes depends on k."""
    if size_class and k > 0:
        gh, gw = grid
        high = interest[0] >= theta
        h = high.float().view(1, 1, gh, gw)
        near = 2 * F.max_pool2d(h, 3, 1, 1) + F.max_pool2d(h, 7, 1, 3)
        score = (near.flatten() + interest[0]).masked_fill(high, -1.0)
        idx = score.topk(k).indices
        sel = torch.zeros(interest.shape[1], dtype=torch.bool, device=interest.device).index_fill_(0, idx, True)
        interest = torch.where(sel[None], torch.full_like(interest, theta), interest)
    return (interest >= theta).float()


class ExpectDev:
    """`ExpectMap._step` for one video at the adopted setting, on device values; `step` returns the tiered map (1, N)."""

    def __init__(self, cfg, theta, srp_value, grid, device):
        assert cfg.feat == "pix" and cfg.score == "e" and not cfg.proto and (cfg.sigma > 0 or cfg.tol > 0), cfg.name
        self.cfg, self.theta, self.srp, self.grid = cfg, theta, srp_value, grid
        gh, gw = grid
        N = gh * gw
        dev = device
        self.std = torch.tensor(STD, device=dev)[:, None, None]
        self.mean = torch.tensor(MEAN, device=dev)[:, None, None]
        self.dcam = _disps(cfg.r_cam, dev)                                       # (G, 2), smallest displacement first
        self.dobj = _disps(cfg.r_obj, dev)
        yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing="ij")
        self.yy, self.xx = yy, xx
        p = torch.stack([yy.flatten(), xx.flatten()], 1)
        q = p[None] - self.dcam[:, None]
        ok = (q[..., 0] >= 0) & (q[..., 0] < gh) & (q[..., 1] >= 0) & (q[..., 1] < gw)
        self.cam_qi = q[..., 0].clamp(0, gh - 1) * gw + q[..., 1].clamp(0, gw - 1)
        self.cam_pi = torch.arange(N, device=dev)[None].expand_as(self.cam_qi)
        self.cam_ok, self.cam_n = ok, ok.sum(1).clamp_min(1)
        qy, qx = p[:, 0][None], p[:, 1][None]                                    # block matching over every patch q
        py, px = qy + self.dobj[:, :1], qx + self.dobj[:, 1:]
        self.bm_ok = (py >= 0) & (py < gh) & (px >= 0) & (px < gw)               # (G, N)
        self.bm_pi = py.clamp(0, gh - 1) * gw + px.clamp(0, gw - 1)
        self.bm_q = torch.arange(N, device=dev)[None].expand_as(self.bm_pi)
        self.ops_sim = 2 * N * N * 48
        self.ops_floor = (2 * cfg.tol + 1) ** 2 * N * 48 * 8
        self.prev = torch.zeros(N, 48, device=dev)                               # persistent (graph capture; no lazy
        self.prev_pool = torch.zeros(3, gh * 4, gw * 4, device=dev)              # allocation inside compiled code)
        z = torch.zeros((), dtype=torch.long, device=dev)
        self.sum_high, self.sum_mid, self.n_global = z.clone(), z.clone(), z.clone()
        self.reset()

    def reset(self):
        self.has_prev = False
        self.sum_high.zero_(), self.sum_mid.zero_(), self.n_global.zero_()
        self.last = None                                                         # (high, mid, global) of the last main frame

    def state(self):
        return [self.prev, self.prev_pool, self.sum_high, self.sum_mid, self.n_global]

    def _keep(self, f, pool):
        self.prev.copy_(f)
        self.prev_pool.copy_(pool)
        self.has_prev = True

    def features(self, pixels):
        """pixels (3, H, W) normalised -> (`unit(pixel_features)` (N, 48), pooled pixels (3, H/4, W/4))."""
        from .surprise import unit
        x = pixels.float() * self.std + self.mean
        pool = F.avg_pool2d(x[None], 4)[0]                                       # `pooled_pixels` = `pixel_features` pooling
        gh, gw = self.grid
        f = pool.reshape(3, gh, 4, gw, 4).permute(1, 3, 0, 2, 4).reshape(gh, gw, 48)
        return unit(f).reshape(gh * gw, -1), pool

    def floor_mismatch(self, Pn, Pp, g):
        """`expect.floor_mismatch` with the camera shift g (2,) a device value."""
        cfg, (gh, gw) = self.cfg, self.grid
        fn = patch_grid(Pn, 0, 0, gh, gw, 0)
        an = floor_norm(fn, fn.reshape(-1, 48).mean(0), cfg.sigma)
        mu_p = patch_grid(Pp, 0, 0, gh, gw, 0).reshape(-1, 48).mean(0)
        py, px = self.yy - g[0], self.xx - g[1]
        border = ((py < 0) | (py >= gh) | (px < 0) | (px >= gw)).flatten()
        qy, qx = py.clamp(0, gh - 1), px.clamp(0, gw - 1)
        best = None
        for oy in range(-cfg.tol, cfg.tol + 1):
            for ox in range(-cfg.tol, cfg.tol + 1):
                bp = floor_norm(patch_grid(Pp, oy, ox, gh, gw, cfg.tol), mu_p, cfg.sigma)[qy, qx]
                e = ((an - bp) ** 2).sum(-1) / 96.0
                best = e if best is None else torch.minimum(best, e)
        e = torch.where(border, torch.ones_like(best.flatten()), best.flatten())
        return e, border

    def block_match(self, S, T):
        """`expect.block_match` for every object at once: T (K, gh, gw) bool -> (K, 2) long."""
        K = T.shape[0]
        Tf = T.reshape(K, 1, -1)
        okq = self.bm_ok[None] & Tf                                              # (K, G, N)
        vals = S[self.bm_pi, self.bm_q]                                          # (G, N)
        s = (vals[None] * okq).sum(-1) / okq.sum(-1).clamp_min(1)
        s = s.masked_fill(okq.sum(-1) * 2 < Tf.sum(-1), -math.inf)
        return self.dobj[s.argmax(-1)]

    def step(self, pixels, interest, objs, passive):
        """pixels (1, 3, H, W); interest (1, N) binary map (`promote_map`) or None; objs: `DevObjects.update` output;
        passive: fallback frame (host decision). Returns (tiered map (1, N) or the input, ops [(name, flops)])."""
        cfg, (gh, gw) = self.cfg, self.grid
        f, pool = self.features(pixels[0])
        if not self.has_prev or passive or interest is None:                    # first frame, fallback frames: keep the
            self._keep(f, pool)                                                  # features only (nothing else is read later)
            return interest, []
        S = f @ self.prev.T                                                      # (N, N), the autocast dtype as now
        ops = [("expect_sim", self.ops_sim), ("expect_floor", self.ops_floor)]
        s = (S[self.cam_pi, self.cam_qi] * self.cam_ok).sum(1) / self.cam_n
        g = self.dcam[s.argmax().view(1)][0]                                     # 1-element index: no host value
        e, border = self.floor_mismatch(pool, self.prev_pool, g)
        high_now = (interest[0] >= self.theta).reshape(gh, gw)
        low = ~high_now
        add_h = torch.zeros(gh, gw, dtype=torch.bool, device=f.device)
        add_m, zone = add_h.clone(), add_h.clone()
        if objs:
            lost, tr = objs["lost"], ~objs["lost"]
            core, speed = objs["core"], objs["speed"]
            if cfg.reloc:
                tany = objs["T"].any((1, 2))
                d = self.block_match(S, objs["T"])
                core = torch.where(tany[:, None, None], core | shift_dev(objs["M"], d), core)
                speed = torch.where(tany, torch.maximum(speed, torch.hypot(d[:, 0].double(), d[:, 1].double())), speed)
            ring = grow_dev(core, 1 + torch.clamp(torch.ceil(speed), min=1).long())
            gr1 = grow_dev(core, torch.ones_like(lost, dtype=torch.long))
            add_h |= (gr1 & tr[:, None, None]).any(0)
            zone |= (ring & tr[:, None, None]).any(0)
            if cfg.ring:
                add_m |= (ring & tr[:, None, None]).any(0)
            if cfg.search:
                n = objs["n"]
                mv = torch.round(n[:, None].double() * objs["v"]).long()
                disc = grow_dev(shift_dev(objs["Mlast"], mv), 2 + n)
                disc = disc & (lost & objs["hasM"])[:, None, None]
                add_m |= disc.any(0)
                zone |= disc.any(0)
        fired = ((e >= cfg.thr) | border).reshape(gh, gw)
        fired_low = fired & low
        add_h |= fired_low & zone
        add_m |= fired_low & ~zone
        glob = fired_low.sum() > cfg.global_frac * low.sum()
        add_m |= low & glob
        new_h = add_h & low
        new_m = add_m & low & ~new_h
        self._keep(f, pool)
        nh, nm = new_h.sum(), new_m.sum()
        self.sum_high += nh
        self.sum_mid += nm
        self.n_global += glob.long()
        self.last = (nh, nm, glob)
        out = torch.where(new_h.flatten(), 1.0, torch.where(new_m.flatten(), self.srp, interest[0]))
        return out[None], ops
