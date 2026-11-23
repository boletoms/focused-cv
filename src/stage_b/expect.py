"""R40 redesign (owner 2026-11-21): interest from an expectation of where objects and background will be, and surprise
as a deviation from it, decided on the current frame before the first merge.

Owner's definitions and rules (`DECISIONS.md` 2026-11-21): we expect the next frame from the objects, the background
and their expected positions; a surprise is any deviation from that expectation (the strict definition is the reference
goal, covered as far as is feasible); surprise only promotes a low-focus area, never demotes; a false positive is
better than a false negative; components that need training are optional future work; when nothing is confidently
tracked the frame runs the compared baseline's own mechanism.

Per frame, at block `at` (0: after block 0's attention, before the first merge; every patch is still a token):
- features `f` of the current frame (`x0` block-0 attention output, `emb` patch embeddings, `pix` pooled pixels),
  centred and normalised (`surprise.unit`), and the previous frame's `prev` (kept in `ExpectState`);
- S = f · prevᵀ (one product serves the three uses below);
- camera displacement g: the integer shift within `r_cam` patches with the highest mean cosine (`camera`);
- per tracked object (from the evaluator: its previous patch mask M, its expected core = M with the centroid shift
  when `motion`): with `reloc`, the block-matching displacement d* of M on the current frame within `r_obj` patches
  (`block_match`) adds M shifted by d* to the core. The core grown by one patch is high interest; a ring of
  1 + max(1, ceil(speed)) patches around it (`ring`) takes the baseline's rules (middle tier);
- per lost object (evaluator: predicted area collapsed): a search region, its last tracked mask moved by its last
  velocity times n and grown by 2 + n patches, n frames after the loss (`search`, middle tier; R39 Q8: the target's
  centroid lies within 2 + n patches of the moved last centroid on 81% of lost frames; growing the mask, not a disc around
  the centroid, keeps a large object's extent, R40 B2); with `proto`, the region's patches whose cosine to the object's
  prototype (mean feature over its last tracked mask) reaches `proto` are high interest;
- surprise: mismatch e(p) = 1 − cosine between f(p) and prev(p − g) (radius 0, or the best within one patch); score e,
  or z = (e − mean) / sqrt(var + floor²) over the last `win` frames at p (with fewer than 3 frames of history every patch
  fires); with `pix` and a variance floor `sigma` or a local tolerance `tol` (R40 B4), e(p) = min over offsets o in
  [−tol, tol]² (4-pixel steps) of ||a'(p) − b'_o(p − g)||² / (2 D), a' the frame-centred patch feature centred per patch
  and divided by sqrt(var + sigma²) (sigma 0, tol 0: 1 − cosine), so flat or blurred patches, whose normalised direction
  is noise, and motion of a few pixels inside a patch do not fire (`floor_mismatch`); fired where the score reaches `thr`, and
  always on new content entering at the border (p − g outside the previous frame). Fired out-of-focus patches inside an
  object's ring or a lost object's disc are high interest (target-related, owner P2); the others take the baseline's
  rules. If more than `global_frac` of the out-of-focus patches fire, all of them take the baseline's rules.
Never demotes: only patches below the interest threshold change. The middle tier is marked with the inner method's
`srp_value` (ALGMSurprise: merged by ALGM's rules among themselves); an inner method without a middle tier protects
every promoted patch. A passive wrapper (fallback frames) only observes: it keeps `prev`, the mismatch history and the
prototypes current. Extra FLOPs (the product S and the prototype products) are reported in `ops`."""
import math
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.stage_a.model import N_PREFIX

from .surprise import pixel_features, unit


@dataclass
class ExpectConfig:
    feat: str = "x0"
    motion: bool = True          # centroid shift in the core (the evaluator applies it to the map)
    reloc: bool = False          # block matching on the current frame
    radius: int = 0              # mismatch search radius (patches) around p − g
    score: str = "e"             # "e" or "z"
    thr: float = 0.5
    ring: bool = True
    search: bool = True
    proto: float = 0.0           # 0: off; else the cosine threshold inside a lost object's disc
    global_frac: float = 0.5
    floor: float = 0.05
    win: int = 10
    r_obj: int = 8
    r_cam: int = 4
    sigma: float = 0.0           # pix: variance floor of the per-patch normalisation in the mismatch (R40 B4)
    tol: int = 0                 # pix: local tolerance of the mismatch, offsets of 4 pixels (R40 B4)
    mbase: bool = False          # R40 M: motion at the block-matching radius -> the frame runs the baseline (`alt`)

    @property
    def name(self):
        s = f"exp-{self.feat}" + ("-mot" if self.motion else "") + ("-rl" if self.reloc else "") + f"-r{self.radius}" + \
            f"-{self.score}{self.thr:g}" + ("-ring" if self.ring else "") + ("-srch" if self.search else "") + \
            (f"-pt{self.proto:g}" if self.proto else "") + (f"-f{self.sigma:g}" if self.sigma else "") + \
            (f"-o{self.tol}" if self.tol else "") + ("-mb" if self.mbase else "")
        return s


def config_from_name(name):
    """'exp-x0-mot-rl-r1-z2.5-ring-srch-pt0.6' -> ExpectConfig (inverse of ExpectConfig.name)."""
    t = name.split("-")
    assert t[0] == "exp", name
    c = ExpectConfig(feat=t[1], motion="mot" in t, reloc="rl" in t, ring="ring" in t, search="srch" in t, mbase="mb" in t)
    for x in t[2:]:
        if x[0] == "r" and x[1:].isdigit():
            c.radius = int(x[1:])
        elif x[0] in "ez" and x not in ("exp",):
            try:
                c.score, c.thr = x[0], float(x[1:])
            except ValueError:
                pass
        elif x.startswith("pt"):
            c.proto = float(x[2:])
        elif x[0] == "f" and x[1:2].isdigit():
            c.sigma = float(x[1:])
        elif x[0] == "o" and x[1:].isdigit():
            c.tol = int(x[1:])
    assert c.name == name, (c.name, name)
    return c


@dataclass
class ExpectState:
    """Per video, shared by the wrappers of one setting (the reduced method and the passive fallback)."""
    prev: torch.Tensor = None                    # (N, D) previous frame's features
    prev_pool: torch.Tensor = None               # (3, H/4, W/4) previous frame's pooled pixels (sigma, tol)
    grid: tuple = None
    ehist: deque = field(default_factory=deque)  # last `win` mismatch grids (N,)
    protos: dict = field(default_factory=dict)   # object id -> (D,) prototype
    objects: list = field(default_factory=list)  # evaluator, per frame: dicts (see ExpectMap)
    last: dict = field(default_factory=dict)     # last frame's counts (diagnostics)


def _disps(r, device):
    d = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    d.sort(key=lambda v: (v[0] ** 2 + v[1] ** 2, v))                            # ties: the smallest displacement
    return torch.tensor(d, device=device)


def camera(S, gh, gw, r):
    """S (N, N): S[p, q] = cos(f_now[p], f_prev[q]). Integer (dy, dx): content at p was at p − g."""
    dev = S.device
    d = _disps(r, dev)                                                           # (G, 2)
    yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing="ij")
    p = torch.stack([yy.flatten(), xx.flatten()], 1)                             # (N, 2)
    q = p[None] - d[:, None]                                                     # (G, N, 2)
    ok = (q[..., 0] >= 0) & (q[..., 0] < gh) & (q[..., 1] >= 0) & (q[..., 1] < gw)
    qi = (q[..., 0].clamp(0, gh - 1) * gw + q[..., 1].clamp(0, gw - 1))
    pi = torch.arange(gh * gw, device=dev)[None].expand_as(qi)
    s = (S[pi, qi] * ok).sum(1) / ok.sum(1).clamp_min(1)
    return tuple(int(v) for v in d[int(torch.argmax(s))])


def mismatch(S, gh, gw, g, radius):
    """e (N,): 1 − cos(f_now[p], f_prev[p − g]) (radius 1: the best within one patch); border (N,) bool: p − g outside."""
    dev = S.device
    yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing="ij")
    py, px = yy.flatten() - g[0], xx.flatten() - g[1]
    border = (py < 0) | (py >= gh) | (px < 0) | (px >= gw)
    best = torch.full((gh * gw,), -math.inf, device=dev)
    pi = torch.arange(gh * gw, device=dev)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            qy, qx = py + dy, px + dx
            ok = (qy >= 0) & (qy < gh) & (qx >= 0) & (qx < gw)
            c = S[pi, qy.clamp(0, gh - 1) * gw + qx.clamp(0, gw - 1)].masked_fill(~ok, -math.inf)
            best = torch.maximum(best, c)
    e = torch.where(torch.isfinite(best), 1 - best, torch.ones_like(best))
    e[border] = 1.0
    return e, border


def pooled_pixels(pixels):
    """Normalised frame (3, H, W) -> RGB in [0, 1] average-pooled over 4 x 4-pixel blocks (3, H/4, W/4), as `pix`."""
    from .surprise import MEAN, STD
    x = pixels.float() * torch.tensor(STD, device=pixels.device)[:, None, None] + \
        torch.tensor(MEAN, device=pixels.device)[:, None, None]
    return F.avg_pool2d(x[None], 4)[0]


def patch_grid(P, oy, ox, gh, gw, pad):
    """`pix` patch features (gh, gw, 48) of the pooled frame P read 4 oy, 4 ox pixels further (edges copied)."""
    Pp = F.pad(P[None], (pad,) * 4, mode="replicate")[0] if pad else P
    Q = Pp[:, pad + oy:pad + oy + 4 * gh, pad + ox:pad + ox + 4 * gw]
    return Q.reshape(3, gh, 4, gw, 4).permute(1, 3, 0, 2, 4).reshape(gh, gw, 48)


def floor_norm(f, mu, sigma):
    """Frame-centred (mu: the frame's mean patch), centred per patch, divided by sqrt(var + sigma²) (sigma 0: layer norm)."""
    f = f - mu
    f = f - f.mean(-1, keepdim=True)
    return f / torch.sqrt(f.var(-1, unbiased=False, keepdim=True) + sigma ** 2 + 1e-5)


def floor_mismatch(Pn, Pp, gh, gw, g, sigma, tol):
    """R40 B4: e (N,) = min over offsets o in [−tol, tol]² of ||a'(p) − b'_o(p − g)||² / (2 D) between the current pooled
    frame Pn and the previous Pp (floor_norm with sigma); border (N,) bool: p − g outside (e 1 there). Also the FLOPs."""
    dev = Pn.device
    fn = patch_grid(Pn, 0, 0, gh, gw, 0)
    an = floor_norm(fn, fn.reshape(-1, 48).mean(0), sigma)
    mu_p = patch_grid(Pp, 0, 0, gh, gw, 0).reshape(-1, 48).mean(0)
    yy, xx = torch.meshgrid(torch.arange(gh, device=dev), torch.arange(gw, device=dev), indexing="ij")
    py, px = yy - g[0], xx - g[1]
    border = ((py < 0) | (py >= gh) | (px < 0) | (px >= gw)).flatten()
    qy, qx = py.clamp(0, gh - 1), px.clamp(0, gw - 1)
    best = None
    for oy in range(-tol, tol + 1):
        for ox in range(-tol, tol + 1):
            bp = floor_norm(patch_grid(Pp, oy, ox, gh, gw, tol), mu_p, sigma)[qy, qx]
            e = ((an - bp) ** 2).sum(-1) / 96.0
            best = e if best is None else torch.minimum(best, e)
    e = best.flatten()
    e[border] = 1.0
    return e, border, (2 * tol + 1) ** 2 * gh * gw * 48 * 8


def block_match(S, M, gh, gw, r):
    """Displacement (dy, dx) maximising the mean of S[q + d, q] over q in M (at least half of M inside)."""
    dev = S.device
    q = torch.nonzero(M.flatten()).flatten()
    if len(q) == 0:
        return (0, 0)
    qy, qx = q // gw, q % gw
    d = _disps(r, dev)
    py, px = qy[None] + d[:, :1], qx[None] + d[:, 1:]
    ok = (py >= 0) & (py < gh) & (px >= 0) & (px < gw)
    pi = py.clamp(0, gh - 1) * gw + px.clamp(0, gw - 1)
    s = (S[pi, q[None].expand_as(pi)] * ok).sum(1) / ok.sum(1).clamp_min(1)
    s = s.masked_fill(ok.sum(1) * 2 < len(q), -math.inf)
    return tuple(int(v) for v in d[int(torch.argmax(s))])


def shift(m, dy, dx):
    """(gh, gw) bool moved by (dy, dx): out[p + d] = m[p]."""
    out = torch.zeros_like(m)
    H, W = m.shape
    if abs(dy) >= H or abs(dx) >= W:
        return out
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


def grow(m, r):
    if r <= 0:
        return m.clone()
    return F.max_pool2d(m.float()[None, None], 2 * r + 1, 1, r)[0, 0] > 0


class ExpectMap:
    """Method wrapper (see the module docstring). `state.objects` is set by the evaluator before each frame: a list of
    dicts {id, M (gh, gw) bool previous patch mask, T (gh, gw) bool template mask, core (gh, gw) bool, speed (patches),
    lost (bool), n (frames since the loss), c (2,) last tracked centroid in patch units, v (2,) last velocity,
    Mlast (gh, gw) bool last tracked patch mask or None}."""
    per_image = True

    def __init__(self, inner, state, cfg, passive=False, at=0, theta=0.01):
        self.inner, self.state, self.cfg, self.passive, self.at = inner, state, cfg, passive, at
        self.theta = getattr(inner, "theta", theta) if inner is not None else theta
        self.needs_cls = getattr(inner, "needs_cls", True) if inner is not None else False
        self.proportional = getattr(inner, "proportional", False) if inner is not None else False
        self.name = f"{cfg.name}:{inner.name if inner is not None else 'dense'}" + (":passive" if passive else "")
        self.ops, self._emb, self.fired_mask = [], None, None
        self.alt, self._alt_frame, self._raw = None, False, None   # R40 M: the baseline (the fallback's inner method)

    def __getattr__(self, attr):
        inner = self.__dict__.get("inner")
        if inner is None or attr.startswith("__"):
            raise AttributeError(attr)
        return getattr(inner, attr)

    def start(self, st):
        self._alt_frame, self._raw = False, st.interest                           # the map as a fallback frame gets it
        if self.inner is not None and hasattr(self.inner, "start"):
            self.inner.start(st)
            self.ops = self.inner.ops
        else:
            self.ops = []
        if not self.passive and st.interest is not None:                          # binary map (0 or 1) after the
            st.interest = (st.interest >= self.theta).float()                     # inner's margin and size classes;
            st.interest0 = st.interest                                            # 0.5 then only marks the middle tier
        if self.alt is not None and not self.passive:                             # R40 M: ready to switch this frame
            if hasattr(self.alt, "start"):
                self.alt.start(st)
            self.alt.ops = self.ops                                               # its FLOPs count in this frame's list

    def prepare(self, x, st):
        if self.cfg.feat == "emb":
            self._emb = x[0, N_PREFIX:].clone()
        if self.inner is not None and hasattr(self.inner, "prepare"):
            return self.inner.prepare(x, st)
        return x

    def after_attn(self, b, x, k, p, st):
        if b == self.at:
            x, k = self._step(b, x, k, st)
        if self.inner is None:
            return x
        return (self.alt if self._alt_frame else self.inner).after_attn(b, x, k, p, st)

    def _features(self, x, st):
        gh, gw = st.grid
        src = st.src[0]
        if self.cfg.feat == "x0":
            f = x[0, N_PREFIX:][src]
        elif self.cfg.feat == "emb":
            f = self._emb[src] if self._emb.shape[0] == src.shape[0] else self._emb
        elif self.cfg.feat == "pix":
            f = pixel_features(st.pixels[0], 16).reshape(gh * gw, -1)
        else:
            raise ValueError(self.cfg.feat)
        return unit(f.reshape(gh, gw, -1)).reshape(gh * gw, -1)

    def _step(self, b, x, k, st):
        cfg, S_, gh, gw = self.cfg, self.state, *st.grid
        f = self._features(x, st)
        N, D = f.shape
        dev = f.device
        floor = cfg.feat == "pix" and (cfg.sigma > 0 or cfg.tol > 0)
        pool = pooled_pixels(st.pixels[0]) if floor else None
        if S_.prev is None or S_.grid != (gh, gw):
            S_.prev, S_.grid, S_.prev_pool = f, (gh, gw), pool
            S_.ehist.clear()
            self._update_protos(None)
            S_.last = {"high": 0, "mid": 0, "global": False, "g": (0, 0), "first": True}
            return x, k
        self._update_protos(S_.prev)                                               # from t − 1 features and masks
        S = f @ S_.prev.T                                                          # (N, N)
        self.ops.append((b, "expect_sim", 2 * N * N * D))
        g = camera(S, gh, gw, cfg.r_cam)
        if floor:
            e, border, fl = floor_mismatch(pool, S_.prev_pool, gh, gw, g, cfg.sigma, cfg.tol)
            self.ops.append((b, "expect_floor", fl))
        else:
            e, border = mismatch(S, gh, gw, g, cfg.radius)
        hist = list(S_.ehist)
        S_.ehist.append(e)
        while len(S_.ehist) > cfg.win:
            S_.ehist.popleft()
        if self.passive or st.interest is None:
            S_.prev, S_.prev_pool = f, pool
            S_.last = {"high": 0, "mid": 0, "global": False, "g": g, "passive": True}
            return x, k
        src = st.src[0]
        high_now = (st.interest[0][src] >= self.theta).reshape(gh, gw)
        add_h = torch.zeros(gh, gw, dtype=torch.bool, device=dev)
        add_m = torch.zeros_like(add_h)
        zone = torch.zeros_like(add_h)
        Sg = S if cfg.reloc else None
        fast = False                                                              # R40 M: motion at the search radius
        for o in S_.objects:
            if o["lost"]:
                if cfg.search and o.get("Mlast") is not None:
                    n = o["n"]                                              # the last tracked mask, moved by its last
                    disc = grow(shift(o["Mlast"], int(round(n * o["v"][0])), int(round(n * o["v"][1]))), 2 + n)
                    add_m |= disc                                           # velocity, grown by 2 + n patches
                    zone |= disc
                    pr = S_.protos.get(o["id"])
                    if cfg.proto and pr is not None:
                        cs = (f @ pr).reshape(gh, gw)
                        self.ops.append((b, "expect_proto", 2 * N * D))
                        add_h |= disc & (cs >= cfg.proto)
                continue
            core, speed = o["core"].clone(), float(o["speed"])
            fast |= cfg.mbase and speed >= cfg.r_obj
            if Sg is not None and bool(o["T"].any()):
                d = block_match(Sg, o["T"], gh, gw, cfg.r_obj)
                core |= shift(o["M"], *d)
                speed = max(speed, math.hypot(*d))
                fast |= cfg.mbase and max(abs(d[0]), abs(d[1])) >= cfg.r_obj
            add_h |= grow(core, 1)
            r = 1 + max(1, math.ceil(speed))
            ring = grow(core, r)
            zone |= ring
            if cfg.ring:
                add_m |= ring
        if fast and self.alt is not None:                                        # the baseline runs this frame, on the
            self._alt_frame = True                                                # input a fallback frame gets
            st.interest = self._raw
            S_.prev, S_.prev_pool = f, pool
            S_.last = {"high": 0, "mid": 0, "global": False, "g": g, "mbase": True}
            return x, k
        if cfg.score == "z" and len(hist) >= 3:                                   # tolerance from up to `win` frames
            H = torch.stack(hist[-cfg.win:])
            sc = (e - H.mean(0)) / torch.sqrt(H.var(0, unbiased=False) + cfg.floor ** 2)
        elif cfg.score == "z":
            sc = torch.full_like(e, math.inf)                                       # no tolerance yet: fire (FP over FN)
        else:
            sc = e
        fired = ((sc >= cfg.thr) | border).reshape(gh, gw)
        low = ~high_now
        fired_low = fired & low
        add_h |= fired_low & zone
        add_m |= fired_low & ~zone
        glob = bool(fired_low.sum() > cfg.global_frac * low.sum())
        if glob:
            add_m |= low
        new_h = add_h & low
        new_m = add_m & low & ~new_h
        self.fired_mask = fired_low
        S_.prev, S_.prev_pool = f, pool
        S_.last = {"high": int(new_h.sum()), "mid": int(new_m.sum()), "global": glob, "g": g}
        return self._apply(b, x, k, st, src[new_h.flatten()], src[new_m.flatten()])

    def _apply(self, b, x, k, st, tok_h, tok_m):
        inner = self.inner
        if len(tok_h) == 0 and len(tok_m) == 0:
            return x, k
        if hasattr(inner, "sched"):                                                # ours (FastInterestToMe): no middle tier
            from .baselines import protect_tokens
            return protect_tokens(inner, b, x, k, st, torch.cat([tok_h, tok_m]))
        st.interest = st.interest.clone()
        st.interest[0, tok_h] = 1.0
        st.interest[0, tok_m] = getattr(inner, "srp_value", 1.0)
        return x, k

    def _update_protos(self, prev):
        """Prototype of every object tracked at t − 1: mean feature of t − 1 over its template mask (kept while lost)."""
        if prev is None:
            return
        for o in self.state.objects:
            if not o["lost"] and bool(o["T"].any()):
                pr = prev[o["T"].flatten()].mean(0)
                self.state.protos[o["id"]] = pr / pr.norm().clamp_min(1e-8)


def objects_from_predictions(idx1, idx2, ids, ostate, motion, theta=0.01):
    """Run-time object states for `ExpectState.objects` (no ground truth). idx1, idx2: (h, w) predicted object-id masks
    at t − 1 and t − 2 (idx2 None at the start); ids: objects that appeared before t; ostate: per-object state kept across
    frames (areas while tracked, last tracked centroid and velocity, frames since the loss). Patch grid as the encoder's
    (padded symmetrically to multiples of 16). An object is lost when its predicted area at t − 1 falls below a tenth of
    the median of its last 10 tracked areas, or below 16 pixels. Returns (objects, all lost)."""
    h, w = idx1.shape
    H16, W16 = -(-h // 16) * 16, -(-w // 16) * 16
    lh, lw = (H16 - h) // 2, (W16 - w) // 2

    def grid(m):
        z = torch.zeros(H16, W16, device=m.device)
        z[lh:lh + h, lw:lw + w] = m.float()
        return F.avg_pool2d(z[None, None], 16)[0, 0]

    def cent(m):
        ys, xs = torch.nonzero(m, as_tuple=True)
        if len(ys) == 0:
            return None
        return ((ys.float().mean().item() + lh - 7.5) / 16, (xs.float().mean().item() + lw - 7.5) / 16)

    out = []
    for o in ids:
        s = ostate.setdefault(o, {"areas": [], "c": None, "v": (0.0, 0.0), "n": 0})
        m1 = idx1 == o
        a1 = int(m1.sum())
        hist = s["areas"][-10:]
        ref = float(sorted(hist)[len(hist) // 2]) if hist else 0.0
        lost = a1 < max(16, 0.1 * ref)
        fr = grid(m1)
        M = fr >= theta
        T = fr >= 0.5
        if not bool(T.any()):
            T = M
        c1 = cent(m1)
        c2 = cent(idx2 == o) if idx2 is not None else None
        dv = (c1[0] - c2[0], c1[1] - c2[1]) if (c1 is not None and c2 is not None) else (0.0, 0.0)
        core = (M | shift(M, int(round(dv[0])), int(round(dv[1])))) if motion else M.clone()
        if lost:
            s["n"] += 1
        else:
            s["areas"].append(a1)
            s["c"], s["v"], s["n"], s["M"] = c1, dv, 0, M
        out.append({"id": o, "M": M, "T": T, "core": core, "speed": math.hypot(*dv), "lost": lost, "n": s["n"],
                    "c": s["c"], "v": s["v"], "Mlast": s.get("M")})
    return out, bool(out) and all(x["lost"] for x in out)
