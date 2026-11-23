"""R40 plan section L (implementation only, method unchanged): one frame of the redesign as the evaluator and the timing
protocol run it (map motion, object states, the main-or-fallback decision, the map's margin and size classes, the
expectation step, the encoder), with two host reads per frame: the decision (with the high count the size classes
need) and the encoder's per-block token counts at the end. The encoder's capacities come from the previous frame's
counts plus a margin (`TieredRunner.encode(caps=...)`); from the first block whose real count exceeds its capacity,
the blocks run again with a read after each (plan section K), so outputs never depend on the guess.
Usage: fr = FrameRunner(model, (H, W), setting); fr.reset() per video; final, st, ops, fallback = fr.run(pixels, ids,
idx1, idx2[, base, new]) (see `run`). FLOPs: `flops.image_flops(st, ops, 0, N0)`. The evaluators reach it through
`vos.dino_encoder` with VOS_GRAPH=2."""
import torch
import torch.nn.functional as F

from .expect_dev import DevObjects, ExpectDev, dilate_map, moved_union, promote_count, promote_map
from .model import State
from .tiered_graph import P, TieredRunner, spec_of


def coverage(m):
    """(H, W) bool -> (1, N) patch coverage (`vos.r40_timing.cov`)."""
    return F.avg_pool2d(m.float()[None, None], 16)[0, 0].flatten()[None]


def pre_step(dobj, idx1, idx2, ids, base, pad, dilate, grid, theta):
    """Object states, map motion, coverage at the encoder's input size, the inner method's margin; returns (objs, the
    margin map (1, N), flags (map not empty, all lost, high count))."""
    if idx1 is not None:
        objs, all_lost = dobj.update(idx1, idx2, ids)
    else:
        objs, all_lost = {}, torch.zeros((), dtype=torch.bool, device=base.device)
    mp = base
    if idx1 is not None and idx2 is not None:
        mp = mp | moved_union(idx1, idx2, dobj.idt)
    imap = coverage(F.pad(mp.float(), pad))
    dm = dilate_map(imap, dilate, grid)
    return objs, dm, torch.stack([imap.any().long(), all_lost.long(), (dm[0] >= theta).sum()])


class FrameRunner:
    def __init__(self, model, hw, setting, step=32, margin=32, fast=True, dtype=torch.bfloat16, graphs=True,
                 kernel="mask", pre_graphs=None, compile=False, overlap=False):
        from .expect import ExpectMap, ExpectState, config_from_name
        from .timing import from_name
        inner, rest = setting.split("~")
        fb, exp = rest.split("@")
        self.cfg = config_from_name(exp)
        blocks = len(model.bb.model.layer)
        self.grid = (hw[0] // 16, hw[1] // 16)
        self.N0 = self.grid[0] * self.grid[1]
        es = ExpectState()
        main = ExpectMap(from_name(inner, self.grid[0], blocks), es, self.cfg)
        fbm = ExpectMap(from_name(fb, self.grid[0], blocks), es, self.cfg, passive=True)
        self.spec = {False: spec_of(main), True: spec_of(fbm)}
        self.inner = main.inner
        self.model, self.step, self.margin, self.dtype = model, step, margin, dtype
        self.runner = TieredRunner(model, hw, step, fast, dtype, graphs, kernel=kernel, compile=compile, overlap=overlap)
        from .tiered_graph import compiled
        self.f_pre, self.f_step = (compiled(pre_step), compiled(ExpectDev.step)) if compile else (pre_step, ExpectDev.step)
        self.dev = next(model.parameters()).device
        self.exd = ExpectDev(self.cfg, self.inner.theta, self.inner.srp_value, self.grid, self.dev)
        self.pre_graphs = graphs if pre_graphs is None else pre_graphs          # the steps before the encoder in graphs
        self.pg, self.inb = {}, {}
        self.marks = None                                                         # a list: CUDA events per stage
        self.coords0 = torch.zeros(1, self.N0, 2, device=self.dev)                # for the evaluators' patch count
        self.zeros = torch.zeros(self.N0, device=self.dev)                        # fallback frames' tiers (unused)
        self.dobj = DevObjects(self.cfg.motion)
        self.reset()

    def reset(self):
        self.exd.reset()
        if self.dobj.s is not None:                                       # new video: object states start empty
            for t in self.dobj.state():
                t.zero_()
        self.last = {False: None, True: None}                     # previous frame's block input counts, per path
        self.stats = {"frames": 0, "fallback": 0, "overflow": 0}

    def caps(self, fb):
        prev = self.last[fb]
        if prev is None or self.margin is None:
            return None
        return [self.N0] + [min(self.N0, max(1, -(-(n + self.margin) // self.step)) * self.step) for n in prev[1:]]

    def _call(self, key, fn):
        """fn() eagerly, or replayed from a CUDA graph captured for `key` (inputs in fixed buffers). Warm-up runs fn on a
        side stream, so the device state and the host flag it advances are restored before the capture."""
        if not self.pre_graphs:
            return fn()
        if key not in self.pg:
            state = self.dobj.state() + self.exd.state()
            snap, flag = [t.clone() for t in state], self.exd.has_prev
            sm = torch.cuda.Stream()
            sm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(sm):
                fn()
            torch.cuda.current_stream().wait_stream(sm)
            for t, c in zip(state, snap):
                t.copy_(c)
            self.exd.has_prev = flag
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.runner.pool):
                out = fn()
            self.pg[key] = (g, out)
        g, out = self.pg[key]
        g.replay()
        return out

    def _fixed(self, name, t):
        """Copy t into a buffer kept per (name, shape, dtype), so a graph that reads another graph's output reads the
        same memory whichever graph wrote it (allocated on the eager warm-up, before any capture)."""
        key = (name, tuple(t.shape), t.dtype)
        b = self.inb.get(key)
        if b is None:
            b = self.inb[key] = torch.zeros_like(t)
        b.copy_(t)
        return b

    def _inputs(self, *ts):
        """Fixed input buffers for the graphs (copies); eager mode uses the tensors as given. None stays None."""
        if not self.pre_graphs:
            return ts
        out = []
        for i, t in enumerate(ts):
            if t is None:
                out.append(None)
                continue
            key = ("in", i, tuple(t.shape), t.dtype)
            if key not in self.inb:
                self.inb[key] = t.clone()
            out.append(self.inb[key].copy_(t))
        return out

    def _mark(self):
        if self.marks is not None:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.marks.append(e)

    @torch.no_grad()
    def run(self, pixels, ids, idx1, idx2, base=None, new=False):
        """pixels (1, 3, H, W); ids: objects seen before t (host list); idx1, idx2: (h, w) id maps at t − 1, t − 2 (None
        when absent; idx1 None: no objects this frame, as the evaluators); base: (h, w) bool map before the motion
        (default idx1 > 0; the evaluators: the previous prediction and any mask given at t), placed in the middle of
        the (H, W) input as Cutie pads; new: a mask is given at t (then not all lost)."""
        th, inner = self.inner.theta, self.inner
        self._mark()                                                            # optional timing marks (profile)
        if base is None:
            base = idx1 > 0
        pixels, idx1, idx2, base = self._inputs(pixels, idx1, idx2, base)
        self.runner.launch_attn0(pixels)                                        # overlap: block 0 runs beside the tiers
        H, W = pixels.shape[-2:]
        h, w = base.shape
        lh, lw = (H - h) // 2, (W - w) // 2
        ids = list(ids) if idx1 is not None else []
        self.dobj.ensure(ids, (h, w), pixels.device)
        ver = self.dobj.version
        for key in [k for k in self.pg if k[0] == "pre" and k[1] != ver]:      # graphs on replaced state rows
            del self.pg[key]

        def pre():
            objs, dm, flags = self.f_pre(self.dobj, idx1, idx2, ids, base, (lw, W - w - lw, lh, H - h - lh),
                                         inner.dilate, self.grid, th)
            return ({k: self._fixed("o_" + k, v) for k, v in objs.items()}, self._fixed("dm", dm),
                    self._fixed("flags", flags))
        objs, dm, flags = self._call(("pre", ver, idx1 is None, idx2 is None, (h, w), (H, W)), pre)
        self._mark()
        flags = flags.tolist()                                                  # read 1: the decision
        lost = bool(flags[1]) and not new
        self.flags = (bool(flags[0]), lost)
        fb = not flags[0] or lost
        ops = []
        K = len(ids)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):   # as the current path
            if fb or not self.exd.has_prev:
                mp = None
                if not fb:
                    k = promote_count(flags[2], inner.size_class, self.N0)
                    mp = self._call(("promote", k), lambda: self._fixed("mp", promote_map(dm, th, inner.size_class, self.grid, k)))
                self._call(("keep", K, fb), lambda: self.f_step(self.exd, pixels, None, objs, True))
                self.exd.has_prev = True
                tiers = self.zeros if fb else mp[0]
            else:
                k = promote_count(flags[2], inner.size_class, self.N0)
                mp = self._call(("promote", k), lambda: self._fixed("mp", promote_map(dm, th, inner.size_class, self.grid, k)))
                def expect():                     # into a fixed buffer: a frame that is redone copies the tiers again,
                    tr, eo = self.f_step(self.exd, pixels, mp, objs, False)     # after encoder graphs that may reuse
                    return self._fixed("tiers", tr), eo                         # pool memory have run
                tiers, eops = self._call(("expect", K), expect)
                tiers = tiers[0]
                ops += [(0, name, fl) for name, fl in eops]
        self._mark()
        st = State(size=None, coords=None, src=None, carried=None, pairs=[], tokens=[], tiles=[], interest=None)
        final, st = self.runner.encode(pixels, self.spec[fb], tiers, ops, st, self.caps(fb))
        self._mark()
        self.stats["overflow"] += self.runner.overflow             # blocks from the first too-small capacity redone
        st.coords0 = self.coords0
        self.last[fb] = [self.N0] + [t - P for _, t in st.tokens]
        self.stats["frames"] += 1
        self.stats["fallback"] += fb
        return final, st, ops, fb
