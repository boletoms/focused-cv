"""R40 plan section K (owner 2026-11-21, "continue with recommended", after R-35): the redesign and its fallback under
CUDA graphs, with the method unchanged. Same tiers, same merges, same order as `expect.ExpectMap` around
`baselines.ALGMTiered` (ALGM's rules outside the high tier, our budget `methods.InterestToMe` on the low tier) and
plain `baselines.ALGM` (the fallback); only the execution differs:
1. The tier decision runs before the encoder (`plan`): with pixel features (`exp-pix-...`) the expectation step reads the
   frame's pixels and the object states only, so running it before block 0 gives the tiers it gives after block 0's
   attention, where `ExpectMap.after_attn` runs it.
2. Tokens live in fixed-capacity buffers: the real patch-token count n is a device value; rows past it are padding that
   attention masks out (zero weight) and that no merge can select or receive. Merges are written without host syncs
   or host-to-device copies (masks, sorts, cumulative sums in place of `nonzero`, `unique` and Python branches on
   device values; no index writes of Python constants), so the same
   tokens merge in the same order as the current code.
3. The forward pass is cut after each block's merges. Segment b runs block b − 1's MLP, block b's attention and block b's
   merges on capacity c (n rounded up to a multiple of `step`); the host reads the new count once per block and replays
   the graph captured for (b, c). Segment 0 starts from the pixels, the last segment ends with the final norm and the
   unmerge to every original patch.
Counted FLOPs follow the current bookkeeping (`method.ops`, `State.tokens`, `pairs`, `tiles`); FLOPs at the padded
capacity are recorded in `State.padded`. `graphs=False` runs the same segments eagerly (CPU tests, K1).
Usage: TieredRunner(model, (H, W), fast=...).run(pixels, method, interest) -> (final (1, P + N0, D) float, State)"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.stage_a.model import N_PREFIX, _rot, patch_coords, rope

from .model import State, _up

P = N_PREFIX


@dataclass(frozen=True)
class Spec:
    plain: bool               # plain ALGM (the fallback): no high tier, no budget, layer-1 merge always applied
    threshold: float
    layers: tuple             # ALGM's 1-based layers
    window: int
    high_value: float = math.inf
    srp_value: float = 0.5
    theta: float = 0.01
    rate: float = 0.0
    floor: int = 0


def spec_of(method):
    """The execution spec of an `ExpectMap` around `ALGMTiered` (the redesign) or around plain `ALGM` (its fallback)."""
    from .baselines import ALGM, ALGMTiered
    from .methods import InterestToMe
    inner = method.inner
    assert method.at == 0 and method.cfg.feat == "pix", "the tier decision runs before the encoder only with pixel features"
    if isinstance(inner, ALGMTiered):
        t = inner.tail
        assert type(t) is InterestToMe and t.first == 0 and t.last >= 99 and not t.proportional and t.theta == inner.theta
        assert not inner.proportional
        return Spec(False, inner.threshold, tuple(inner.layers), inner.window, inner.high_value, inner.srp_value,
                    inner.theta, t.rate, t.floor)
    assert type(inner) is ALGM and not inner.proportional, f"no graph path for {inner.name}"
    return Spec(True, inner.threshold, tuple(inner.layers), inner.window)


def plan(method, pixels, interest):
    """The wrapper's start and expectation step before the encoder, on the State the encoder starts from (identity
    source map). Leaves the tiers in `st.interest` (1, N0) and the expectation's FLOPs in `method.ops`."""
    H, W = pixels.shape[-2:]
    gh, gw = H // 16, W // 16
    N0, dev = gh * gw, pixels.device
    st = State(size=torch.ones(1, N0, device=dev), coords=patch_coords(gh, gw, dev)[None],
               src=torch.arange(N0, device=dev)[None].clone(), carried=None, pairs=[], tokens=[], tiles=[],
               interest=None if interest is None else interest.to(dev).float())
    st.coords0, st.interest0, st.grid, st.pixels = st.coords, st.interest, (gh, gw), pixels
    method.start(st)
    method._step(0, None, None, st)
    return st


# ---- merges on padded buffers (n: device count of real patch tokens; c: capacity, host) ----

def _merge_weighted(x, size, coords, interest, src, assign, c):
    """`baselines.merge_by_assignment` on capacity c: assign (c,) -> 0..m-1 for real rows, c (discarded) for padding.
    Size-weighted means; rows past m come out zero."""
    w = size
    D = x.shape[-1]
    assign = assign.clamp(max=c)                                      # an overflowing frame (redone) can count past c
    vals = torch.cat([x[0, P:].float() * w[:, None], coords * w[:, None], (interest * w)[:, None], w[:, None]], 1)
    acc = torch.zeros(c + 1, D + 4, device=x.device).index_add_(0, assign, vals)[:c]   # one scatter, same sums per column
    tot = acc[:, D + 3]
    ok = tot > 0
    out = torch.where(ok[:, None], acc[:, :D + 3] / torch.where(ok, tot, torch.ones_like(tot))[:, None], 0.0)
    xn = torch.cat([x[:, :P], out[None, :, :D].to(x.dtype)], 1)
    return xn, tot, out[:, D:D + 2], out[:, D + 2], assign[src.clamp(max=c - 1)]   # sizes are the weights


def _algm_windows(x, n, interest, spec, grid, high):
    """ALGM's layer-1 merge (2 x 2 windows above the threshold; windows touching the high tier excluded). Runs on the
    full grid (c = n = N0). Returns (assign, any merge)."""
    gh, gw = grid
    T = x.shape[1] - P
    ws = spec.window
    hh, ww = gh // ws * ws, gw // ws * ws
    feat = x[0, P:].float().view(gh, gw, -1)[:hh, :ww]
    win = feat.reshape(hh // ws, ws, ww // ws, ws, -1).permute(0, 2, 1, 3, 4).reshape(-1, ws * ws, feat.shape[-1])
    sims = F.cosine_similarity(win[:, :, None, :], win[:, None, :, :], dim=-1) * (1 - torch.eye(ws * ws, device=x.device))
    sim_map = sims.sum((-1, -2)) / ((ws * ws) * (ws * ws - 1))
    ids = torch.arange(T, device=x.device).view(gh, gw)
    sub = ids[:hh, :ww].reshape(hh // ws, ws, ww // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
    merge = sim_map >= spec.threshold
    if high is not None:
        merge = merge & ~high[sub].any(-1)
    a = torch.arange(T, device=x.device)
    a[sub.flatten()] = torch.where(merge[:, None], sub[:, -1:].expand(-1, ws * ws), sub).flatten()
    present = torch.zeros(T, dtype=torch.bool, device=x.device)
    present.index_fill_(0, a, True)                                   # torch.unique(a, return_inverse=True) by ranks
    new = torch.cumsum(present.long(), 0) - 1
    return new[a], merge.any()


def _algm_thr(node_max, va, layer):
    """`node_max.mean() + node_max.std() / layer` over the real rows, with the current path's roundings (R-38): under
    autocast the scores are bf16 and mean and std keep that dtype (each computed in fp32, rounded once), while sum and
    pow are autocast to fp32, so the threshold is built in fp32 and each statistic is cast back to the scores' dtype."""
    na = va.sum()
    v = node_max.float()
    mean = v.masked_fill(~va, 0.0).sum() / na
    std = ((v - mean).masked_fill(~va, 0.0) ** 2).sum().div(na - 1).sqrt()
    return mean.to(node_max.dtype) + std.to(node_max.dtype) / layer


def _algm_match(x, n, layer, high):
    """ALGM's later merge (turbo matching: even tokens join their most similar odd token when above mean + std / layer of
    the best scores; pairs touching the high tier excluded). Returns (assign, any merge, merged count)."""
    c = x.shape[1] - P
    dev = x.device
    metric = F.normalize(x[0, P:].float(), dim=-1)
    A, Bi = torch.arange(0, c, 2, device=dev), torch.arange(1, c, 2, device=dev)
    va, vb = A < n, Bi < n
    scores = (metric[A] @ metric[Bi].T).masked_fill(~vb[None, :], -math.inf)
    node_max = scores.max(-1).values
    thr = _algm_thr(node_max, va, layer)                               # same threshold rule as ALGM
    if high is not None:
        scores = scores.masked_fill(high[A][:, None] | high[Bi][None, :], -math.inf)
    node_max, node_idx = scores.max(-1)
    srcm = va & (node_max >= thr)
    a = torch.arange(c, device=dev)
    a[A] = torch.where(srcm, Bi[node_idx], A)
    keep = torch.arange(c, device=dev) < n
    keep[A] = keep[A] & ~srcm
    new = torch.cumsum(keep.long(), 0) - 1
    assign = torch.where(torch.arange(c, device=dev) < n, new[a], torch.full_like(a, c))
    k = srcm.sum()
    return assign, k > 0, n - k


def _budget(x, size, coords, interest, src, n, metric, spec):
    """Our budget on the low tier (`InterestToMe` through `ToMe.after_attn`, no proportional attention): r =
    min(floor(rate x low), low even, low odd, low − floor) merges among low tokens; ToMe's order after merging
    (unmerged even tokens by score, then the odd tokens). Returns the new buffers, n and r (device)."""
    c = x.shape[1] - P
    dev = x.device
    valid = torch.arange(c, device=dev) < n
    low = (interest < spec.theta) & valid
    la, lb, nl = low[0::2].sum(), low[1::2].sum(), low.sum()
    r = torch.minimum(torch.minimum(torch.floor(nl.double() * spec.rate).long(), torch.minimum(la, lb)),
                      (nl - spec.floor).clamp_min(0))
    pen = torch.where(low, torch.zeros_like(interest), torch.full_like(interest, 1e4))   # no host scalars in capture
    va, vb = valid[0::2], valid[1::2]
    na_c, nb_c = va.shape[0], vb.shape[0]
    scores = metric[0::2] @ metric[1::2].T - torch.maximum(pen[0::2, None], pen[None, 1::2])
    scores = scores.masked_fill(~vb[None, :], -math.inf).masked_fill(~va[:, None], -math.inf)
    node_max, node_idx = scores.max(-1)
    j = torch.arange(na_c, device=dev)                                  # stable descending order without a sort
    ahead = (node_max[None, :] > node_max[:, None]) | ((node_max[None, :] == node_max[:, None]) & (j[None, :] < j[:, None]))
    rank = ahead.sum(1)                                                 # (plan L: one fused kernel in place of a
    edge = torch.empty_like(rank).scatter_(0, rank, j)                  # radix sort per block; the same permutation)
    nu = (n + 1) // 2 - r
    merged = rank < r
    new_a = torch.where(merged, nu + node_idx, rank - r)
    new_a = torch.where(va, new_a, torch.full_like(new_a, c))
    new_b = torch.where(vb, nu + torch.arange(nb_c, device=dev), torch.full((nb_c,), c, device=dev))
    assign = torch.empty(c, dtype=torch.long, device=dev)
    assign[0::2], assign[1::2] = new_a, new_b
    xn, sz, co, it, sr = _merge_weighted(x, size, coords, interest, src, assign, c)
    j = torch.arange(c, device=dev)                                     # unmerged even tokens are copied, as ToMe
    raw = 2 * edge[(r + j).clamp(max=na_c - 1)]
    single = j < nu
    feat = torch.where(single[:, None], x[0, P:][raw], xn[0, P:])
    xn = torch.cat([x[:, :P], feat[None]], 1)
    co = torch.where(single[:, None], coords[raw], co)
    it = torch.where(single, interest[raw], it)
    sz = torch.where(single, size[raw], sz)
    do = r > 0
    pick = lambda a_, b_: torch.where(do if a_.dim() == 0 else do.view(*([1] * a_.dim())), a_, b_)
    return pick(xn, x), pick(sz, size), pick(co, coords), pick(it, interest), pick(sr, src), n - r, r


def merges(b, x, k, size, coords, interest, src, n, spec, grid):
    """Block b's merges as the current code (`ALGMTiered.after_attn` then its budget, or plain `ALGM.after_attn`).
    Returns the new buffers, n, and per-block counts (n after ALGM, ALGM merged anything, budget r)."""
    layer = b + 1
    zero = torch.zeros((), dtype=torch.long, device=x.device)
    if spec.plain:
        if layer not in spec.layers:
            return x, size, coords, interest, src, n, torch.stack([n, zero, zero])
        if layer == spec.layers[0]:
            assign, anym = _algm_windows(x, n, interest, spec, grid, None)
            m = assign.max() + 1
            x, size, coords, interest, src = _merge_weighted(x, size, coords, interest, src, assign, x.shape[1] - P)
            return x, size, coords, interest, src, m, torch.stack([m, torch.ones_like(zero), zero])
        assign, anym, m = _algm_match(x, n, layer, None)
        out = _merge_weighted(x, size, coords, interest, src, assign, x.shape[1] - P)
        sel = lambda a_, b_: torch.where(anym.view(*([1] * a_.dim())) if a_.dim() else anym, a_, b_)
        x, size, coords, interest, src = (sel(o, i) for o, i in zip(out, (x, size, coords, interest, src)))
        return x, size, coords, interest, src, m, torch.stack([m, anym.long(), zero])
    c = x.shape[1] - P
    raw_metric = F.normalize(k[0, :, P:].float().mean(0), dim=-1)
    metric, n1, anym = raw_metric, n, torch.zeros((), dtype=torch.bool, device=x.device)
    if layer in spec.layers:
        high = interest >= spec.high_value
        mid = (interest >= spec.theta) & ~high
        if layer == spec.layers[0]:
            assign, anym = _algm_windows(x, n, interest, spec, grid, high)
            m = assign.max() + 1
        else:
            assign, anym, m = _algm_match(x, n, layer, high)
        xm, sm, cm, im, srm = _merge_weighted(x, size, coords, interest, src, assign, c)
        tracked = torch.zeros(c + 1, raw_metric.shape[-1], device=x.device).index_add_(0, assign, raw_metric * size[:, None])[:c]
        has_mid = torch.zeros(c + 1, device=x.device).index_add_(0, assign, mid.float())[:c] > 0
        im = torch.where(has_mid & (im < spec.high_value), torch.full_like(im, spec.srp_value), im)
        sel = lambda a_, b_: torch.where(anym.view(*([1] * a_.dim())) if a_.dim() else anym, a_, b_)
        x, size, coords, interest, src = sel(xm, x), sel(sm, size), sel(cm, coords), sel(im, interest), sel(srm, src)
        metric = sel(F.normalize(tracked, dim=-1), raw_metric)
        n1 = torch.where(anym, m, n)
    x, size, coords, interest, src, n2, r = _budget(x, size, coords, interest, src, n1, metric, spec)
    return x, size, coords, interest, src, n2, torch.stack([n1, anym.long(), r])


# ---- the encoder on padded buffers ----

try:
    from torch.nn.attention.varlen import varlen_attn                 # module scope: traceable inside torch.compile
except ImportError:                                                   # older PyTorch: the masked kernel only
    varlen_attn = None


def _varlen(q, k, v, used):
    """Flash attention (the kernel the current path's unmasked SDPA call runs) over the whole buffer of queries, reading
    only the first `used` keys (device value): no mask tensor and no host value, so it captures in a graph."""
    B, H, T, hd = q.shape
    cu = torch.arange(2, dtype=torch.int32, device=q.device) * T
    qs, ks, vs = (t[0].transpose(0, 1).contiguous() for t in (q, k, v))
    used = used.clamp(max=T)                                          # an overflowing frame (redone) counts past T
    o = varlen_attn(qs, ks, vs, cu, cu, T, T, seqused_k=used.to(torch.int32).view(1))
    return o.transpose(0, 1)[None]


def attention(model, b, x, coords, n, fast, kernel="mask"):
    """Block b's attention and residual (`EffModel.block_b`, no method bias), keys past P + n masked out. Fast path:
    kernel 'mask' (SDPA with a padding mask), 'varlen' (flash, keys past P + n not read) or 'dense' (plain SDPA, only
    when the buffer holds no padding)."""
    layer = model.bb.model.layer[b]
    B, T, D = x.shape
    H, hd = model.heads, model.hd
    att = layer.attention
    y = layer.norm1(x)
    q, k, v = (p(y).view(B, T, H, hd).transpose(1, 2) for p in (att.q_proj, att.k_proj, att.v_proj))
    cos, sin = rope(coords[None], model.inv_freq)
    cos, sin = cos[:, None], sin[:, None]
    q = torch.cat([q[:, :, :P], _rot(q[:, :, P:], cos.to(q.dtype), sin.to(q.dtype))], 2)
    k = torch.cat([k[:, :, :P], _rot(k[:, :, P:], cos.to(k.dtype), sin.to(k.dtype))], 2)
    pad = torch.arange(T, device=x.device) >= P + n
    if fast and kernel == "varlen":
        o = _varlen(q, k, v, P + n)
    elif fast and kernel == "dense":
        o = F.scaled_dot_product_attention(q, k, v)
    elif fast:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=~pad[None, None, None])
    else:
        with torch.autocast("cuda", enabled=False):
            s = (q.float() @ k.float().transpose(-1, -2)) * att.scaling
            s = s.masked_fill(pad, -math.inf)
            s -= s.amax(-1, keepdim=True)
            s.exp_()
            s /= s.sum(-1, keepdim=True)
            o = s @ v.float()
    o = o.transpose(1, 2).reshape(B, T, D)
    return x + layer.layer_scale1(att.o_proj(o.to(y.dtype))), k


def mlp(model, b, x):
    layer = model.bb.model.layer[b]
    return x + layer.layer_scale2(layer.mlp(layer.norm2(x)))


def seg_core(model, spec, grid, fast, kernel, b, x, size, coords, interest, src, n, o, allc, mlp_first):
    """One block segment after its input (`TieredRunner._segment`): the MLP of block b − 1 (when mlp_first), block b's
    attention and merges, and the writes to the output bank; one function, so `compiled` fuses across it."""
    if mlp_first:
        x = mlp(model, b - 1, x)
    x, k = attention(model, b, x, coords, n, fast, kernel)
    x, size, coords, interest, src, n, counts = merges(b, x, k, size, coords, interest, src, n, spec, grid)
    c = x.shape[1] - P
    o["x"][:, :P + c].copy_(x)
    o["size"][:c].copy_(size)
    o["coords"][:c].copy_(coords)
    o["interest"][:c].copy_(interest)
    o["src"].copy_(src)
    o["n"].copy_(n)
    o["counts"].copy_(counts)
    allc[b].copy_(counts)


def seg0_attn(model, fast, kernel, x, coords, n, x0, k0):
    """Block 0's attention after the patch embedding (independent of the tiers), into fixed buffers."""
    x, k = attention(model, 0, x, coords, n, fast, kernel)
    x0.copy_(x)
    k0.copy_(k)


def seg0_merge(spec, grid, x, k, size, coords, interest, src, n, o, allc):
    """Block 0's merges on the tiers, written to bank 1."""
    x, size, coords, interest, src, n, counts = merges(0, x, k, size, coords, interest, src, n, spec, grid)
    c = x.shape[1] - P
    o["x"][:, :P + c].copy_(x)
    o["size"][:c].copy_(size)
    o["coords"][:c].copy_(coords)
    o["interest"][:c].copy_(interest)
    o["src"].copy_(src)
    o["n"].copy_(n)
    o["counts"].copy_(counts)
    allc[0].copy_(counts)


_COMPILED = {}


def compiled(fn):
    """Plan section L, avenue 3: `fn` through `torch.compile` (Inductor fuses element-wise work and reductions into
    fewer kernels; dynamic shapes, so one compilation serves every capacity of a block), captured inside the block
    graphs like the eager kernels. Same operations; rounding may differ (R-37)."""
    if fn not in _COMPILED:                                           # R40_COMPILE_BACKEND=aot_eager: CPU tests
        import os
        import torch._dynamo.config as dc
        for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(dc, name):                                     # one specialisation per block and spec
                setattr(dc, name, max(getattr(dc, name), 1024))
        import torch._inductor.config as ic
        if os.environ.get("R40_INDUCTOR_COMBO") == "1":               # plan L: horizontal fusion of small kernels
            ic.combo_kernels = True
        for kv in filter(None, os.environ.get("R40_INDUCTOR_OPTS", "").split(",")):   # e.g. aggressive_fusion=1
            k, v = kv.split("=")
            setattr(ic, k, type(getattr(ic, k))(int(v)) if isinstance(getattr(ic, k), (bool, int)) else v)
        mode = os.environ.get("R40_COMPILE_MODE")                     # e.g. max-autotune-no-cudagraphs
        _COMPILED[fn] = torch.compile(fn, dynamic=True, fullgraph=True, mode=mode,
                                      backend=os.environ.get("R40_COMPILE_BACKEND", "inductor"))
    return _COMPILED[fn]


class TieredRunner:
    """Runs the redesign (or its fallback) on padded buffers, eagerly or with one CUDA graph per (block, capacity)."""

    def __init__(self, model, hw, step=32, fast=True, dtype=torch.bfloat16, graphs=True, warmup=2, kernel="mask",
                 compile=False, overlap=False):
        self.model, self.step, self.fast, self.dtype, self.graphs, self.warmup = model, step, fast, dtype, graphs, warmup
        assert kernel != "dense" or (step == 1 and not graphs), "the plain SDPA call needs buffers without padding"
        self.kernel = kernel
        self.f_att, self.f_mlp, self.f_merge = ((compiled(attention), compiled(mlp), compiled(merges)) if compile
                                                else (attention, mlp, merges))
        self.f_seg = compiled(seg_core) if compile == "segment" else None
        self.f_s0a, self.f_s0m = (compiled(seg0_attn), compiled(seg0_merge)) if compile else (seg0_attn, seg0_merge)
        # plan L: block 0's embedding and attention on a side stream while the tiers are planned (graphs only; its own
        # memory pool, since it runs beside the other graphs); block 0's merges then wait for it
        self.overlap = overlap and graphs
        self.side = torch.cuda.Stream() if self.overlap else None
        self.pool0 = torch.cuda.graph_pool_handle() if self.overlap else None
        self.ev0, self.g0 = None, None
        self.grid = (hw[0] // 16, hw[1] // 16)
        self.N0 = self.grid[0] * self.grid[1]
        self.blocks = len(model.bb.model.layer)
        self.cache = {}                                    # (spec, b, c) -> (graph, outputs)
        self.buf = None
        self.pool = torch.cuda.graph_pool_handle() if graphs else None

    def cls(self, n):
        return min(self.N0, -(-n // self.step) * self.step)

    def _alloc(self, x0, dev):
        N0, D = self.N0, x0.shape[-1]
        bank = lambda: {"x": torch.zeros(1, P + N0, D, device=dev, dtype=x0.dtype), "size": torch.zeros(N0, device=dev),
                        "coords": torch.zeros(N0, 2, device=dev), "interest": torch.zeros(N0, device=dev),
                        "src": torch.zeros(N0, dtype=torch.long, device=dev), "n": torch.zeros((), dtype=torch.long, device=dev),
                        "counts": torch.zeros(3, dtype=torch.long, device=dev)}
        H, hd = self.model.heads, self.model.hd
        self.buf = {"pixels": None, "interest0": torch.zeros(N0, device=dev),
                    "x0": torch.zeros(1, P + N0, D, device=dev, dtype=x0.dtype),
                    "k0": torch.zeros(1, H, P + N0, hd, device=dev, dtype=x0.dtype),
                    "banks": [None] + [bank() for _ in range(self.blocks)],   # bank b: block b's input, one per block,
                    "final": torch.zeros(1, P + N0, D, device=dev),          # so an overflow resumes at its block
                    "allcounts": torch.zeros(self.blocks, 3, dtype=torch.long, device=dev)}

    def _segment(self, spec, b, c):
        """Segment b on capacity c: (MLP of block b − 1,) attention and merges of block b; reads bank b % 2 (or the
        pixels), writes bank (b + 1) % 2. b == blocks: the last MLP, the final norm and the unmerge."""
        m, buf = self.model, self.buf
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            if b == -1 or (b == 0 and self.overlap):
                N0, dev = self.N0, buf["x0"].device
                coords, n = patch_coords(*self.grid, dev), torch.full((), N0, dtype=torch.long, device=dev)
                if b == -1:
                    self.f_s0a(m, self.fast, self.kernel, m.bb.embeddings(buf["pixels"]), coords, n, buf["x0"], buf["k0"])
                else:
                    self.f_s0m(spec, self.grid, buf["x0"], buf["k0"], torch.ones(N0, device=dev), coords, buf["interest0"],
                               torch.arange(N0, device=dev), n, buf["banks"][1], buf["allcounts"])
                return
            if b == 0:
                px = buf["pixels"]
                x = m.bb.embeddings(px)
                size = torch.ones(c, device=x.device)
                coords = patch_coords(*self.grid, x.device)
                interest, src = buf["interest0"], torch.arange(c, device=x.device)
                n = torch.full((), c, dtype=torch.long, device=x.device)
            else:
                i = buf["banks"][b]
                x, size, coords, interest, src, n = (i["x"][:, :P + c], i["size"][:c], i["coords"][:c], i["interest"][:c],
                                                     i["src"], i["n"])
            if self.f_seg is not None and b < self.blocks:                   # plan L avenue 3: the whole segment compiled
                self.f_seg(m, spec, self.grid, self.fast, self.kernel, b, x, size, coords, interest, src, n,
                           buf["banks"][b + 1], buf["allcounts"], b > 0)
                return
            if b > 0:
                x = self.f_mlp(m, b - 1, x)
            if b == self.blocks:
                out = m.bb.norm(x)
                patches = torch.gather(out, 1, (src.clamp(max=c - 1) + P)[None, :, None].expand(-1, -1, out.shape[-1]))
                buf["final"].copy_(torch.cat([out[:, :P], patches], 1).float())
                return
            x, k = self.f_att(m, b, x, coords, n, self.fast, self.kernel)
            x, size, coords, interest, src, n, counts = self.f_merge(b, x, k, size, coords, interest, src, n, spec, self.grid)
            o = buf["banks"][b + 1]
            o["x"][:, :P + c].copy_(x)
            o["size"][:c].copy_(size)
            o["coords"][:c].copy_(coords)
            o["interest"][:c].copy_(interest)
            o["src"].copy_(src)
            o["n"].copy_(n)
            o["counts"].copy_(counts)
            buf["allcounts"][b].copy_(counts)

    def _run_segment(self, spec, b, c):
        if not self.graphs:
            return self._segment(spec, b, c)
        key = (spec, b, c)
        g = self.cache.get(key)
        if g is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(self.warmup):
                    self._segment(spec, b, c)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool), torch.no_grad():
                self._segment(spec, b, c)
            self.cache[key] = g
        g.replay()

    def _run_attn0_inline(self):
        self.side.wait_stream(torch.cuda.current_stream())
        self.launch_attn0(self.buf["pixels"])
        self._launched = False
        torch.cuda.current_stream().wait_event(self.ev0)

    def _ensure(self, pixels):
        if self.buf is None:
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                x0 = self.model.bb.embeddings(pixels)
            self._alloc(x0, pixels.device)
            self.buf["pixels"] = pixels.clone()

    @torch.no_grad()
    def launch_attn0(self, pixels):
        """Plan L (overlap): the frame into the pixel buffer, then block 0's embedding and attention replayed on the side
        stream; `encode` skips its own pixel copy and makes block 0's merges wait for it."""
        if not self.overlap:
            return
        self._ensure(pixels)
        self.buf["pixels"].copy_(pixels)
        if self.g0 is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(self.warmup):
                    self._segment(None, -1, self.N0)
            torch.cuda.current_stream().wait_stream(s)
            self.g0 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g0, pool=self.pool0):
                self._segment(None, -1, self.N0)
            self.ev0 = torch.cuda.Event()
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            self.g0.replay()
            self.ev0.record(self.side)
        self._launched = True

    @torch.no_grad()
    def run(self, pixels, method, interest):
        """pixels (1, 3, H, W); method: an ExpectMap (main or passive fallback); interest: the evaluator's map or None.
        Returns (final (1, P + N0, D) float, State with tokens, pairs, tiles, padded (tokens at capacity))."""
        spec = spec_of(method)
        st = plan(method, pixels, interest)
        assert spec.plain or st.interest is not None, "the redesign without a map runs dense (not on this path)"
        tiers = st.interest[0] if st.interest is not None else torch.zeros(self.N0, device=pixels.device)
        return self.encode(pixels, spec, tiers, method.ops, st)

    def encode(self, pixels, spec, tiers, ops, st, caps=None):
        """The encoder on the planned tiers (N0,). caps None: capacities from the count read after each block (K).
        caps (blocks + 1 host ints, caps[0] = N0): plan section L, capacities chosen before the frame, one read of
        every block's counts at the end; from the first block whose real input exceeded its capacity on, the blocks
        run again with a read after each (`self.overflow`). FLOPs bookkeeping appends to `ops` and `st`."""
        self._ensure(pixels)
        launched, self._launched = getattr(self, "_launched", False), False
        if launched:                                                       # block 0's attention is on the side stream
            self.buf["interest0"].copy_(tiers)
            torch.cuda.current_stream().wait_event(self.ev0)
        else:
            if self.overlap:                                               # (not launched: run it here, in order)
                self.buf["pixels"].copy_(pixels)
                self._run_attn0_inline()
            else:
                self.buf["pixels"].copy_(pixels)
            self.buf["interest0"].copy_(tiers)
        D, hd, ws = self.buf["final"].shape[-1], self.model.hd, spec.window
        good = 0                                                           # blocks whose input fitted its capacity
        self.overflow = False
        if caps is not None:
            for b in range(self.blocks + 1):
                self._run_segment(spec, b, caps[b])
            allc = self.buf["allcounts"].tolist()                        # the one read of the frame's counts
            n = self.N0
            while good <= self.blocks and n <= caps[good]:
                if good < self.blocks:
                    n = allc[good][0] - allc[good][2]
                good += 1
            self.overflow = good <= self.blocks                         # block `good` got more tokens than its capacity:
        st.tokens, st.pairs, st.tiles, st.padded = [], [], [], []       # from it on, capacities from the counts read
        n, c = self.N0, self.N0                                          # after each block (bank `good` holds its input)
        for b in range(self.blocks):
            if b < good:
                n1, anym, r = allc[b]
            else:
                if b == good and b > 0:
                    c = self.cls(n)
                self._run_segment(spec, b, c)
                n1, anym, r = self.buf["banks"][b + 1]["counts"].tolist()   # one host read per block from here on
            n_out = n1 - r
            layer = b + 1
            if layer in spec.layers:                                   # FLOPs as the current bookkeeping
                if layer == spec.layers[0]:
                    ops.append((b, "clap", 2 * n * ws * ws * D))
                else:
                    ops.append((b, "gbm", 2 * ((n + 1) // 2) * (n // 2) * D))
                if anym:
                    ops.append((b, "merge", 2 * n * D))
            if not spec.plain and r > 0:
                ops.append((b, "tome_match", 2 * ((n1 + 1) // 2) * (n1 // 2) * hd))
                ops.append((b, "tome_merge", 4 * r * D))
            T = P + n
            st.tokens.append((T, P + n_out))
            st.pairs.append([float(T * T)])                          # plain lists (`image_flops` reads [i]):
            st.tiles.append([float(_up(T) * _up(T))])                # no tensor creation on the host per block
            c_next = caps[b + 1] if b + 1 < good else self.cls(n_out)
            st.padded.append((P + c, P + c_next))
            n, c = n_out, c_next
        if good <= self.blocks:
            self._run_segment(spec, self.blocks, c)
        o = self.buf["banks"][self.blocks]
        st.src = o["src"][None].clone()
        st.size = o["size"][:n][None].clone()
        return self.buf["final"].clone(), st
