"""Stage B methods (plan: docs/records/STAGE_B_EFFICIENCY_PLAN.md). Each is a hook object for EffModel.

- ToMe: bipartite soft matching on the keys (mean over heads), merging r pairs of patch tokens per block, between
  attention and MLP, with proportional attention (Bolya et al. 2023).
- Prune: at chosen blocks, keep the patch tokens CLS attends to most (mean over heads) and fuse the rest into one
  token weighted by that attention (EViT, Liang et al. 2022). A pruned patch's output is its feature at pruning.
- Group: per block and head, spherical k-means on the current queries and keys (as Stage A's k-means); each query
  attends to the keys of its best m clusters; a head falls back to dense in a block if, on a fixed sample of 64
  queries, the attention its mask would cut is above tau.
- Oracle: the Stage A per-head teacher's partitions (saved by a2g_calib), as masks. Upper bound: needs dense attention.
- Windows: the Stage A windows with global blocks.
- Combo: grouped attention (any of the above) plus merging or pruning.
Every method's own extra work is recorded through `ops` for the FLOP counter (flops.py)."""
import math

import torch
import torch.nn.functional as F

from src.stage_a.model import N_PREFIX
from src.stage_a.structures import NEG, GLOBAL_BLOCKS, window_ids

SAMPLE = 64


class Method:
    name = "dense"

    def __init__(self):
        self.ops = []            # (block, kind, flops per image)

    def start(self, st):
        self.ops = []


def _patch_bias(allowed, T):
    """allowed (B, H, n, n) bool over patch tokens -> (B, H, T, T) bias; CLS and registers see and are seen by all."""
    B, H, n, _ = allowed.shape
    bias = torch.zeros(B, H, T, T, device=allowed.device)
    bias[:, :, N_PREFIX:, N_PREFIX:] = torch.where(allowed, 0.0, NEG)
    return bias


class ToMe(Method):
    """r: tokens removed per merging block; first: first block (0-based) that merges."""
    proportional = True

    def __init__(self, r, first=0, last=99, prop=True):
        super().__init__()
        self.r, self.first, self.last, self.proportional = r, first, last, prop
        self.name = f"tome_r{r}_from{first + 1}" + ("" if prop else "_noprop")

    def _select(self, scores, st, r):
        """Merge pairs: the r a-tokens with the most similar b partner. Returns (src_a, dst_b, unmerged a)."""
        node_max, node_idx = scores.max(-1)
        edge = torch.sort(node_max, dim=-1, descending=True, stable=True).indices
        src_a, unm_a = edge[:, :r], edge[:, r:]                         # indices into a
        return src_a, torch.gather(node_idx, 1, src_a), unm_a

    def after_attn(self, b, x, k, p, st):
        if not (self.first <= b <= self.last):
            return x
        n = st.size.shape[1]
        r = min(self.r, n // 2)
        if r <= 0:
            return x
        P = N_PREFIX
        tracked = getattr(self, "_metric", None)                        # one-shot merging: features carried across rounds
        metric = tracked if tracked is not None else F.normalize(k[:, :, P:].float().mean(1), dim=-1)   # (B, n, hd)
        a, bb = metric[:, ::2], metric[:, 1::2]
        na, nb = a.shape[1], bb.shape[1]
        scores = a @ bb.transpose(-1, -2)                               # (B, na, nb)
        self.ops.append((b, "tome_match", 2 * na * nb * metric.shape[-1]))
        if st.interest is not None:     # prior: merges touching a in-focus token (either side) come last
            pen = getattr(self, "_penalty", None)
            w = st.interest if pen is None else pen
            scores = scores - torch.maximum(w[:, ::2, None], w[:, None, 1::2])
        src_a, dst_b, unm_a = self._select(scores, st, r)
        r = src_a.shape[1]
        if r == 0:
            return x
        B, D = x.shape[0], x.shape[-1]
        xp = x[:, P:]
        xa, xb = xp[:, ::2], xp[:, 1::2]
        sa, sb = st.size[:, ::2], st.size[:, 1::2]
        ca, cb = st.coords[:, ::2], st.coords[:, 1::2]
        g = lambda t, i: torch.gather(t, 1, i[..., None].expand(-1, -1, t.shape[-1])) if t.dim() == 3 else torch.gather(t, 1, i)
        w_src = g(sa, src_a)
        # weighted sums into the b tokens
        xb_new = (xb.float() * sb[..., None]).scatter_add(1, dst_b[..., None].expand(-1, -1, D), g(xa, src_a).float() * w_src[..., None])
        cb_new = (cb * sb[..., None]).scatter_add(1, dst_b[..., None].expand(-1, -1, 2), g(ca, src_a) * w_src[..., None])
        sb_new = sb.scatter_add(1, dst_b, w_src)
        xb_new, cb_new = xb_new / sb_new[..., None], cb_new / sb_new[..., None]
        self.ops.append((b, "tome_merge", 4 * r * D))
        x_new = torch.cat([x[:, :P], g(xa, unm_a), xb_new.to(x.dtype)], 1)
        if st.interest is not None:
            ia, ib = st.interest[:, ::2], st.interest[:, 1::2]
            ib_new = (ib * sb).scatter_add(1, dst_b, g(ia, src_a) * w_src) / sb_new
            st.interest = torch.cat([g(ia, unm_a), ib_new], 1)
        if tracked is not None:
            ma, mb = tracked[:, ::2], tracked[:, 1::2]
            mb_new = (mb * sb[..., None]).scatter_add(1, dst_b[..., None].expand(-1, -1, mb.shape[-1]), g(ma, src_a) * w_src[..., None])
            self._metric = torch.cat([g(ma, unm_a), F.normalize(mb_new, dim=-1)], 1)
        st.size = torch.cat([g(sa, unm_a), sb_new], 1)
        st.coords = torch.cat([g(ca, unm_a), cb_new], 1)
        # where every old patch token goes: kept a tokens first, then the b tokens
        nu = na - r
        new_a = torch.empty(B, na, dtype=torch.long, device=x.device)
        new_a.scatter_(1, unm_a, torch.arange(nu, device=x.device)[None].expand(B, -1))
        new_a.scatter_(1, src_a, nu + dst_b)
        new_b = nu + torch.arange(nb, device=x.device)[None].expand(B, -1)
        new_idx = torch.empty(B, n, dtype=torch.long, device=x.device)
        new_idx[:, ::2], new_idx[:, 1::2] = new_a, new_b
        live = st.src >= 0
        st.src = torch.where(live, torch.gather(new_idx, 1, st.src.clamp_min(0)), st.src)
        return x_new


def dilate_interest(st, d):
    """Enlarge the focus map by d patches in every direction (max over a square), on the patch grid."""
    if d and st.interest is not None:
        gh, gw = st.grid if st.grid else (int(round(st.interest.shape[1] ** 0.5)),) * 2
        st.interest = torch.nn.functional.max_pool2d(st.interest.view(-1, 1, gh, gw), 2 * d + 1, 1, d).flatten(1)
        st.interest0 = st.interest


def promote(interest, theta, step, grid):
    """N7 size classes: round the protected count up to a multiple of `step` by protecting extra patches (nearest to
    the protected set first, then by interest), so frames of one class share every shape. Over-coverage costs only
    compute (P1). Returns the new map and the class size. Batch 1."""
    high = interest[0] >= theta
    n, N = int(high.sum()), interest.shape[1]
    c = min(N, -(-n // step) * step)
    if c > n:
        gh, gw = grid
        h = high.float().view(1, 1, gh, gw)
        near = 2 * F.max_pool2d(h, 3, 1, 1) + F.max_pool2d(h, 7, 1, 3)
        score = (near.flatten() + interest[0]).masked_fill(high, -1.0)
        idx = score.topk(c - n).indices
        interest = interest.clone()
        interest[0, idx] = theta
    return interest, c


class Prune(Method):
    """keep: fraction of current patch tokens kept at each pruning block; at: 0-based pruning blocks; restore_at:
    0-based block before which every pruned patch is put back at its position (Token Cropr's last-layer fusion), or None."""

    def __init__(self, keep, at=(3, 6, 9), restore_at=None):
        super().__init__()
        self.keep, self.at, self.restore_at = keep, tuple(at), restore_at
        self.name = f"prune_k{keep:g}_at{'-'.join(str(a + 1) for a in self.at)}" + ("" if restore_at is None else f"_restore{restore_at + 1}")

    def _kept(self, n, st):
        return max(1, round(self.keep * n))

    def _score(self, p, st):
        score = p[:, :, 0, N_PREFIX:].mean(1)                            # (B, n) CLS attention
        if st.interest is not None:                                      # prior: keep by attention times interest
            score = score * st.interest
        return score

    def after_attn(self, b, x, k, p, st):
        if b not in self.at:
            return x
        P = N_PREFIX
        n = st.size.shape[1]
        nk = self._kept(n, st)
        if nk >= n:
            return x
        score = self._score(p, st)
        top = score.topk(nk, dim=-1).indices.sort(-1).values             # kept, in their current order
        drop_mask = torch.ones_like(score, dtype=torch.bool).scatter(1, top, False)
        B, D = x.shape[0], x.shape[-1]
        xp = x[:, P:]
        # carried outputs for every original patch whose current token is dropped
        live = st.src >= 0
        dropped_orig = live & torch.gather(drop_mask, 1, st.src.clamp_min(0))
        feat = torch.gather(xp, 1, st.src.clamp_min(0)[..., None].expand(-1, -1, D))
        st.carried = torch.where(dropped_orig[..., None], feat.to(st.carried.dtype), st.carried)
        # fused token: CLS-attention-weighted mean of the dropped tokens
        w = torch.where(drop_mask, score, 0.0)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        fused = (w[..., None] * xp.float()).sum(1, keepdim=True)
        fc = (w[..., None] * st.coords).sum(1, keepdim=True)
        fs = torch.where(drop_mask, st.size, 0.0).sum(1, keepdim=True)
        self.ops.append((b, "prune_fuse", 2 * (n - nk) * D))
        g = lambda t: torch.gather(t, 1, top[..., None].expand(-1, -1, t.shape[-1]))
        x_new = torch.cat([x[:, :P], g(xp), fused.to(x.dtype)], 1)
        if st.interest is not None:
            fi = (w * st.interest).sum(1, keepdim=True)
            st.interest = torch.cat([torch.gather(st.interest, 1, top), fi], 1)
        st.coords = torch.cat([g(st.coords), fc], 1)
        st.size = torch.cat([torch.gather(st.size, 1, top), fs], 1)
        new_idx = torch.full((B, n), -1, dtype=torch.long, device=x.device)
        new_idx.scatter_(1, top, torch.arange(nk, device=x.device)[None].expand(B, -1))
        st.src = torch.where(live & ~dropped_orig, torch.gather(new_idx, 1, st.src.clamp_min(0)), torch.full_like(st.src, -1))
        return x_new


def kmeans(q, k, n_clusters, iters, seed):
    """Spherical k-means on centred queries and keys together (as src.stage_a.structures.kmeans_ids).
    q, k: (B, H, n, d). Returns centroids (B, H, K, d), normalized centred q and k."""
    B, H, n, d = q.shape
    q, k = q.float(), k.float()
    q, k = F.normalize(q - q.mean(2, keepdim=True), dim=-1), F.normalize(k - k.mean(2, keepdim=True), dim=-1)
    pts = torch.cat([q, k], 2)
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.stack([torch.randperm(2 * n, generator=g)[:n_clusters] for _ in range(B * H)]).view(B, H, n_clusters)
    cent = torch.gather(pts, 2, init.to(pts.device)[..., None].expand(-1, -1, -1, d))
    for _ in range(iters):
        ids = (pts @ cent.transpose(-1, -2)).argmax(-1)
        one = F.one_hot(ids, n_clusters).float()
        s = one.transpose(-1, -2) @ pts
        cent = torch.where(one.sum(2)[..., None] > 0, F.normalize(s, dim=-1), cent)
    return cent, q, k


class Group(Method):
    """clusters: k-means clusters per head; m: key clusters per query; tau: dense fallback threshold (None = never);
    blocks: 0-based blocks that use grouped attention (others dense)."""

    def __init__(self, clusters, m=1, tau=None, blocks=tuple(range(12)), iters=10, seed=0):
        super().__init__()
        self.K, self.m, self.tau, self.blocks, self.iters, self.seed = clusters, m, tau, tuple(blocks), iters, seed
        t = "none" if tau is None else f"{tau:g}"
        self.name = f"group_k{clusters}_m{m}_tau{t}" + ("" if len(self.blocks) == 12 else f"_b{self.blocks[0] + 1}-{self.blocks[-1] + 1}")
        self.dense_heads = []

    def start(self, st):
        super().start(st)
        self.dense_heads = []

    def bias(self, b, q, k, st):
        mk = self.mask(b, q, k, st)
        return None if mk is None else _patch_bias(mk[0], q.shape[2])

    def sort_key(self, b, q, k, st):
        return self._last_key

    def mask(self, b, q, k, st):
        """(allowed (B, H, n, n) over patch tokens, sort key (B, H, n) that makes groups contiguous) or None."""
        if b not in self.blocks:
            return None
        P = N_PREFIX
        B, H, T, d = q.shape
        n = T - P
        cent, qn, kn = kmeans(q[:, :, P:], k[:, :, P:], self.K, self.iters, self.seed)
        self.ops.append((b, "kmeans", H * (self.iters + 1) * 2 * (2 * n) * self.K * d + H * self.iters * 2 * (2 * n) * d))
        kid = (kn @ cent.transpose(-1, -2)).argmax(-1)                                  # (B, H, n)
        qtop = (qn @ cent.transpose(-1, -2)).topk(self.m, dim=-1).indices              # (B, H, n, m)
        qsel = F.one_hot(qtop, self.K).sum(-2).bool()                                   # (B, H, n, K)
        allowed = torch.gather(qsel, 3, kid[:, :, None, :].expand(-1, -1, n, -1))       # (B, H, n, n)
        if self.tau is not None:
            rows = torch.linspace(0, n - 1, min(SAMPLE, n), device=q.device).long()
            with torch.autocast("cuda", enabled=False):
                s = (q[:, :, P:][:, :, rows].float() @ k.float().transpose(-1, -2)) * (d ** -0.5)
                pr = s.softmax(-1)[..., P:]                                             # sampled rows, patch keys
            sev = (pr * ~allowed[:, :, rows]).sum((-1, -2)) / pr.sum((-1, -2)).clamp_min(1e-12)   # (B, H)
            dense = sev > self.tau
            allowed = allowed | dense[..., None, None]
            self.dense_heads.append(dense.float().mean().item())
            self.ops.append((b, "fallback_sample", H * len(rows) * T * d * 4))
        self._last_key = kid
        return allowed, kid


class Oracle(Method):
    """partitions: per image, [block][head] -> memb (N, m) tensor (from the a2g store). Set .current before a batch."""

    def __init__(self, setting):
        super().__init__()
        self.setting, self.name, self.current = setting, f"oracle_{setting}", None

    def bias(self, b, q, k, st):
        return _patch_bias(self.mask(b, q, k, st)[0], q.shape[2])

    def mask(self, b, q, k, st):
        """Partitions are over the original patches; after merging, a token belongs to every cluster any of its
        patches belongs to (pruned patches drop out)."""
        mk = merged_masks([img[b] for img in self.current], st, q)
        self._last_key = mk[1]
        return mk

    def sort_key(self, b, q, k, st):
        return self._last_key


def merged_masks(per_image, st, q):
    """per_image: for each image, a list of partitions (one per head, or one shared) as {"memb": (N0, m)}.
    Returns (allowed (B, H, n, n), sort key (B, H, n)) over the current tokens; a shared partition is used by all heads."""
    B, H, T, _ = q.shape
    n = T - N_PREFIX
    out, keys = [], []
    for i, parts in enumerate(per_image):
        src = st.src[i]
        live = src >= 0
        ms, ks = [], []
        for L in parts:
            memb = L["memb"].long().to(q.device)
            N0 = memb.shape[0]
            M = torch.zeros(N0, int(memb.max()) + 1, device=q.device)
            for s in range(memb.shape[1]):
                ok = memb[:, s] >= 0
                M[ok.nonzero()[:, 0], memb[ok, s]] = 1.0
            if n != N0 or not torch.equal(src, torch.arange(N0, device=q.device)):
                M = torch.zeros(n, M.shape[1], device=q.device).index_add_(0, src[live], M[live]).clamp_max(1.0)
            ms.append((M @ M.T) > 0)
            ks.append(M.argmax(1))
        a, kk = torch.stack(ms), torch.stack(ks)
        if a.shape[0] == 1:
            a, kk = a.expand(H, -1, -1), kk.expand(H, -1)
        out.append(a)
        keys.append(kk)
    return torch.stack(out), torch.stack(keys)


def _key_from_mask(method, b, q, k, st):
    mk = method.mask(b, q, k, st)
    return None if mk is None else mk[1]


class Windows(Method):
    def __init__(self, w, grid):
        super().__init__()
        self.w, self.grid, self.name = w, grid, f"windows_w{w}"
        self.ids = None

    def bias(self, b, q, k, st):
        mk = self.mask(b, q, k, st)
        return None if mk is None else _patch_bias(mk[0], q.shape[2])

    def mask(self, b, q, k, st):
        if b in GLOBAL_BLOCKS:
            return None
        B, H, T, _ = q.shape
        if self.ids is None or self.ids.device != q.device:
            self.ids = window_ids(self.grid, self.w, q.device)
        assert T - N_PREFIX == len(self.ids), "windows need the full token grid"
        allowed = (self.ids[:, None] == self.ids[None, :])[None, None].expand(B, H, -1, -1)
        return allowed, self.ids[None, None].expand(B, H, -1)

    def sort_key(self, b, q, k, st):
        return self.ids[None, None].expand(q.shape[0], 1, -1)


class Combo(Method):
    """Attention masks from `attn` (Group), token reduction from `reduce` (ToMe or Prune)."""

    def __init__(self, attn, reduce):
        super().__init__()
        self.attn, self.reduce = attn, reduce
        self.name = f"{attn.name}+{reduce.name}"
        self.proportional = getattr(reduce, "proportional", False)
        self.restore_at = getattr(reduce, "restore_at", None)
        if hasattr(attn, "block_inputs"):
            self.block_inputs = attn.block_inputs

    def start(self, st):
        self.attn.start(st)
        self.reduce.start(st)
        self.ops = []

    def bias(self, b, q, k, st):
        return self.attn.bias(b, q, k, st)

    def mask(self, b, q, k, st):
        return self.attn.mask(b, q, k, st)

    def sort_key(self, b, q, k, st):
        return self.attn.sort_key(b, q, k, st) if hasattr(self.attn, "sort_key") else None

    def after_attn(self, b, x, k, p, st):
        return self.reduce.after_attn(b, x, k, p, st)

    @property
    def all_ops(self):
        return self.attn.ops + self.reduce.ops


def ops_of(method):
    if method is None:
        return []
    return method.all_ops if hasattr(method, "all_ops") else method.ops


class Graph(Method):
    """Stage A graph teacher (full overlap, summaries visible to siblings), from the trees a2g-calib saved.
    Set .trees (one tree dict list per image) before each batch. Oracle: built from dense attention."""

    def __init__(self, setting, grid):
        super().__init__()
        self.setting, self.grid, self.name = setting, grid, f"graph_{setting}"
        self.ts, self.trees = None, None

    def start(self, st):
        from src.stage_a.structures import TreeStructure
        super().start(st)
        n = st.size.shape[1]
        self.ts = TreeStructure(self.trees, n, self.grid, st.size.device)

    def sort_key(self, b, q, k, st):
        from src.stage_a.model import LEVEL_OF_BLOCK
        M = self.ts.lv[LEVEL_OF_BLOCK[b]][0]
        B, N0, C = M.shape
        n = q.shape[2] - N_PREFIX
        if n == N0 and torch.equal(st.src, torch.arange(N0, device=q.device)[None].expand(B, -1)):
            return M.argmax(-1)[:, None]
        idx = st.src.clamp_min(0)[..., None].expand(-1, -1, C)
        Mt = torch.zeros(B, n, C, device=q.device).scatter_add_(1, idx, M * (st.src >= 0).float()[..., None])
        return Mt.argmax(-1)[:, None]

    def block_inputs(self, b, x, st):
        """Stage A's bias and summaries; after merging, a token's membership is the mean of its patches' memberships
        (so its bias matches an unmerged token's, proportional attention adds its size), and each summary is the
        membership-weighted mean over the original patches (a merged token counts once per patch it holds)."""
        from src.stage_a.model import LEVEL_OF_BLOCK, BlockInputs
        l = LEVEL_OF_BLOCK[b]
        M, Q, Sib, valid = self.ts.lv[l]
        B, N0, C = M.shape
        D = x.shape[-1]
        self.ops.append((b, "summaries", 2 * (M > 0).sum((1, 2)).float().mean().item() * D
                         + 2 * 2 * valid.sum(1).float().mean().item() * D * D))
        n = x.shape[1] - N_PREFIX
        if n == N0 and torch.equal(st.src, torch.arange(N0, device=x.device)[None].expand(B, -1)):
            return self.ts(b, x)
        live = (st.src >= 0).float()[..., None]
        idx = st.src.clamp_min(0)[..., None].expand(-1, -1, C)
        Mt = torch.zeros(B, n, C, device=x.device).scatter_add_(1, idx, M * live) / st.size[..., None]
        Qt = (Mt > 0).float()
        T = n + N_PREFIX
        bias = torch.zeros(B, 1, T, T + C, device=x.device)
        w = Qt @ Mt.transpose(1, 2)
        bias[:, 0, N_PREFIX:, N_PREFIX:T] = torch.where(w > 0, w.clamp_min(1e-30).log(), NEG)
        tot = M.sum(1)
        vis = ((Qt @ Sib.float()) > 0) & (Mt == 0) & valid[:, None, :]
        bias[:, 0, N_PREFIX:, T:] = torch.where(vis, tot.clamp_min(1e-30).log()[:, None, :], NEG)
        bias[:, 0, :N_PREFIX, T:] = NEG
        xs = x[:, N_PREFIX:].float() * st.size[..., None]
        summ = (Mt.transpose(1, 2) @ xs) / tot.clamp_min(1e-12)[..., None]
        coords = (M.transpose(1, 2) @ self.ts.coords) / tot.clamp_min(1e-12)[..., None]
        return BlockInputs(bias=bias, extra=(summ.to(x.dtype), coords))


class BlockOracle(Oracle):
    """Per-block teacher (heads averaged): one partition per block shared by all heads, from the dense attention.
    .current: per image, [block] -> {"memb": (N, m)}. Works after merging like the per-head teacher."""

    def __init__(self, setting):
        super().__init__(setting)
        self.name = f"blockoracle_{setting}"

    def mask(self, b, q, k, st):
        mk = merged_masks([[img[b]] for img in self.current], st, q)
        self._last_key = mk[1]
        return mk


class InterestToMe(ToMe):
    """Interest-driven budget (owner direction, future-work note): tokens with interest >= theta are never merged; in
    every block from `first`, a fraction `rate` of the remaining out-of-focus tokens is merged away (bipartite matching
    among out-of-focus tokens only, at most half of them per block), never below `floor` out-of-focus (context) tokens. Compute therefore follows how much of the image is
    of interest. Token counts differ per image, so it runs on one image at a time (per_image)."""
    per_image = True

    def __init__(self, rate, theta=0.5, first=0, last=99, floor=0, dilate=0, prop=True):
        super().__init__(0, first, last, prop)
        self.rate, self.theta, self.floor, self.dilate = rate, theta, floor, dilate
        self.name = f"itome_l{rate:g}_t{theta:g}" + ("" if first == 0 else f"_from{first + 1}") + \
            ("" if floor == 0 else f"_floor{floor}") + ("" if dilate == 0 else f"_dil{dilate}") + ("" if prop else "_noprop")

    def start(self, st):
        """Safety margin: enlarge the given focus map by `dilate` patches in every direction (max over a square)."""
        super().start(st)
        dilate_interest(st, self.dilate)

    def after_attn(self, b, x, k, p, st):
        if not (self.first <= b <= self.last) or st.interest is None:
            return x
        assert x.shape[0] == 1, "InterestToMe runs one image at a time"
        low = st.interest[0] < self.theta
        la, lb = int(low[::2].sum()), int(low[1::2].sum())
        nl = int(low.sum())
        self.r = min(int(self.rate * nl), la, lb, max(0, nl - self.floor))   # keep at least `floor` context tokens
        if self.r <= 0:
            return x
        self._penalty = torch.where(low, 0.0, 1e4)[None]                 # any merge touching a high token is ranked last
        try:
            return ToMe.after_attn(self, b, x, k, p, st)                 # carries the real interest through the merge
        finally:
            self._penalty = None


class InterestPrune(Prune):
    """Previous-frame-gated pruning baseline (in the style of RS3-Prune and Video Patch Pruning): at block(s) `at`, keep
    every token whose interest is >= theta plus a fraction `keep_low` of the others (those CLS attends to most); the
    rest are fused into one token (EViT) and their outputs carried, or put back before `restore_at`. Per image."""
    per_image = True

    def __init__(self, keep_low, at=(2,), theta=0.01, dilate=0, restore_at=None):
        super().__init__(1.0, at=at, restore_at=restore_at)
        self.keep_low, self.theta, self.dilate = keep_low, theta, dilate
        self.name = (f"iprune_k{keep_low:g}_at{'-'.join(str(a + 1) for a in self.at)}_t{theta:g}"
                     + ("" if dilate == 0 else f"_dil{dilate}") + ("" if restore_at is None else f"_restore{restore_at + 1}"))

    def start(self, st):
        super().start(st)
        dilate_interest(st, self.dilate)

    def _kept(self, n, st):
        if st.interest is None:
            return n
        high = int((st.interest[0] >= self.theta).sum())
        return min(n, high + int(round(self.keep_low * (n - high))))

    def _score(self, p, st):
        return p[:, :, 0, N_PREFIX:].mean(1) + 1e4 * (st.interest >= self.theta).float()   # interest tokens always kept


class GradedToMe(ToMe):
    """Graded interest (owner: a weighted score per region): every token carries its region's weight w in [0, 1].
    Tokens with w >= 1 - eps are never merged; in each block, a share rate * (1 - w) of each weight class is merged,
    only with tokens of the same class (classes never mix, so weights stay exact); the zero-weight class keeps at least
    `floor` tokens. Weights are rounded to `levels` steps to form classes. Per image."""
    per_image = True

    def __init__(self, rate, floor=0, dilate=0, levels=4, first=0, last=99, prop=False):
        super().__init__(0, first, last, prop)
        self.rate, self.floor, self.dilate, self.levels = rate, floor, dilate, levels
        self.name = f"gtome_l{rate:g}_floor{floor}" + ("" if dilate == 0 else f"_dil{dilate}") + ("" if prop else "_noprop")

    def start(self, st):
        super().start(st)
        if st.interest is not None:
            st.interest = torch.round(st.interest * self.levels) / self.levels
            st.interest0 = st.interest
        dilate_interest(st, self.dilate)

    def _select(self, scores, st, r):
        w = st.interest[0]
        wa, wb = w[::2], w[1::2]
        same = (wa[:, None] == wb[None, :]) & (wa[:, None] < 1 - 1e-6)
        sc = scores[0].masked_fill(~same, float("-inf"))
        node_max, node_idx = sc.max(-1)
        picks = []
        for c in torch.unique(wa):
            c = float(c)
            if c >= 1 - 1e-6:
                continue
            members = (wa == c) & torch.isfinite(node_max)
            n_c = int((w == c).sum())
            r_c = int(self.rate * (1 - c) * n_c)
            if c == 0:
                r_c = min(r_c, max(0, n_c - self.floor))
            r_c = min(r_c, int(members.sum()), int((wb == c).sum()))
            if r_c <= 0:
                continue
            cand = torch.where(members, node_max, torch.full_like(node_max, float("-inf")))
            picks.append(torch.topk(cand, r_c).indices)
        na = wa.shape[0]
        if not picks:
            e = torch.empty(1, 0, dtype=torch.long, device=w.device)
            return e, e, torch.arange(na, device=w.device)[None]
        src = torch.cat(picks)
        keep = torch.ones(na, dtype=torch.bool, device=w.device)
        keep[src] = False
        return src[None], node_idx[src][None], keep.nonzero()[:, 0][None]

    def after_attn(self, b, x, k, p, st):
        if not (self.first <= b <= self.last) or st.interest is None:
            return x
        assert x.shape[0] == 1, "GradedToMe runs one image at a time"
        n = st.size.shape[1]
        self.r = n // 2                                                   # upper bound; _select sets the real count
        saved, self._penalty = getattr(self, "_penalty", None), torch.zeros_like(st.interest)
        try:
            return ToMe.after_attn(self, b, x, k, p, st)
        finally:
            self._penalty = saved


class Reuse(Method):
    """Skipping unchanged regions in video (Eventful-style) on top of an interest method (`inner`). Keep one instance per
    video. A patch is skipped in a frame if its interest is below `theta`, its patch embedding moved less than `delta`
    (relative L2) since it was last computed, and it was computed within the last `max_age` frames. A skipped patch is
    not processed: its output is the cached one, and at every block it still serves as a key and value from the cache
    (rows shared by merged patches are deduplicated, with log(count) added to their scores). Per image (batch 1)."""
    per_image = True

    def __init__(self, inner, delta=0.05, max_age=8, theta=0.01):
        super().__init__()
        self.inner, self.delta, self.max_age, self.theta = inner, delta, max_age, theta
        self.name = f"reuse{delta:g}a{max_age}" + ("" if theta == 0.01 else f"t{theta:g}") + f":{inner.name}"
        self.proportional = getattr(inner, "proportional", False)
        self.restore_at = getattr(inner, "restore_at", None)
        self.cache = None                                                # per video
        self.frame = 0

    def start(self, st):
        self.ops = []
        if hasattr(self.inner, "start"):
            self.inner.start(st)

    def prepare(self, x, st):
        P = N_PREFIX
        e = x[0, P:].float()                                             # (N0, D) patch embeddings of this frame
        N0 = e.shape[0]
        c = self.cache
        self.skip = torch.zeros(N0, dtype=torch.bool, device=x.device)
        if c is not None and st.interest is not None and c["emb"].shape == e.shape:
            moved = (e - c["emb"]).norm(dim=-1) / c["emb"].norm(dim=-1).clamp_min(1e-6)
            fresh = (self.frame - c["age"]) < self.max_age
            self.skip = (st.interest[0] < self.theta) & (moved < self.delta) & fresh & c["valid"]
        self.frame += 1
        self.e = e
        if not self.skip.any():
            return x
        keep = (~self.skip).nonzero()[:, 0]
        st.carried[0, self.skip] = c["out"][self.skip].to(st.carried.dtype)
        new_idx = torch.full((N0,), -1, dtype=torch.long, device=x.device)
        new_idx[keep] = torch.arange(len(keep), device=x.device)
        st.src = new_idx[None]
        st.size, st.coords = st.size[:, keep], st.coords[:, keep]
        if st.interest is not None:
            st.interest = st.interest[:, keep]
        return torch.cat([x[:, :P], x[:, P:][:, keep]], 1)

    def observe_kv(self, b, k, v, st):
        self._kv = getattr(self, "_kv", {})
        self._kv[b] = (k[0, :, N_PREFIX:].detach(), v[0, :, N_PREFIX:].detach(), st.src[0].clone())

    def extra_kv(self, b, st):
        c = self.cache
        if c is None or not self.skip.any():
            return None
        gid = c["gid"][b][self.skip]                                    # cached token id per skipped patch
        uniq, inv, cnt = torch.unique(gid, return_inverse=True, return_counts=True)
        first = torch.zeros(len(uniq), dtype=torch.long, device=gid.device).scatter_(0, inv, torch.arange(len(gid), device=gid.device))
        rows = self.skip.nonzero()[:, 0][first]
        ke, ve = c["k"][b][:, rows][None], c["v"][b][:, rows][None]     # (1, H, S, hd)
        self.ops.append((b, "reuse_dedup", 0))
        return ke, ve, cnt.float().log()

    def after_attn(self, b, x, k, p, st):
        return self.inner.after_attn(b, x, k, p, st) if hasattr(self.inner, "after_attn") else x

    def finish(self, x, st):
        """Cache, per original patch, the final (pre-norm) output, the embedding and the per-block K, V of the token that
        held it when it was last computed."""
        P, N0 = N_PREFIX, st.src.shape[1]
        dev = x.device
        if not getattr(self, "_kv", None):
            return
        if self.cache is None or self.cache["emb"].shape != self.e.shape:
            H, hd = self._kv[0][0].shape[0], self._kv[0][0].shape[2]
            self.cache = {"emb": self.e.clone(), "age": torch.zeros(N0, dtype=torch.long, device=dev),
                          "valid": torch.zeros(N0, dtype=torch.bool, device=dev),
                          "out": torch.zeros(N0, x.shape[-1], device=dev),
                          "k": [torch.zeros(H, N0, hd, device=dev) for _ in range(12)],
                          "v": [torch.zeros(H, N0, hd, device=dev) for _ in range(12)],
                          "gid": [torch.zeros(N0, dtype=torch.long, device=dev) for _ in range(12)]}
        c = self.cache
        self._kv = getattr(self, "_kv", {})
        done = ~self.skip                                                # patches computed in this frame
        live = st.src[0] >= 0
        upd = done & live
        c["out"][upd] = x[0, P:][st.src[0][upd]].float()
        c["emb"][done] = self.e[done]
        c["age"][done] = self.frame - 1
        c["valid"][done] = True
        for b, (k, v, src) in self._kv.items():
            ok = done & (src >= 0)
            c["k"][b][:, ok] = k[:, src[ok]].float()
            c["v"][b][:, ok] = v[:, src[ok]].float()
            c["gid"][b][ok] = (self.frame - 1) * 100000 + src[ok]
        self._kv = {}

    @property
    def all_ops(self):
        return self.ops + ops_of(self.inner)


class OneShotToMe(InterestToMe):
    """Speed variant of the focus budget (Q3): merging happens at one block only. At block `at`, out-of-focus tokens
    are merged down to max(floor, keep_low x their count) in a few rounds of bipartite matching among themselves (the
    matching features of merged tokens are their size-weighted mean, so no new keys are needed between rounds); every
    later block runs on the reduced set. Far fewer operations than merging in every block. Per image."""

    def __init__(self, keep_low, at=1, theta=0.01, floor=0, dilate=0, prop=False, max_rounds=8):
        super().__init__(0.5, theta=theta, first=at, last=at, floor=floor, dilate=dilate, prop=prop)
        self.keep_low, self.at, self.max_rounds = keep_low, at, max_rounds
        self.name = f"otome_k{keep_low:g}_at{at + 1}_t{theta:g}_floor{floor}" + ("" if dilate == 0 else f"_dil{dilate}") + \
            ("" if prop else "_noprop")

    def after_attn(self, b, x, k, p, st):
        if b != self.at or st.interest is None:
            return x
        assert x.shape[0] == 1, "OneShotToMe runs one image at a time"
        self._metric = F.normalize(k[:, :, N_PREFIX:].float().mean(1), dim=-1)
        target = max(self.floor, int(self.keep_low * int((st.interest[0] < self.theta).sum())))
        try:
            for _ in range(self.max_rounds):
                low = st.interest[0] < self.theta
                nl = int(low.sum())
                la, lb = int(low[::2].sum()), int(low[1::2].sum())
                self.r = min(nl - target, la, lb)
                if self.r <= 0:
                    break
                self._penalty = torch.where(low, 0.0, 1e4)[None]
                x = ToMe.after_attn(self, b, x, k, p, st)
        finally:
            self._metric, self._penalty = None, None
        return x


class FastInterestToMe(Method):
    """Speed-engineered focus budget (N1). Same treatment as InterestToMe (protected tokens never merged, a share
    `rate` of the out-of-focus tokens merged per block down to `floor`, size-weighted means, carried positions and
    interest), arranged for speed: once per frame the patch tokens are reordered so protected tokens come first, and the
    whole merge schedule is computed from the initial counts (the frame's only host sync); each block then merges inside
    the contiguous out-of-focus tail with sizes known in advance (no per-block syncs, no full-sequence gathers).
    Bipartite sides alternate within the tail, so merges can differ from InterestToMe. Per image (batch 1)."""
    per_image = True
    needs_cls = False    # merging uses keys only; the fast path skips the CLS attention row
    n_high_fixed = None  # N7 CUDA graphs: protected count given by the caller (no host sync inside the graph)

    def __init__(self, rate, theta=0.01, floor=0, dilate=0, first=0, last=99, prop=False, size_class=0):
        super().__init__()
        self.rate, self.theta, self.floor, self.dilate, self.first, self.last = rate, theta, floor, dilate, first, last
        self.proportional, self.size_class = prop, size_class
        self.name = f"ftome_l{rate:g}_t{theta:g}_floor{floor}" + ("" if dilate == 0 else f"_dil{dilate}") + \
            ("" if prop else "_noprop") + ("" if not size_class else f"_cls{size_class}")

    def start(self, st):
        self.ops = []
        dilate_interest(st, self.dilate)
        if self.size_class and st.interest is not None and self.n_high_fixed is None:
            st.interest, _ = promote(st.interest, self.theta, self.size_class, st.grid)
            st.interest0 = st.interest

    def prepare(self, x, st):
        if st.interest is None:
            self.sched = {}
            return x
        P = N_PREFIX
        high = st.interest[0] >= self.theta
        order = torch.argsort((~high).to(torch.int8), stable=True)       # protected first, original order kept
        self.n_high = self.n_high_fixed if self.n_high_fixed is not None else int(high.sum())   # the one host sync
        n_low, self.sched = st.size.shape[1] - self.n_high, {}
        for b in range(self.first, min(self.last, 63) + 1):              # known in advance: low count after each block
            r = min(int(self.rate * n_low), max(0, n_low - self.floor), n_low // 2)
            if r <= 0:
                break
            self.sched[b] = r
            n_low -= r
        inv = torch.empty_like(order)
        inv[order] = torch.arange(len(order), device=x.device)
        st.src = inv[st.src.clamp_min(0)].where(st.src >= 0, st.src)
        st.size, st.coords, st.interest = st.size[:, order], st.coords[:, order], st.interest[:, order]
        return torch.cat([x[:, :P], x[:, P:][:, order]], 1)

    def after_attn(self, b, x, k, p, st):
        r = self.sched.get(b, 0) if hasattr(self, "sched") else 0
        if r <= 0:
            return x
        P, h = N_PREFIX, self.n_high
        D = x.shape[-1]
        n = st.size.shape[1]
        t0 = P + h                                                        # start of the out-of-focus tail
        metric = F.normalize(k[:, :, t0:].float().mean(1), dim=-1)       # (1, nl, hd)
        a, bb = metric[:, ::2], metric[:, 1::2]
        na, nb = a.shape[1], bb.shape[1]
        scores = a @ bb.transpose(-1, -2)
        self.ops.append((b, "tome_match", 2 * na * nb * metric.shape[-1]))
        node_max, node_idx = scores.max(-1)
        edge = torch.sort(node_max, dim=-1, descending=True, stable=True).indices
        src_a, unm_a = edge[:, :r], edge[:, r:]
        dst_b = torch.gather(node_idx, 1, src_a)
        tail = x[:, t0:]
        xa, xb = tail[:, ::2], tail[:, 1::2]
        sz = st.size[:, h:]
        sa, sb = sz[:, ::2], sz[:, 1::2]
        co = st.coords[:, h:]
        ca, cb = co[:, ::2], co[:, 1::2]
        it = st.interest[:, h:]
        ia, ib = it[:, ::2], it[:, 1::2]
        g = lambda t, i: torch.gather(t, 1, i[..., None].expand(-1, -1, t.shape[-1])) if t.dim() == 3 else torch.gather(t, 1, i)
        w = g(sa, src_a)
        sb_new = sb.scatter_add(1, dst_b, w)
        xb_new = ((xb.float() * sb[..., None]).scatter_add(1, dst_b[..., None].expand(-1, -1, D), g(xa, src_a).float() * w[..., None])
                  / sb_new[..., None])
        cb_new = (cb * sb[..., None]).scatter_add(1, dst_b[..., None].expand(-1, -1, 2), g(ca, src_a) * w[..., None]) / sb_new[..., None]
        ib_new = (ib * sb).scatter_add(1, dst_b, g(ia, src_a) * w) / sb_new
        self.ops.append((b, "tome_merge", 4 * r * D))
        x_new = torch.cat([x[:, :t0], g(xa, unm_a), xb_new.to(x.dtype)], 1)
        st.size = torch.cat([st.size[:, :h], g(sa, unm_a), sb_new], 1)
        st.coords = torch.cat([st.coords[:, :h], g(ca, unm_a), cb_new], 1)
        st.interest = torch.cat([st.interest[:, :h], g(ia, unm_a), ib_new], 1)
        nu = na - r                                                       # new index of every old tail token
        new_a = torch.empty(1, na, dtype=torch.long, device=x.device)
        new_a.scatter_(1, unm_a, h + torch.arange(nu, device=x.device)[None])
        new_a.scatter_(1, src_a, h + nu + dst_b)
        new_b = h + nu + torch.arange(nb, device=x.device)[None]
        new_idx = torch.cat([torch.arange(h, device=x.device)[None], torch.empty(1, n - h, dtype=torch.long, device=x.device)], 1)
        new_idx[:, h::2], new_idx[:, h + 1::2] = new_a, new_b
        live = st.src >= 0
        st.src = torch.where(live, torch.gather(new_idx, 1, st.src.clamp_min(0)), st.src)
        return x_new
