"""Attention structures for the reference implementation (brief sections 6 and 7.1): biases, summaries, costs and
severed attention. Every structure returns, per block, a BlockInputs for the wrapper and a BlockStructure for the
cost counter. CLS and registers always attend to everything; patch queries see them unless an ablation says not.
"""
import math

import torch

from .cost import N_PREFIX, BlockStructure
from .model import LEVEL_OF_BLOCK, BlockInputs, patch_coords
from .teacher import linked

NEG = float("-inf")
GLOBAL_BLOCKS = (1, 3, 7, 11)  # last block of each level (ViTDet-style windows)


# ---------------------------------------------------------------- teacher trees

def level_tensors(levels_batch, l, n_patch, device, key_only=False, nested=True):
    """Membership matrices for level l over a batch of trees. Returns M (B,N,C) weights, Q (B,N,C) query mask,
    Sib (B,C,C) sibling mask, valid (B,C)."""
    B = len(levels_batch)
    C = max(len(t[l]["parent"]) for t in levels_batch)
    M = torch.zeros(B, n_patch, C, device=device)
    core = torch.zeros(B, n_patch, C, device=device)
    valid = torch.zeros(B, C, dtype=torch.bool, device=device)
    Sib = torch.zeros(B, C, C, dtype=torch.bool, device=device)
    rows = torch.arange(n_patch, device=device)
    for b, tree in enumerate(levels_batch):
        L = tree[l]
        memb, w = L["memb"].to(device).long(), L["weight"].to(device).float()
        for s in range(memb.shape[1]):
            ok = memb[:, s] >= 0
            M[b, rows[ok], memb[ok, s]] = w[ok, s]
        core[b, rows, L["core"].to(device).long()] = 1.0
        nc = len(L["parent"])
        valid[b, :nc] = True
        par = L["parent"].to(device).long()
        if "links" in L and l < 3:                       # graph teacher: siblings share at least one parent
            lk = linked(L["links"].float()).to(device).float()
            Sib[b, :nc, :nc] = (lk @ lk.T) > 0
        elif nested and l < 3:
            Sib[b, :nc, :nc] = par[:, None] == par[None, :]
        else:
            Sib[b, :nc, :nc] = True
        Sib[b, :nc, :nc] &= ~torch.eye(nc, dtype=torch.bool, device=device)
    Q = core if key_only else (M > 0).float()
    return M, Q, Sib, valid


class TreeStructure:
    """Teacher-tree variants. variant: full | key_only | strict | nonnested (strict and nonnested trees are built
    differently; at use they behave like full). Ablations: summaries=False, globals_visible=False."""

    def __init__(self, trees, n_patch, grid, device, key_only=False, nested=True, summaries=True, globals_visible=True):
        self.n, self.device = n_patch, device
        self.key_only, self.summaries, self.globals_visible = key_only, summaries, globals_visible
        self.coords = patch_coords(grid, grid, device)
        self.lv = [level_tensors(trees, l, n_patch, device, key_only, nested) for l in range(4)]
        self._bias = [None] * 4

    def allowed(self, l):
        M, Q, _, _ = self.lv[l]
        return (Q @ M.transpose(1, 2)) > 0                # (B, N, N) patch query -> patch key

    def visible_summaries(self, l):
        M, Q, Sib, valid = self.lv[l]
        return ((Q @ Sib.float()) > 0) & (M == 0) & valid[:, None, :]   # (B, N, C)

    def bias(self, l):
        if self._bias[l] is None:
            M, Q, Sib, valid = self.lv[l]
            B, N, C = M.shape
            T = N + N_PREFIX
            S = C if self.summaries else 0
            bias = torch.zeros(B, 1, T, T + S, device=self.device)
            w = Q @ M.transpose(1, 2)
            bias[:, 0, N_PREFIX:, N_PREFIX:T] = torch.where(w > 0, w.clamp_min(1e-30).log(), NEG)
            if not self.globals_visible:
                bias[:, 0, N_PREFIX:, :N_PREFIX] = NEG
            if S:
                tot = M.sum(1)                                     # (B, C) total membership weight
                vis = self.visible_summaries(l)
                bias[:, 0, N_PREFIX:, T:] = torch.where(vis, tot.clamp_min(1e-30).log()[:, None, :], NEG)
                bias[:, 0, :N_PREFIX, T:] = NEG
            self._bias[l] = bias
        return self._bias[l]

    def summaries_at(self, l, x):
        """Membership-weighted mean of each cluster's tokens in x (B, T, D), at the weighted centroid."""
        M = self.lv[l][0]
        tot = M.sum(1).clamp_min(1e-12)                            # (B, C)
        s = (M.transpose(1, 2) @ x[:, N_PREFIX:].float()) / tot[..., None]
        c = (M.transpose(1, 2) @ self.coords) / tot[..., None]
        return s.to(x.dtype), c

    def __call__(self, b, x):
        l = LEVEL_OF_BLOCK[b]
        extra = self.summaries_at(l, x) if self.summaries else None
        return BlockInputs(bias=self.bias(l), extra=extra)

    def block_structures(self, i):
        """Per-block cost structures for image i of the batch."""
        out = []
        for l in range(4):
            M, Q, Sib, valid = self.lv[l]
            nc = int(valid[i].sum())
            qn = (Q[i, :, :nc] > 0).sum(0).tolist()
            kn = (M[i, :, :nc] > 0).sum(0).tolist()
            sp = int(self.visible_summaries(l)[i].sum()) if self.summaries else 0
            bs = BlockStructure(groups=list(zip(qn, kn)), summary_pairs=sp, n_summaries=nc if self.summaries else 0,
                                patch_sees_globals=self.globals_visible)
            out.append(bs)
        return [out[LEVEL_OF_BLOCK[b]] for b in range(12)]


class HeadTeacherStructure:
    """Per-head, per-block teacher (owner decision 2026-11-17): in every block and head, the teacher's clusters
    from that head's own dense attention; a query attends to the keys it shares a cluster with (full overlap).
    No summaries, like k-means. levels[b][h] is a Level for block b, head h (one image)."""

    def __init__(self, levels, n_patch, device):
        self.n, self.device, self.levels = n_patch, device, levels

    @staticmethod
    def member_mask(L, n_patch, device):
        m = torch.zeros(n_patch, len(L.clusters), device=device)
        for s in range(L.memb.shape[1]):
            ok = L.memb[:, s] >= 0
            m[ok.nonzero()[:, 0].to(device), L.memb[ok, s].to(device)] = 1.0
        return m

    def allowed_block(self, b):
        """(H, N, N) bool."""
        out = []
        for L in self.levels[b]:
            m = self.member_mask(L, self.n, self.device)
            out.append((m @ m.T) > 0)
        return torch.stack(out)

    def block_structures(self, i=0):
        out = []
        for hs in self.levels:
            groups = [(len(P), len(P)) for L in hs for P in L.clusters]
            out.append(BlockStructure(groups=groups, heads=len(hs)))
        return out


# ---------------------------------------------------------------- fixed and per-layer groupings

def _group_bias(groups, T, device):
    """groups: (H, N) group ids per head (or (1, N)). Returns (1, H, T, T) bias: 0 within group, -inf across."""
    same = groups[:, :, None] == groups[:, None, :]
    bias = torch.zeros(1, groups.shape[0], T, T, device=device)
    bias[0, :, N_PREFIX:, N_PREFIX:] = torch.where(same, 0.0, NEG)
    return bias


def window_ids(grid, w, device):
    ys, xs = torch.meshgrid(torch.arange(grid, device=device), torch.arange(grid, device=device), indexing="ij")
    return ((ys // w) * (grid // w) + xs // w).flatten()


class WindowStructure:
    def __init__(self, grid, w, device):
        assert grid % w == 0
        self.grid, self.w = grid, w
        self.ids = window_ids(grid, w, device)[None]
        self.b = _group_bias(self.ids, grid * grid + N_PREFIX, device)

    def __call__(self, b, x):
        return None if b in GLOBAL_BLOCKS else BlockInputs(bias=self.b)

    def allowed_block(self, b):
        return None if b in GLOBAL_BLOCKS else (self.ids[:, :, None] == self.ids[:, None, :])

    def block_structures(self, i=0):
        n_win = (self.grid // self.w) ** 2
        return [None if b in GLOBAL_BLOCKS else BlockStructure(groups=[(self.w ** 2, self.w ** 2)] * n_win) for b in range(12)]


class RandomStructure:
    """One fixed random permutation per head, split into near-equal groups; same for all images and blocks."""

    def __init__(self, n_patch, n_groups, heads, seed, device):
        g = torch.Generator().manual_seed(seed)
        ids = torch.empty(heads, n_patch, dtype=torch.long)
        for h in range(heads):
            perm = torch.randperm(n_patch, generator=g)
            ids[h, perm] = torch.arange(n_patch) * n_groups // n_patch
        self.ids, self.heads, self.seed, self.n_groups = ids.to(device), heads, seed, n_groups
        self.b = _group_bias(self.ids, n_patch + N_PREFIX, device)
        self.groups = [(int(s), int(s)) for h in range(heads) for s in torch.bincount(ids[h], minlength=n_groups)]

    def __call__(self, b, x):
        return BlockInputs(bias=self.b)

    def allowed_block(self, b):
        return self.ids[:, :, None] == self.ids[:, None, :]          # (H, N, N)

    def block_structures(self, i=0):
        return [BlockStructure(groups=self.groups, heads=self.heads)] * 12


def kmeans_ids(q, k, n_clusters, iters, seed):
    """q, k: (B, H, N, d) patch queries and keys. Seeded spherical k-means on queries and keys together, after
    centering queries and keys separately per image and head (uncentered, queries and keys fall into separate
    clusters and most queries get no keys). Returns query ids and key ids, each (B, H, N)."""
    B, H, N, d = q.shape
    q, k = q.float(), k.float()
    q, k = q - q.mean(2, keepdim=True), k - k.mean(2, keepdim=True)
    pts = torch.nn.functional.normalize(torch.cat([q, k], 2), dim=-1)              # (B, H, 2N, d)
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.stack([torch.randperm(2 * N, generator=g)[:n_clusters] for _ in range(B * H)]).view(B, H, n_clusters)
    cent = torch.gather(pts, 2, init.to(pts.device)[..., None].expand(-1, -1, -1, d))
    for _ in range(iters):
        ids = (pts @ cent.transpose(-1, -2)).argmax(-1)                             # (B, H, 2N)
        one = torch.nn.functional.one_hot(ids, n_clusters).float()
        s = one.transpose(-1, -2) @ pts
        cnt = one.sum(2)[..., None]
        cent = torch.where(cnt > 0, torch.nn.functional.normalize(s, dim=-1), cent)
    ids = (pts @ cent.transpose(-1, -2)).argmax(-1)
    return ids[..., :N], ids[..., N:]


class KMeansStructure:
    """SVG2-style: in each block and head, k-means on that block's patch queries and keys; a query attends to the
    keys in its own cluster. Records per-image cost structures as it runs."""

    def __init__(self, n_clusters, seed, iters=10):
        self.k, self.seed, self.iters = n_clusters, seed, iters
        self.records = {}

    def mask_from(self, q, k):
        qi, ki = kmeans_ids(q[:, :, N_PREFIX:], k[:, :, N_PREFIX:], self.k, self.iters, self.seed)
        return qi, ki, qi[..., :, None] == ki[..., None, :]

    def _fn(self, b):
        def f(q, k):
            qi, ki, allowed = self.mask_from(q, k)
            B, H, N = qi.shape
            T = N + N_PREFIX
            bias = torch.zeros(B, H, T, k.shape[2], device=q.device)
            bias[:, :, N_PREFIX:, N_PREFIX:T] = torch.where(allowed, 0.0, NEG)
            oq = torch.nn.functional.one_hot(qi, self.k).sum(2)   # (B, H, k) queries per cluster
            ok = torch.nn.functional.one_hot(ki, self.k).sum(2)
            self.records[b] = (oq.cpu(), ok.cpu())
            return bias
        return f

    def __call__(self, b, x):
        return BlockInputs(bias_fn=self._fn(b))

    def block_structures(self, i):
        out = []
        for b in range(12):
            oq, ok = self.records[b]
            H = oq.shape[1]
            groups = [(int(a), int(c)) for h in range(H) for a, c in zip(oq[i, h].tolist(), ok[i, h].tolist())]
            out.append(BlockStructure(groups=groups, heads=H))
        return out


def severed(p_patch, allowed):
    """p_patch: (B, H, N, N) dense patch-to-patch attention; allowed broadcastable bool. Returns (B,) severed share."""
    tot = p_patch.sum((1, 2, 3)).clamp_min(1e-12)
    return (p_patch * (~allowed).float()).sum((1, 2, 3)) / tot
