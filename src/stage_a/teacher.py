"""The teacher's cluster tree (brief section 5): top-down spectral splits of each level's dense attention, with
shared boundary tokens, nested across levels.

Conventions
- A[l] is (N, N) patch-to-patch attention for level l (0-based: 0 = level 1), averaged over the level's blocks and
  heads, rows not renormalized. Rows are queries.
- A token's "share" to a set S within cluster P is sum_{j in S} A[t, j] / sum_{j in P} A[t, j].
- Each token carries, per level, its memberships (at most 2) and one core cluster. The core is carried through the
  splits: when the core cluster splits, the core moves to the half that holds the larger share (or the token's
  only half). Core sizes are checked at every accepted split, so final cores keep at least s_min tokens.
- Final weights at each level: the token's attention shares to its clusters at that level, normalized to sum to 1.
- Graph teacher (owner decision 2026-11-17): every level is split from the whole image using its own attention,
  with up to max_memb memberships per token; clusters of adjacent levels are linked by membership overlap, so a
  cluster can have several parents and a parent shares children with other parents. links[c, p] is the share of
  child c's membership weight that lies in parent p; c links to p if that share is at least LINK (or is c's largest).
"""
from dataclasses import dataclass, field

import torch

MAX_MEMB, HALO, LINK = 2, 0.25, 0.1


@dataclass(frozen=True)
class TreeConfig:
    eps: float = 0.05
    beta: float = 0.3      # > 1 gives the strict tree (no boundary tokens)
    s_min: int = 32
    nested: bool = True
    max_memb: int = MAX_MEMB
    graph: bool = False    # levels built independently, then linked by overlap (implies not nested)

    @property
    def name(self):
        b = "inf" if self.beta > 1 else f"{self.beta:g}"
        n = f"eps{self.eps:g}_beta{b}_s{self.s_min}"
        if self.graph:
            return n + f"_graph_m{self.max_memb}"
        return n + ("" if self.nested else "_nonnested") + ("" if self.max_memb == MAX_MEMB else f"_m{self.max_memb}")

    @property
    def independent(self):
        return self.graph or not self.nested


@dataclass
class Level:
    clusters: list            # list of LongTensor member indices (cpu)
    parent: list              # parent cluster index one level up (-1 at level 4 or non-nested)
    memb: torch.Tensor        # (N, max_memb) cluster ids, -1 = none
    weight: torch.Tensor      # (N, max_memb) float32, rows sum to 1
    core: torch.Tensor        # (N,) cluster id
    crossing: list = field(default_factory=list)  # crossing share of every accepted cut
    links: torch.Tensor | None = None  # graph teacher: (C, P) overlap shares to the clusters one level up

    def to_dict(self):
        d = {"memb": self.memb.short(), "weight": self.weight.half(), "core": self.core.short(),
             "parent": torch.tensor(self.parent, dtype=torch.short), "crossing": torch.tensor(self.crossing)}
        if self.links is not None:
            d["links"] = self.links.half()
        return d


def fiedler(W: torch.Tensor) -> torch.Tensor:
    """Second-smallest eigenvector of the normalized Laplacian of W, sign fixed for determinism."""
    d = W.sum(1).clamp_min(1e-12).rsqrt()
    L = torch.eye(len(W), device=W.device, dtype=W.dtype) - d[:, None] * W * d[None, :]
    try:
        v = torch.linalg.eigh(L)[1][:, 1]
    except torch.linalg.LinAlgError:                      # rare non-convergence in fp32: retry once in fp64
        v = torch.linalg.eigh(L.double())[1][:, 1].to(W.dtype)
    i = v.abs().argmax()
    return v * torch.sign(v[i])


def best_cut(A_P, limited, core_in, beta, s_min):
    """Evaluate every cut of P sorted by Fiedler value; return the one with the lowest crossing share.

    A_P: (n, n) attention restricted to P (rows queries). limited: (n,) tokens already at the membership limit.
    core_in: (n,) tokens whose core is P. Returns None if no candidate, else a dict of member masks and stats.
    """
    n = len(A_P)
    ks = torch.arange(s_min, n - s_min + 1, device=A_P.device)
    if len(ks) == 0:
        return None
    order = torch.argsort(fiedler((A_P + A_P.T) / 2), stable=True)
    Ap = A_P[order][:, order]
    lim, cin = limited[order], core_in[order]
    r = Ap.sum(1).clamp_min(1e-12)
    cum = torch.cumsum(Ap, 1)
    shX = cum[:, ks - 1].T / r                              # (K, n) share to the first k positions
    shY = 1 - shX
    pos = torch.arange(n, device=Ap.device)
    sideX = pos[None, :] < ks[:, None]
    sideX = torch.where(lim[None, :], shX >= shY, sideX)    # limited tokens descend to the half with the larger share
    nX, nY = sideX.sum(1), (~sideX).sum(1)

    # Boundary tokens: share to the other half >= beta, below the membership limit. Every shared token sits in both
    # halves, so the halo limit is: shared <= 25% of each enlarged half. Highest shares first; the limit only gets
    # harder as tokens are added, so the selection is the longest prefix that satisfies it.
    other = torch.where(sideX, shY, shX)
    cand = ~lim[None] & (other >= beta)
    o = torch.argsort(torch.where(cand, other, torch.full_like(other, -1.0)), dim=1, descending=True, stable=True)
    c_s = torch.gather(cand, 1, o)
    x_s = torch.gather(sideX, 1, o)
    n_sh = torch.cumsum(c_s.long(), 1)
    toY, toX = torch.cumsum((c_s & x_s).long(), 1), torch.cumsum((c_s & ~x_s).long(), 1)
    ok = c_s & (n_sh <= HALO * torch.minimum(nX[:, None] + toX, nY[:, None] + toY))
    m = torch.cumprod(ok.long(), 1).sum(1)
    rank = torch.argsort(o, dim=1, stable=True)
    join = cand & (rank < m[:, None])
    joinY, joinX = join & sideX, join & ~sideX
    Xo, Yo = (sideX & ~joinY).float(), (~sideX & ~joinX).float()
    cross = ((Xo @ Ap) * Yo).sum(1) + ((Yo @ Ap) * Xo).sum(1)
    cross = cross / Ap.sum().clamp_min(1e-12)
    coreX = cin[None] & ((sideX & ~joinY) | (sideX & joinY & (shX >= shY)) | (~sideX & joinX & (shX > shY)))
    coreY = cin[None] & ~coreX
    valid = (coreX.sum(1) >= s_min) & (coreY.sum(1) >= s_min)   # a cut must leave both cores at s_min or more
    if not bool(valid.any()):
        return None
    cross = torch.where(valid, cross, torch.full_like(cross, float("inf")))
    k = int(torch.argmin(cross))
    inv = torch.empty_like(order)
    inv[order] = pos
    pick = lambda m: m[k][inv]
    return {"cross": float(cross[k]), "X": pick(sideX | joinX), "Y": pick(~sideX | joinY),
            "shared": pick(joinX | joinY), "coreX": pick(coreX), "coreY": pick(coreY),
            "ncoreX": int(coreX[k].sum()), "ncoreY": int(coreY[k].sum())}


def build_level(A, start, counts, core, cfg: TreeConfig, cache=None):
    """Split each starting cluster recursively. start: list of (member LongTensor, parent id).
    counts: (N,) memberships per token at this level so far (from the copied start clusters). core: (N,) index into start.
    Returns clusters, parents, core (N,) final cluster ids, crossing shares."""
    dev = A.device
    work = [(m.to(dev), p, i) for i, (m, p) in enumerate(start)]
    core = core.to(dev).clone()
    counts = counts.to(dev).clone()
    done, parents, crossing = [], [], []
    next_id = len(start)  # working ids for new halves; remapped at the end
    while work:
        P, par, cid = work.pop()
        key = None
        res = None
        if len(P) >= 2 * cfg.s_min:
            limited, core_in = counts[P] >= cfg.max_memb, core[P] == cid
            if cache is not None:
                key = (P.cpu().numpy().tobytes(), limited.cpu().numpy().tobytes(), core_in.cpu().numpy().tobytes())
                res = cache.get(key)
            if res is None:
                res = best_cut(A[P][:, P], limited, core_in, cfg.beta, cfg.s_min)
                if cache is not None:
                    cache[key] = res if res is not None else False
            res = res or None
        if res is not None and res["cross"] <= cfg.eps and min(res["ncoreX"], res["ncoreY"]) >= cfg.s_min:
            x_id, y_id = next_id, next_id + 1
            next_id += 2
            core[P[res["coreX"]]] = x_id
            core[P[res["coreY"]]] = y_id
            counts[P[res["shared"]]] += 1
            crossing.append(res["cross"])
            work.append((P[res["Y"]], par, y_id))
            work.append((P[res["X"]], par, x_id))
        else:
            done.append((P, par, cid))
    remap = {cid: i for i, (_, _, cid) in enumerate(done)}
    core = torch.tensor([remap[int(c)] for c in core.tolist()], device=dev)
    return [P.cpu() for P, _, _ in done], [p for _, p, _ in done], core.cpu(), crossing


def finalize(A, clusters, parents, core, crossing, max_memb=MAX_MEMB) -> Level:
    N = len(A)
    memb = torch.full((N, max_memb), -1, dtype=torch.long)
    weight = torch.zeros(N, max_memb)
    fill = torch.zeros(N, dtype=torch.long)
    for c, P in enumerate(clusters):
        for t in P.tolist():
            memb[t, fill[t]] = c
            fill[t] += 1
    assert int(fill.max()) <= max_memb and int(fill.min()) >= 1
    # attention share of token t to each of its clusters
    M = torch.zeros(N, len(clusters), device=A.device)
    for c, P in enumerate(clusters):
        M[P.to(A.device), c] = 1.0
    toc = (A @ M).cpu()                                   # (N, C) attention from t to cluster c's members
    for s in range(max_memb):
        ok = memb[:, s] >= 0
        weight[ok, s] = toc[ok.nonzero()[:, 0], memb[ok, s]]
    weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-12)
    single = memb[:, 1] < 0
    weight[single, 0] = 1.0
    return Level(clusters, parents, memb, weight, core, crossing)


def build_tree(A_levels, cfg: TreeConfig, caches=None) -> list[Level]:
    """A_levels: list of 4 (N, N) tensors (level 1..4). Returns 4 Levels (index 0 = level 1)."""
    N = len(A_levels[0])
    out = [None] * 4
    for l in (3, 2, 1, 0):
        A = A_levels[l]
        if l == 3 or cfg.independent:
            start = [(torch.arange(N), -1)]
            counts, core = torch.ones(N, dtype=torch.long), torch.zeros(N, dtype=torch.long)
        else:
            up = out[l + 1]
            start = [(P, c) for c, P in enumerate(up.clusters)]
            counts = (up.memb >= 0).sum(1)
            core = up.core.clone()
        cache = None if caches is None else caches.setdefault((l, cfg.beta, cfg.s_min, cfg.independent, cfg.max_memb), {})
        clusters, parents, core_f, crossing = build_level(A, start, counts, core, cfg, cache)
        out[l] = finalize(A, clusters, parents, core_f, crossing, cfg.max_memb)
    if cfg.graph:
        for l in range(3):
            out[l].links = overlap_links(out[l], out[l + 1])
            out[l].parent = out[l].links.argmax(1).tolist()   # largest parent, for display only
    return out


def build_single(A, cfg: TreeConfig, cache=None) -> Level:
    """One level split from the whole image (the per-head, per-block teacher uses one per block and head)."""
    N = len(A)
    start = [(torch.arange(N), -1)]
    clusters, parents, core_f, crossing = build_level(A, start, torch.ones(N, dtype=torch.long),
                                                      torch.zeros(N, dtype=torch.long), cfg, cache)
    return finalize(A, clusters, parents, core_f, crossing, cfg.max_memb)


def membership_matrix(L: Level) -> torch.Tensor:
    """(N, C) membership weights."""
    M = torch.zeros(L.memb.shape[0], len(L.clusters))
    for s in range(L.memb.shape[1]):
        ok = L.memb[:, s] >= 0
        M[ok.nonzero()[:, 0], L.memb[ok, s]] = L.weight[ok, s]
    return M


def overlap_links(lo: Level, up: Level) -> torch.Tensor:
    """(C, P): share of each child's membership weight that lies in each parent (weighted by the parent
    membership); rows sum to 1."""
    Ml, Mu = membership_matrix(lo), membership_matrix(up)
    W = Ml.T @ Mu
    return W / W.sum(1, keepdim=True).clamp_min(1e-12)


def linked(links: torch.Tensor) -> torch.Tensor:
    """Boolean (C, P) parent links: share >= LINK, and always the largest parent."""
    return (links >= LINK) | (links == links.max(1, keepdim=True).values)


def tree_stats(levels: list[Level]) -> dict:
    s = {}
    for l, L in enumerate(levels, 1):
        sizes = torch.bincount(L.core, minlength=len(L.clusters))
        s[f"l{l}_clusters"] = len(L.clusters)
        s[f"l{l}_core_min"] = int(sizes.min())
        s[f"l{l}_core_median"] = float(sizes.float().median())
        s[f"l{l}_core_max"] = int(sizes.max())
        s[f"l{l}_boundary_share"] = float((L.memb[:, 1] >= 0).float().mean())
        s[f"l{l}_memberships_per_token"] = float((L.memb >= 0).sum(1).float().mean())
        if L.links is not None:
            s[f"l{l}_parents_per_child"] = float(linked(L.links.float()).sum(1).float().mean())
        s[f"l{l}_crossing_mean"] = float(torch.tensor(L.crossing).mean()) if L.crossing else 0.0
    return s
