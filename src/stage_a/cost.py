"""Cost counter (brief section 6), in floating-point operations; one multiply-add = 2.

Attention per block: what the packed implementation computes. For each cluster or group, its queries times its keys
(shared tokens counted in every cluster they join), plus the context pass: every patch query against the
N_PREFIX global tokens, the visible summary pairs, and the global queries against every token. QK^T and AV each
cost 2 * width per pair, hence 4 * width per pair. Dense = (patches + globals)^2 pairs.
Unchanged parts per block: q, k, v, o projections and the MLP; summaries add their k and v projections.
"""
from dataclasses import dataclass

WIDTH, MLP_RATIO, N_PREFIX, N_BLOCKS = 384, 4, 5, 12


@dataclass
class BlockStructure:
    groups: list[tuple[int, int]]  # (queries, keys) per cluster or group, patch tokens only
    summary_pairs: int = 0         # patch query -> summary key pairs that are visible
    n_summaries: int = 0           # summary tokens appended in this block
    patch_sees_globals: bool = True
    heads: int = 1                 # groups listed for this many heads; patch pairs are averaged over them


def attention_pairs(n_patch: int, bs: BlockStructure | None) -> int:
    if bs is None:
        return (n_patch + N_PREFIX) ** 2
    patch = sum(q * k for q, k in bs.groups) / bs.heads
    return patch + n_patch * N_PREFIX * bs.patch_sees_globals + bs.summary_pairs + N_PREFIX * (n_patch + N_PREFIX)


def attention_cost(n_patch: int, bs: BlockStructure | None, width: int = WIDTH) -> int:
    return 4 * width * attention_pairs(n_patch, bs)


def other_cost(n_patch: int, bs: BlockStructure | None, width: int = WIDTH) -> int:
    t = n_patch + N_PREFIX
    s = bs.n_summaries if bs is not None else 0
    return 2 * (4 * t * width * width + 2 * s * width * width + 2 * t * width * MLP_RATIO * width)


def model_cost(n_patch: int, blocks: list[BlockStructure | None], width: int = WIDTH) -> dict:
    """blocks: one entry per block (None = dense). Returns absolute costs and ratios to dense."""
    assert len(blocks) == N_BLOCKS
    att = sum(attention_cost(n_patch, b, width) for b in blocks)
    tot = att + sum(other_cost(n_patch, b, width) for b in blocks)
    d_att = N_BLOCKS * attention_cost(n_patch, None, width)
    d_tot = d_att + N_BLOCKS * other_cost(n_patch, None, width)
    return {"attention": att, "total": tot, "attention_ratio": att / d_att, "total_ratio": tot / d_tot}
