"""Whole-model FLOP count per image (one multiply-add = 2), from what a forward pass actually did.

Per block: q, k, v, o projections on the tokens entering attention; attention 4 * width per computed query-key pair
(QK^T and AV), pairs averaged over heads; the MLP on the tokens left after merging or pruning; plus the method's own
work as recorded in `ops` (matching, clustering, the fallback sample, fusing). Patch embedding is counted once.
Norms, softmax, RoPE and residual adds are left out for every method alike."""
from src.stage_a.cost import MLP_RATIO, WIDTH

PATCH = 16


def image_flops(st, ops, i, n_patch0, width=WIDTH):
    """st: the model's State for a batch; ops: the method's (block, kind, flops) list; i: image index in the batch."""
    proj = mlp = att = att_tile = 0.0
    tiles = st.tiles if st.tiles else [None] * len(st.pairs)
    for (t_att, t_mlp), pairs, tl in zip(st.tokens, st.pairs, tiles):
        proj += 2 * 4 * t_att * width * width
        mlp += 2 * 2 * t_mlp * width * MLP_RATIO * width
        att += 4 * width * float(pairs[i])
        att_tile += 4 * width * float(tl[i]) if tl is not None else 0.0
    embed = 2 * n_patch0 * 3 * PATCH * PATCH * width
    extra = float(sum(f for _, _, f in ops))
    total = proj + mlp + att + embed + extra
    out = {"flops": total, "flops_attention": att, "flops_mlp": mlp, "flops_proj": proj, "flops_overhead": extra}
    if st.tiles:
        out["flops_tile"] = proj + mlp + att_tile + embed + extra   # attention counted by the 64 x 64 tiles touched
    return out


def dense_flops_tile(n_patch0, n_prefix=5, width=WIDTH, tile=64):
    t = n_patch0 + n_prefix
    tt = -(-t // tile) * tile
    return dense_flops(n_patch0, n_prefix, width) - 12 * 4 * width * (t * t - tt * tt)



def dense_flops(n_patch0, n_prefix=5, width=WIDTH, blocks=12):
    t = n_patch0 + n_prefix
    return blocks * (2 * 4 * t * width * width + 2 * 2 * t * width * MLP_RATIO * width + 4 * width * t * t) + 2 * n_patch0 * 3 * PATCH * PATCH * width
