"""Fast attention for wall-clock timing (B2). Same maths as the eager fp32 path in model.py, with real kernels:
- no mask: scaled_dot_product_attention (flash or memory-efficient kernel);
- proportional attention only (ToMe): scaled_dot_product_attention with a log(size) key bias;
- grouped masks (Group, Oracle, Windows): FlexAttention with a block mask. Patch tokens are reordered per head so each
  group is contiguous (the method's sort key), the block mask skips empty blocks, and outputs are put back in order.
Attention runs in the dtype of q (bf16 in timing runs; never fp16). Mask building (from `start`, recorded by the caller before the method builds its mask, to the block mask) is
timed separately via `events`."""
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_flex = {"fn": None}


def flex(compiled):
    if not compiled:
        return flex_attention
    if _flex["fn"] is None:
        _flex["fn"] = torch.compile(flex_attention, dynamic=False)
    return _flex["fn"]


def attend(q, k, v, P, mask=None, logsize=None, compiled=True, events=None, start=None):
    """q, k, v: (B, H, T, hd) after RoPE. mask: (allowed (B, H, n, n), key (B, H, n)) or None. logsize: (B, n) or None.
    Returns (B, H, T, hd)."""
    B, H, T, hd = q.shape
    if mask is None:
        am = None
        if logsize is not None:
            am = torch.zeros(B, 1, 1, T, device=q.device, dtype=q.dtype)
            am[:, 0, 0, P:] = logsize.to(q.dtype)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=am)
    allowed, key = mask
    n = T - P
    perm = torch.sort(key, dim=-1, stable=True).indices                          # (B, H, n)
    gi = perm[..., None].expand(-1, -1, -1, hd)
    qp, kp, vp = (torch.cat([t[:, :, :P], torch.gather(t[:, :, P:], 2, gi)], 2) for t in (q, k, v))
    al = torch.gather(allowed, 2, perm[..., None].expand(-1, -1, -1, n))
    al = torch.gather(al, 3, perm[:, :, None, :].expand(-1, -1, n, -1)).contiguous()

    def mask_mod(b, h, qi, ki):
        return (qi < P) | (ki < P) | al[b, h, (qi - P).clamp(min=0), (ki - P).clamp(min=0)]

    bm = create_block_mask(mask_mod, B, H, T, T, device=q.device)
    score_mod = None
    if logsize is not None:
        ls = torch.zeros(B, H, T, device=q.device, dtype=torch.float32)
        ls[:, :, P:] = torch.gather(logsize[:, None].expand(-1, H, -1).float(), 2, perm)

        def score_mod(score, b, h, qi, ki):
            return score + ls[b, h, ki]
    if events is not None:
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        events.append((start, e1))
    op = flex(compiled)(qp, kp, vp, score_mod=score_mod, block_mask=bm)
    out = torch.empty_like(op)
    out[:, :, :P] = op[:, :, :P]
    out[:, :, P:] = op[:, :, P:].scatter(2, gi, op[:, :, P:])
    return out
