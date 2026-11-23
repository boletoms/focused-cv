"""Stage B forward pass: the frozen DINOv3 ViT-S/16 with a variable number of patch tokens (merging, pruning) and
optional attention masks (grouped attention). Same weights and numerics as src/stage_a/model.py: linear layers and MLP
in the autocast dtype (bf16, never fp16), attention scores, softmax and value product in fp32.

State carried through the blocks, per image (patch tokens only; CLS and the 4 registers are never merged or pruned):
- size (B, n): how many original patches each current token stands for;
- coords (B, n, 2): its position for RoPE, the size-weighted mean of its patches' positions;
- src (B, N0): for every original patch, the index of the current token that holds it, or -1 once pruned;
- carried (B, N0, D): a pruned patch's last feature, which becomes its output;
- interest (B, n), optional: a prior weight per token in [0, 1] (owner, future ablation: regions of interest can be any
  part of the image, or all of it). Merging ranks merges by similarity minus the larger interest of the pair and
  carries the size-weighted mean; pruning keeps by CLS attention times interest and carries the weighted mean.
  Uniform interest gives exactly the behaviour without a prior.
At the end every original patch gets an output (unmerge), so the dense VOC probe applies unchanged.

Late restore (Token Cropr, CVPR 2025): a method with `restore_at = b` puts every original patch back before block b:
kept patches take their current token's feature (copies for merged tokens), pruned patches their carried feature;
block b and later then run on the full grid.

A method is an object with optional hooks:
- bias(b, q, k, st) -> additive (B or 1, H or 1, T, T) fp32 bias, after RoPE, before softmax;
- proportional (bool): add log(size) on the key side (ToMe's proportional attention);
- after_attn(b, x, k, p, st) -> x: may merge or prune patch tokens, updating st;
- record(b, kind, value): FLOP bookkeeping (see flops.py).
"""
from dataclasses import dataclass

import torch

from src.stage_a.model import N_PREFIX, Output, Wrapped, _rot, patch_coords, rope

from .fast_attn import attend


@dataclass
class State:
    size: torch.Tensor
    coords: torch.Tensor
    src: torch.Tensor
    carried: torch.Tensor
    pairs: list               # per block: attention pairs computed, per image (mean over heads)
    tokens: list              # per block: (tokens entering attention, tokens entering the MLP)
    tiles: list = None        # per block: query-key pairs in the 64 x 64 tiles a block-sparse kernel must compute
    interest: torch.Tensor | None = None   # (B, n) optional interest prior in [0, 1], carried like coords
    grid: tuple = None                     # patch grid (rows, columns)
    coords0: torch.Tensor | None = None    # original patch positions and interest, for restoring
    interest0: torch.Tensor | None = None
    pixels: torch.Tensor | None = None     # the normalised input frames (R28 surprise on pooled pixels)


TILE = 64


def _up(t):
    return -(-t // TILE) * TILE


def tile_pairs(allowed, key=None):
    """Query-key pairs inside the TILE x TILE tiles that contain any allowed pair, after reordering each head's patch
    tokens by `key` (B, H, n) as the block-sparse kernel does (CLS and registers stay first). allowed: (B, H or 1, T, Tk)
    bool, Tk >= T (extra keys, e.g. summaries, follow). Returns (B,) pairs, mean over heads."""
    B, H, T, Tk = allowed.shape
    if key is not None:
        H = max(H, key.shape[1])
        allowed, key = allowed.expand(B, H, T, Tk), key.expand(B, H, key.shape[-1])
        P, n = N_PREFIX, key.shape[-1]
        perm = torch.sort(key, dim=-1, stable=True).indices
        full = torch.cat([torch.arange(P, device=key.device).expand(B, H, P), perm + P,
                          torch.arange(P + n, Tk, device=key.device).expand(B, H, Tk - P - n)], -1)
        allowed = torch.gather(allowed, 2, full[..., :T, None].expand(-1, -1, -1, Tk))
        allowed = torch.gather(allowed, 3, full[:, :, None, :].expand(-1, -1, T, -1))
    a = torch.nn.functional.pad(allowed, (0, _up(Tk) - Tk, 0, _up(T) - T))
    t = a.view(B, H, _up(T) // TILE, TILE, _up(Tk) // TILE, TILE).any(5).any(3)
    return t.float().sum((-1, -2)).mean(1) * TILE * TILE


class Adapter(torch.nn.Module):
    """Residual low-rank adapter after a block (Q10): x + up(down(norm(x))), zero-initialized so it starts as identity."""

    def __init__(self, width, rank=32):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.down = torch.nn.Linear(width, rank)
        self.up = torch.nn.Linear(rank, width)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(torch.nn.functional.gelu(self.down(self.norm(x)))).to(x.dtype)


class EffModel(Wrapped):
    fast = False         # True: real kernels for timing (fast_attn.py); False: eager fp32 attention (accuracy runs)
    compiled = True      # compile FlexAttention in the fast path
    events = None        # list of (start, end) CUDA events around mask building, in the fast path
    chunk_elems = 2 ** 27  # eager path: attention scores held at once (B x heads x T x Tk), about 0.5 GB in fp32

    def attn_fast(self, b, q, k, v, st, method, scale):
        P = N_PREFIX
        assert not hasattr(method, "block_inputs"), "summary tokens are not in the fast path (graph teacher is an oracle)"
        start = None
        if self.events is not None and method is not None and hasattr(method, "mask"):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        mk = method.mask(b, q, k, st) if method is not None and hasattr(method, "mask") else None
        logsize = st.size.log() if method is not None and getattr(method, "proportional", False) else None
        if method is not None and hasattr(method, "observe_kv"):            # temporal reuse in the fast path
            method.observe_kv(b, k, v, st)
        ekv = method.extra_kv(b, st) if method is not None and hasattr(method, "extra_kv") else None
        if ekv is not None:
            ke, ve, lc = ekv
            T = q.shape[2]
            kk, vv = torch.cat([k, ke.to(k.dtype)], 2), torch.cat([v, ve.to(v.dtype)], 2)
            am = torch.zeros(q.shape[0], 1, 1, kk.shape[2], device=q.device, dtype=q.dtype)
            if logsize is not None:
                am[:, 0, 0, P:T] = logsize.to(q.dtype)
            am[:, 0, 0, T:] = lc.to(q.dtype)
            o = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, attn_mask=am)
        else:
            o = attend(q, k, v, P, mk, logsize, self.compiled, self.events if mk is not None else None, start)
        p = None
        if method is not None and hasattr(method, "after_attn") and getattr(method, "needs_cls", True):   # CLS row (pruning)
            s0 = (q[:, :, :1].float() @ k.float().transpose(-1, -2)) * scale
            if logsize is not None:
                s0[..., P:] += logsize[:, None, None]
            p = s0.softmax(-1)
        T = q.shape[2]                     # FLOPs as in the eager path for unmasked attention; masked: not counted here
        st.pairs.append(torch.full((q.shape[0],), float(T * T) if mk is None else float("nan")))
        st.tiles.append(torch.full((q.shape[0],), float(_up(T) * _up(T)) if mk is None else float("nan")))
        return o, p

    @staticmethod
    def restore(x, st):
        P, D = N_PREFIX, x.shape[-1]
        kept = st.src >= 0
        feat = torch.gather(x[:, P:], 1, st.src.clamp_min(0)[..., None].expand(-1, -1, D))
        patches = torch.where(kept[..., None], feat, st.carried.to(x.dtype))
        B, N0 = st.src.shape
        st.size = torch.ones(B, N0, device=x.device)
        st.coords, st.interest = st.coords0, st.interest0
        st.src = torch.arange(N0, device=x.device)[None].expand(B, -1).clone()
        return torch.cat([x[:, :P], patches], 1)

    def block_b(self, b, layer, x, st: State, method):
        B, T, D = x.shape
        P, H, hd = N_PREFIX, self.heads, self.hd
        att = layer.attention
        y = layer.norm1(x)
        q, k, v = (p(y).view(B, T, H, hd).transpose(1, 2) for p in (att.q_proj, att.k_proj, att.v_proj))
        cos, sin = rope(st.coords, self.inv_freq)                    # (B, n, hd)
        cos, sin = cos[:, None], sin[:, None]
        q = torch.cat([q[:, :, :P], _rot(q[:, :, P:], cos.to(q.dtype), sin.to(q.dtype))], 2)
        k = torch.cat([k[:, :, :P], _rot(k[:, :, P:], cos.to(k.dtype), sin.to(k.dtype))], 2)
        if self.fast:
            o, p = self.attn_fast(b, q, k, v, st, method, att.scaling)
            o = o.transpose(1, 2).reshape(B, T, D)
            x = x + layer.layer_scale1(att.o_proj(o.to(y.dtype)))
            if method is not None and hasattr(method, "after_attn"):
                x = method.after_attn(b, x, k, p, st)
            st.tokens.append((T, x.shape[1]))
            return x + layer.layer_scale2(layer.mlp(layer.norm2(x)))
        if method is not None and hasattr(method, "observe_kv"):            # temporal reuse: remember this frame's K, V
            method.observe_kv(b, k, v, st)
        ekv = method.extra_kv(b, st) if method is not None and hasattr(method, "extra_kv") else None
        inp = method.block_inputs(b, x, st) if method is not None and hasattr(method, "block_inputs") else None
        if inp is not None and inp.extra is not None:          # summary tokens (graph teacher): extra keys and values
            et, ec = inp.extra
            S = et.shape[1]
            ey = layer.norm1(et)
            ek, ev = (p(ey).view(B, S, H, hd).transpose(1, 2) for p in (att.k_proj, att.v_proj))
            c2, s2 = rope(ec, self.inv_freq)
            ek = _rot(ek, c2[:, None].to(ek.dtype), s2[:, None].to(ek.dtype))
            k, v = torch.cat([k, ek], 2), torch.cat([v, ev], 2)
        with torch.autocast("cuda", enabled=False):
            bias = inp.bias.float() if inp is not None else (method.bias(b, q, k, st) if method is not None and hasattr(method, "bias") else None)
            Tk = k.shape[2]
            if bias is not None:
                fin = torch.isfinite(bias).float().sum((-1, -2))                # computed pairs (weighted or not)
                st.pairs.append((fin.mean(-1) if bias.shape[1] > 1 else fin[:, 0]).expand(B).cpu())
                key = method.sort_key(b, q, k, st) if hasattr(method, "sort_key") else None
                st.tiles.append(tile_pairs(torch.isfinite(bias), key).expand(B).cpu())
            else:
                st.pairs.append(torch.full((B,), float(T * T)))
                st.tiles.append(torch.full((B,), float(_up(T) * _up(Tk))))
            lg = None
            if method is not None and getattr(method, "proportional", False):
                lg = torch.zeros(B, 1, 1, Tk, device=x.device)                  # summary keys (if any) get no size term
                lg[:, 0, 0, P:T] = st.size.log()
            if ekv is not None:            # cached keys and values of skipped patches, log(count) for deduplicated rows
                ke, ve, lc = ekv
                kf_extra, vf_extra = ke.to(k.dtype), ve.to(v.dtype)
                k, v = torch.cat([k, kf_extra], 2), torch.cat([v, vf_extra], 2)
                S2 = ke.shape[2]
                lg_new = torch.zeros(B, 1, 1, Tk + S2, device=x.device)
                if lg is not None:
                    lg_new[..., :Tk] = lg
                lg_new[:, 0, 0, Tk:] = lc
                lg, Tk = lg_new, Tk + S2
                st.pairs[-1] = st.pairs[-1] * 0 + float(T * Tk) if bias is None else st.pairs[-1]
                st.tiles[-1] = torch.full((B,), float(_up(T) * _up(Tk))) if bias is None else st.tiles[-1]
            # attention a few heads at a time (memory: B x hc x T x Tk), softmax in place; same result up to fp32 rounding
            hc = H if B * H * T * Tk <= self.chunk_elems else max(1, self.chunk_elems // (B * T * Tk))
            o = torch.empty(B, H, T, hd, device=x.device)
            p_cls = torch.empty(B, H, 1, Tk, device=x.device)
            kf, vf = k.float(), v.float()
            for h0 in range(0, H, hc):
                h1 = min(H, h0 + hc)
                s = (q[:, h0:h1].float() @ kf[:, h0:h1].transpose(-1, -2)) * att.scaling
                if bias is not None:
                    s += bias if bias.shape[1] == 1 else bias[:, h0:h1]
                if lg is not None:
                    s += lg
                if torch.is_grad_enabled():                               # training (Q10 adapters): out of place
                    s = s.softmax(-1)
                else:
                    s -= s.amax(-1, keepdim=True)
                    s.exp_()
                    s /= s.sum(-1, keepdim=True)
                p_cls[:, h0:h1] = s[:, :, :1]
                o[:, h0:h1] = s @ vf[:, h0:h1]
                del s
            del bias
            o = o.transpose(1, 2).reshape(B, T, D)
        x = x + layer.layer_scale1(att.o_proj(o.to(y.dtype)))
        if method is not None and hasattr(method, "after_attn"):           # real tokens only (no summary keys)
            x = method.after_attn(b, x, k[:, :, :T], p_cls[..., :T], st)    # CLS row is all pruning reads
        st.tokens.append((T, x.shape[1]))
        return x + layer.layer_scale2(layer.mlp(layer.norm2(x)))

    def forward(self, pixels, structure=None, dtype=torch.float32, attn_cb=None, return_levels=False, state_out=None,
                interest=None):
        """structure: a Stage B method (or None = dense); interest: optional (B, N0) prior per patch.
        Returns Output with one final token per original token."""
        method = structure
        H, W = pixels.shape[-2:]
        B = pixels.shape[0]
        N0 = (H // 16) * (W // 16)
        dev = pixels.device
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            x = self.bb.embeddings(pixels)
            D = x.shape[-1]
            st = State(size=torch.ones(B, N0, device=dev), coords=patch_coords(H // 16, W // 16, dev)[None].expand(B, -1, -1),
                       src=torch.arange(N0, device=dev)[None].expand(B, -1).clone(),
                       carried=torch.zeros(B, N0, D, device=dev, dtype=x.dtype), pairs=[], tokens=[], tiles=[],
                       interest=None if interest is None else interest.to(dev).float())
            st.coords0, st.interest0, st.grid = st.coords, st.interest, (H // 16, W // 16)
            st.pixels = pixels
            if method is not None and hasattr(method, "start"):
                method.start(st)
            if method is not None and hasattr(method, "prepare"):            # temporal reuse: drop unchanged patches
                x = method.prepare(x, st)
            adapters = getattr(self, "adapters", None)
            for b, layer in enumerate(self.bb.model.layer):
                if method is not None and getattr(method, "restore_at", None) == b:
                    x = self.restore(x, st)
                if method is not None and hasattr(method, "before_block"):    # R1: clustering after a whole block
                    x = method.before_block(b, x, st)
                x = self.block_b(b, layer, x, st, method)
                if adapters is not None and method is not None:          # adapters only when compressing (Q10)
                    x = adapters[b](x)
            if method is not None and hasattr(method, "finish"):
                method.finish(x, st)
            out = self.bb.norm(x)
            if method is not None and hasattr(method, "reconstruct_out"):   # R1 Expedite: weighted reconstruction
                out = method.reconstruct_out(out, st)
            kept = st.src >= 0
            idx = (st.src.clamp_min(0) + N_PREFIX)[..., None].expand(-1, -1, D)
            patches = torch.where(kept[..., None], torch.gather(out, 1, idx), self.bb.norm(st.carried).to(out.dtype))
            final = torch.cat([out[:, :N_PREFIX], patches], 1)
        if state_out is not None:
            state_out.append(st)
        return Output(final=final.float())


def load_adapters(model, path):
    """Attach Q10 adapters saved by src.stage_b.adapt (used only when a method compresses)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    width = model.bb.config.hidden_size
    model.adapters = torch.nn.ModuleList([Adapter(width, ck["rank"]) for _ in model.bb.model.layer])
    model.adapters.load_state_dict(ck["state"])
    model.adapters.to(next(model.parameters()).device).eval()
    return model
