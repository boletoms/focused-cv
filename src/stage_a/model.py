"""DINOv3 ViT-S/16 with an exposed attention forward (brief sections 3 and 7.1).

Reuses the checkpoint's modules unchanged (frozen, eval mode). Each block is re-implemented around them so that
queries, keys and values are visible, an additive bias can be added to the scores, and extra key/value tokens
(summaries) can be appended. Attention scores, softmax and the value product are always fp32; with
`dtype=torch.bfloat16` only the linear layers and MLP run in bf16 (never fp16).
"""
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from transformers import AutoModel

REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
N_PREFIX = 5  # CLS + 4 registers, in that order, before the patch tokens
LEVEL_BLOCKS = ((0, 1), (2, 3), (4, 5, 6, 7), (8, 9, 10, 11))  # level 1..4 = blocks 1-2, 3-4, 5-8, 9-12
LEVEL_OF_BLOCK = tuple(l for l, bs in enumerate(LEVEL_BLOCKS) for _ in bs)


def load_backbone(device="cuda", repo=REPO) -> nn.Module:
    m = AutoModel.from_pretrained(repo, attn_implementation="eager").eval().to(device)
    return m.requires_grad_(False)


def patch_coords(h: int, w: int, device=None) -> torch.Tensor:
    """(h*w, 2) patch centres (y, x) in [-1, 1], row-major; same as the checkpoint (core1 backbone/rope.py)."""
    ys = torch.arange(0.5, h, device=device) / h
    xs = torch.arange(0.5, w, device=device) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return 2.0 * torch.stack([yy, xx], -1).flatten(0, 1) - 1.0


def rope(coords: torch.Tensor, inv_freq: torch.Tensor):
    """cos, sin (..., N, head_dim) for real-valued coordinates; the checkpoint's formula."""
    a = 2 * math.pi * coords[..., :, None].float() * inv_freq[None, :]
    a = a.flatten(-2).tile(2)
    return a.cos(), a.sin()


def _rot(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


@dataclass
class BlockInputs:
    """What a structure supplies for one block. `bias`: additive, broadcastable to (B, heads, T, T + S).
    `extra`: (tokens (B, S, D) in the residual stream, coords (B, S, 2)) appended as keys/values only."""
    bias: torch.Tensor | None = None
    extra: tuple[torch.Tensor, torch.Tensor] | None = None
    bias_fn: Callable | None = None  # (q, k) after rotary -> bias; for structures that depend on the block's own q, k


@dataclass
class Output:
    final: torch.Tensor                      # (B, T, D) after the final norm
    levels: list = field(default_factory=list)  # residual stream after each level's last block (no norm)


class Wrapped(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.bb = backbone
        cfg = backbone.config
        self.heads, self.hd = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        self.register_buffer("inv_freq", 1 / cfg.rope_theta ** torch.arange(0, 1, 4 / self.hd), persistent=False)

    def block(self, layer, x, cos, sin, inp: BlockInputs | None, attn_cb=None, b=0):
        B, T, D = x.shape
        att = layer.attention
        y = layer.norm1(x)
        q, k, v = (p(y).view(B, T, self.heads, self.hd).transpose(1, 2) for p in (att.q_proj, att.k_proj, att.v_proj))
        P = N_PREFIX
        q = torch.cat([q[:, :, :P], _rot(q[:, :, P:], cos.to(q.dtype), sin.to(q.dtype))], 2)
        k = torch.cat([k[:, :, :P], _rot(k[:, :, P:], cos.to(k.dtype), sin.to(k.dtype))], 2)
        if inp is not None and inp.extra is not None:
            et, ec = inp.extra
            S = et.shape[1]
            ey = layer.norm1(et)
            ek, ev = (p(ey).view(B, S, self.heads, self.hd).transpose(1, 2) for p in (att.k_proj, att.v_proj))
            c2, s2 = rope(ec, self.inv_freq)
            ek = _rot(ek, c2[:, None].to(ek.dtype), s2[:, None].to(ek.dtype))
            k, v = torch.cat([k, ek], 2), torch.cat([v, ev], 2)
        with torch.autocast("cuda", enabled=False):
            s = (q.float() @ k.float().transpose(-1, -2)) * att.scaling
            if inp is not None and inp.bias is not None:
                s = s + inp.bias.float()
            if inp is not None and inp.bias_fn is not None:
                s = s + inp.bias_fn(q, k).float()
            p = s.softmax(-1)
            if attn_cb is not None:
                attn_cb(b, p, q, k)
            o = (p @ v.float()).transpose(1, 2).reshape(B, T, D)
        x = x + layer.layer_scale1(att.o_proj(o.to(y.dtype)))
        return x + layer.layer_scale2(layer.mlp(layer.norm2(x)))

    def forward(self, pixels: torch.Tensor, structure: Callable[[int, torch.Tensor], BlockInputs | None] | None = None,
                dtype=torch.float32, attn_cb=None, return_levels=False) -> Output:
        """structure(block_index, block_input) -> BlockInputs or None (dense). attn_cb(block_index, probs (B,h,T,T+S), q, k)."""
        H, W = pixels.shape[-2:]
        coords = patch_coords(H // 16, W // 16, pixels.device)
        cos, sin = rope(coords, self.inv_freq)
        levels = []
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            x = self.bb.embeddings(pixels)
            for b, layer in enumerate(self.bb.model.layer):
                x = self.block(layer, x, cos, sin, structure(b, x) if structure else None, attn_cb, b)
                if return_levels and b in (1, 3, 7, 11):
                    levels.append(x.float())
            out = self.bb.norm(x)
        return Output(final=out.float(), levels=levels)
