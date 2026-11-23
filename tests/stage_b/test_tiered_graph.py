"""R40 plan section K1 (`src/stage_b/tiered_graph.py`): on padded buffers with garbage past the real count, block merges
give the current code's assignment of original patches to tokens, and its features, sizes, positions and interest
(ALGM's two layers and our budget, the fallback's plain ALGM, no-merge and floor cases); the eager padded forward of the
redesign and of its fallback matches the current forward on DAVIS frames, with the same tiers and counted FLOPs. CPU."""
import math

import pytest
import torch

from src.stage_a.model import N_PREFIX, load_backbone
from src.stage_b import tiered_graph as tg
from src.stage_b.baselines import ALGM, ALGMTiered
from src.stage_b.model import EffModel, State

P = N_PREFIX


def _state(n, N0, gen, interest, sizes):
    src = torch.cat([torch.arange(n), torch.randint(0, n, (N0 - n,), generator=gen)])[torch.randperm(N0, generator=gen)]
    st = State(size=sizes[None].clone(), coords=torch.rand(1, n, 2, generator=gen) * 2 - 1, src=src[None].clone(),
               carried=None, pairs=[], tokens=[], tiles=[], interest=interest[None].clone())
    return st


def _case(b, n, plain=False, threshold=0.6, floor=8, rate=0.7, grid=(8, 10), seed=0, pad=21, high_frac=0.15, mid_frac=0.25):
    """Current code against the padded merges at block b on random tokens; returns nothing, asserts equality."""
    gen = torch.Generator().manual_seed(seed)
    D, H, hd = 48, 4, 12
    pad = 0 if b == 0 else pad                                     # block 0 always runs on the full grid
    N0 = grid[0] * grid[1]
    u = torch.rand(n, generator=gen)
    interest = torch.where(u < high_frac, 1.0, torch.where(u < high_frac + mid_frac, 0.5, 0.0))
    sizes = torch.ones(n) if b == 0 else torch.randint(1, 5, (n,), generator=gen).float()
    base = torch.randn(1, P + n, D, generator=gen)
    if b == 0:                                                     # similar 2 x 2 windows, so layer 1 merges some
        f = torch.randn(grid[0] // 2, grid[1] // 2, D, generator=gen).repeat_interleave(2, 0).repeat_interleave(2, 1)
        base[0, P:] = f.reshape(-1, D)[:n] + 0.6 * torch.randn(n, D, generator=gen)
    k = torch.randn(1, H, P + n, hd, generator=gen)
    st = _state(n, N0, gen, interest, sizes)
    if b == 0:
        st.src = torch.arange(N0)[None]
    st.grid = grid
    m = ALGM(threshold) if plain else ALGMTiered(threshold, rate, floor)
    m.start(st)
    x_pad = torch.cat([base, torch.randn(1, pad, D, generator=gen) * 50], 1)
    k_pad = torch.cat([k, torch.randn(1, H, pad, hd, generator=gen) * 50], 2)
    c = n + pad
    ext = lambda t, v: torch.cat([t, torch.full((pad,) + t.shape[1:], v, dtype=t.dtype)])
    size_p, coords_p = ext(sizes, 3.0), ext(st.coords[0], 0.7)
    interest_p, src_p = ext(interest, 0.0), st.src[0].clone()
    spec = tg.Spec(plain, threshold, (1, 5), 2, getattr(m, "high_value", math.inf), 0.5, 0.01, rate, floor)
    st_ref = st
    x_ref = m.after_attn(b, base.clone(), k.clone(), None, st_ref)
    xo, so, co, io, sro, no, counts = tg.merges(b, x_pad, k_pad, size_p, coords_p, interest_p, src_p,
                                                torch.tensor(n), spec, grid)
    n_ref = x_ref.shape[1] - P
    assert int(no) == n_ref, (b, int(no), n_ref)
    assert xo.shape[1] == P + c
    assert torch.equal(sro, st_ref.src[0]), "assignment of original patches"
    assert torch.allclose(xo[0, :P + n_ref], x_ref[0], atol=1e-5)
    assert torch.allclose(so[:n_ref], st_ref.size[0], atol=1e-5)
    assert torch.allclose(co[:n_ref], st_ref.coords[0], atol=1e-5)
    assert torch.allclose(io[:n_ref], st_ref.interest[0], atol=1e-6)
    return counts, n_ref


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_redesign_merges_layer1(seed):
    counts, n_out = _case(0, 80, seed=seed)
    assert counts[1] == 1 and counts[2] > 0, "this case should exercise both ALGM's windows and the budget"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_redesign_merges_layer5(seed):
    counts, _ = _case(4, 61, seed=seed)
    assert counts[1] == 1


def test_redesign_merges_budget_only_and_floor():
    counts, _ = _case(2, 70, seed=3)
    assert counts[2] > 0
    counts, n_out = _case(2, 70, seed=3, floor=1000)                    # floor reached: no budget merge
    assert counts[2] == 0 and n_out == 70


def test_redesign_no_algm_merge():
    counts, _ = _case(0, 80, threshold=1.01, seed=4)
    assert counts[1] == 0
    counts, _ = _case(4, 61, seed=4, high_frac=1.0, mid_frac=0.0)       # every pair touches the high tier
    assert counts[1] == 0 and counts[2] == 0


def test_high_tier_under_algm_variant():
    _case(4, 61, seed=5, high_frac=0.0, mid_frac=0.5)


@pytest.mark.parametrize("b,n", [(0, 80), (4, 61), (2, 61)])
def test_plain_algm_fallback(b, n):
    _case(b, n, plain=True)


def test_algm_threshold_rounds_as_current_path():
    """R-38: on bf16 best scores the padded threshold equals the current code's `mean() + std() / layer` (mean and std
    rounded to bf16 each), with garbage past the real rows; the form built in fp32 and rounded once (before the fix)
    differs on some draws, so the test separates the two."""
    gen = torch.Generator().manual_seed(0)
    same = old_same = draws = 0
    for i in range(400):
        na, pad, layer = int(torch.randint(40, 1500, (1,), generator=gen)), int(torch.randint(0, 300, (1,), generator=gen)), 5
        lo = 0.3 + 0.6 * torch.rand(1, generator=gen).item()
        real = (lo + (0.995 - lo) * torch.rand(na, generator=gen) ** 0.5).to(torch.bfloat16)
        node_max = torch.cat([real, (torch.randn(pad, generator=gen) * 3).to(torch.bfloat16)])
        va = torch.arange(na + pad) < na
        ref = real.mean() + real.std() / layer
        thr = tg._algm_thr(node_max, va, layer)
        assert thr.dtype == torch.bfloat16
        mean = node_max.masked_fill(~va, 0.0).float().sum() / va.sum()
        dev = (node_max.float() - mean).to(torch.bfloat16).float()   # bf16 minus a 0-dim fp32 tensor stays bf16
        old = (mean + (dev.masked_fill(~va, 0.0) ** 2).sum().div(va.sum() - 1).sqrt() / layer).to(torch.bfloat16)
        same += bool(thr == ref)
        old_same += bool(old == ref)
        draws += 1
    assert same >= 0.995 * draws, (same, draws)
    assert old_same <= draws - 20, (old_same, draws)
    fp = torch.rand(300, generator=gen) * 0.5 + 0.4                     # fp32 (no autocast): plain mean + std / layer
    assert torch.isclose(tg._algm_thr(fp, torch.ones(300, dtype=torch.bool), 5), fp.mean() + fp.std() / 5, rtol=0, atol=1e-6)


def _davis_or_skip():
    from src.stage_b.p2_timing import ROOT
    if not ROOT.exists():
        pytest.skip(f"{ROOT} not available")


def test_padded_forward_matches_current_on_davis():
    """Two copies of the redesign with their own expectation states, frame by frame over a DAVIS sequence: the current
    forward (`EffModel` with the wrapper) against the padded eager runner (step 32): tiers, outputs, token counts and
    counted FLOPs; then the fallback on the last frame."""
    _davis_or_skip()
    from src.stage_b.expect import ExpectMap, ExpectState, config_from_name, objects_from_predictions
    from src.stage_b.flops import image_flops
    from src.stage_b.methods import ops_of
    from src.stage_b.p2_davis import shift_by_velocity
    from src.stage_b.p2_timing import HW
    from src.stage_b.timing import from_name
    from src.stage_b.vos.r40_compare import S_FINAL
    from src.stage_b.vos.r40_timing import cov, sequences
    model = EffModel(load_backbone("cpu"))
    blocks = len(model.bb.model.layer)
    seq = sequences("cpu", 6, 1)[0]
    inner, rest = S_FINAL.split("~")
    fb_name, exp_name = rest.split("@")
    cfg = config_from_name(exp_name)

    def pair():
        st = ExpectState()
        return (st, ExpectMap(from_name(inner, HW[0] // 16, blocks), st, config_from_name(exp_name)),
                ExpectMap(from_name(fb_name, HW[0] // 16, blocks), st, config_from_name(exp_name), passive=True))
    (sa, ma, fa), (sb, mb, fb) = pair(), pair()
    oa, ob, seen = {}, {}, []
    runner = tg.TieredRunner(model, HW, fast=False, dtype=torch.float32, graphs=False)
    for t, (x, ids, prev1, prev2, _) in enumerate(seq):
        imap = cov((prev1 > 0) | (shift_by_velocity(prev1, prev2) > 0))
        sa.objects, _ = objects_from_predictions(prev1, prev2, ids, oa, cfg.motion)
        sb.objects, _ = objects_from_predictions(prev1, prev2, ids, ob, cfg.motion)
        use_fb = t == len(seq) - 1
        A, B = (fa, fb) if use_fb else (ma, mb)
        sts = []
        ref = model(x, structure=A, dtype=torch.float32, interest=imap, state_out=sts).final
        out, st = runner.run(x, B, imap)
        assert sa.last == sb.last, (t, sa.last, sb.last)
        assert [tuple(map(int, a)) for a in sts[0].tokens] == [tuple(map(int, a)) for a in st.tokens], t
        assert torch.equal(sts[0].src, st.src), t
        assert (out - ref).abs().max() < 1e-4, (t, float((out - ref).abs().max()))
        N0 = imap.shape[1]
        fa_, fb_ = image_flops(sts[0], ops_of(A), 0, N0)["flops"], image_flops(st, ops_of(B), 0, N0)["flops"]
        assert abs(fa_ - fb_) <= 1e-9 * fa_, (t, fa_, fb_)
        assert all(pc >= t_ for (pc, _), (t_, _) in zip(st.padded, st.tokens))
        seen.append((use_fb, sa.last.get("high", 0) + sa.last.get("mid", 0), int(st.tokens[-1][1])))
    assert any(not f and hm > 0 for f, hm, _ in seen), seen          # the expectation added tiers on some frame
    assert all(n < 1400 for _, _, n in seen), seen                       # every frame merged
