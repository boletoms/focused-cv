"""R40 expectation focus map (`src/stage_b/expect.py`): block matching and camera shift recover known displacements; the mismatch is
zero on unchanged content and one on new border content; the wrapper only promotes (high 1.0, middle tier 0.5), never
demotes, sends target-related surprises to high interest, applies the global rule, and a passive wrapper only observes;
run-time object states detect a collapse and keep the last tracked centroid; names round-trip. CPU."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.stage_a.model import N_PREFIX
from src.stage_b import expect as ex


def _feat(gh, gw, D=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(gh, gw, D, generator=g), dim=-1)


def _moved(f, dy, dx, fill_seed=9):
    """Content at p moves to p + d; uncovered positions get new random content."""
    gh, gw, D = f.shape
    out = _feat(gh, gw, D, fill_seed)
    m = torch.ones(gh, gw, dtype=torch.bool)
    src = ex.shift(m, dy, dx)
    out[src] = f[ex.shift(m, -dy, -dx)]
    return out


def test_block_match_and_camera():
    gh, gw = 10, 12
    prev = _feat(gh, gw)
    now = _moved(prev, 2, -3)
    S = now.reshape(-1, now.shape[-1]) @ prev.reshape(-1, prev.shape[-1]).T
    M = torch.zeros(gh, gw, dtype=torch.bool)
    M[3:6, 5:8] = True
    assert ex.block_match(S, M, gh, gw, 4) == (2, -3)
    assert ex.camera(S, gh, gw, 4) == (2, -3)
    e, border = ex.mismatch(S, gh, gw, (2, -3), 0)
    inside = ~border.reshape(gh, gw)
    assert float(e.reshape(gh, gw)[inside].max()) < 1e-5
    assert bool(border.reshape(gh, gw)[:2].all()) and float(e.reshape(gh, gw)[:2].min()) == 1.0


class _Inner:
    srp_value, theta, needs_cls, proportional, name = 0.5, 0.01, False, False, "inner"

    def __init__(self):
        self.ops = []

    def start(self, st):
        self.ops = []

    def after_attn(self, b, x, k, p, st):
        return x


def _st(f, high):
    gh, gw, D = f.shape
    N = gh * gw
    st = SimpleNamespace(grid=(gh, gw), src=torch.arange(N)[None], interest=high.flatten().float()[None] * 0.3,
                         pixels=None)
    x = torch.cat([torch.zeros(1, N_PREFIX, D), f.reshape(1, N, D)], 1)
    return st, x


def test_wrapper_promotes_only_and_routes_surprise():
    gh, gw = 12, 16
    cfg = ex.ExpectConfig(feat="x0", motion=False, reloc=True, radius=0, score="e", thr=0.5, ring=True, search=False)
    state = ex.ExpectState()
    w = ex.ExpectMap(_Inner(), state, cfg)
    f0 = _feat(gh, gw)
    M = torch.zeros(gh, gw, dtype=torch.bool)
    M[4:6, 4:6] = True
    obj = {"id": 1, "M": M, "T": M, "core": M.clone(), "speed": 0.0, "lost": False, "n": 0, "c": (4.5, 4.5),
           "v": (0.0, 0.0)}
    state.objects = [obj]
    st, x = _st(f0, ex.grow(M, 1))
    w.start(st)
    w.after_attn(0, x, None, None, st)                               # first frame: observes only
    assert state.prev is not None and float(st.interest.max()) == 1.0
    f1 = _moved(f0, 0, 4)                                            # the object (and everything) moves 4 patches right
    f1[11, 15] = _feat(1, 1, f0.shape[-1], 5)[0, 0]                  # an unexpected change outside the object's ring
    high = ex.grow(M, 1)
    st, x = _st(f1, high)
    w.start(st)
    before = st.interest.clone()
    w.after_attn(0, x, None, None, st)
    it = st.interest[0].reshape(gh, gw)
    assert bool((it[high] == 1.0).all())                             # never demoted
    assert bool((it[ex.grow(ex.shift(M, 0, 4), 1)] == 1.0).all())    # relocalised object: high interest
    assert float(it[11, 15]) == 0.5                                  # surprise far from the object: baseline's rules
    assert bool((st.interest >= before).all())


def test_global_rule_and_passive():
    gh, gw = 8, 8
    cfg = ex.ExpectConfig(feat="x0", motion=False, reloc=False, score="e", thr=0.5, ring=False, search=False)
    state = ex.ExpectState()
    w = ex.ExpectMap(_Inner(), state, cfg)
    passive = ex.ExpectMap(_Inner(), state, cfg, passive=True)
    state.objects = []
    f0 = _feat(gh, gw, seed=1)
    st, x = _st(f0, torch.zeros(gh, gw, dtype=torch.bool))
    passive.start(st)
    passive.after_attn(0, x, None, None, st)
    assert state.prev is not None and st.interest is None             # observes; the baseline gets no map (audit)
    f1 = _feat(gh, gw, seed=2)                                       # everything changed
    st, x = _st(f1, torch.zeros(gh, gw, dtype=torch.bool))
    w.start(st)
    w.after_attn(0, x, None, None, st)
    assert state.last["global"] and bool((st.interest == 0.5).all())


def test_objects_from_predictions():
    h, w = 64, 80
    idx2 = torch.zeros(h, w, dtype=torch.long)
    idx1 = torch.zeros(h, w, dtype=torch.long)
    idx2[10:30, 10:30] = 1
    idx1[10:30, 26:46] = 1                                            # moved 16 px (one patch) right
    ost = {}
    objs, all_lost = ex.objects_from_predictions(idx1, idx2, [1], ost, motion=True)
    o = objs[0]
    assert not o["lost"] and not all_lost and abs(o["v"][1] - 1.0) < 1e-6 and abs(o["v"][0]) < 1e-6
    assert bool(o["core"].sum() > o["M"].sum())
    gone = torch.zeros(h, w, dtype=torch.long)
    objs, all_lost = ex.objects_from_predictions(gone, idx1, [1], ost, motion=True)
    assert objs[0]["lost"] and all_lost and objs[0]["n"] == 1 and objs[0]["c"] == o["c"]
    assert bool((objs[0]["Mlast"] == o["M"]).all())


def test_search_region_follows_last_mask():
    gh, gw = 16, 20
    cfg = ex.ExpectConfig(feat="x0", motion=False, reloc=False, score="e", thr=9.0, ring=False, search=True)
    state = ex.ExpectState()
    w = ex.ExpectMap(_Inner(), state, cfg)
    Ml = torch.zeros(gh, gw, dtype=torch.bool)
    Ml[2:8, 2:8] = True                                              # a large object, lost one frame, moving right
    lost = {"id": 1, "M": torch.zeros(gh, gw, dtype=torch.bool), "T": torch.zeros(gh, gw, dtype=torch.bool),
            "core": torch.zeros(gh, gw, dtype=torch.bool), "speed": 0.0, "lost": True, "n": 1, "c": (4.5, 4.5),
            "v": (0.0, 2.0), "Mlast": Ml}
    state.objects = [lost]
    f0 = _feat(gh, gw, seed=3)
    st, x = _st(f0, torch.zeros(gh, gw, dtype=torch.bool))
    w.start(st); w.after_attn(0, x, None, None, st)
    st, x = _st(f0, torch.zeros(gh, gw, dtype=torch.bool))
    w.start(st); w.after_attn(0, x, None, None, st)
    it = st.interest[0].reshape(gh, gw)
    region = ex.grow(ex.shift(Ml, 0, 2), 3)
    assert bool((it[region] == 0.5).all()) and bool((it[~region] == 0.0).all())


def test_config_name_round_trip():
    for n in ("exp-x0-mot-rl-r1-z2.5-ring-srch-pt0.6", "exp-pix-mot-r0-e0.45-ring-srch", "exp-emb-r1-e0.3"):
        assert ex.config_from_name(n).name == n


def test_algm_tiered_merges_middle_with_low_never_high():
    from types import SimpleNamespace as NS
    from src.stage_b.baselines import ALGMTiered
    gh, gw, D = 4, 4, 8
    m = ALGMTiered(0.5, 0.7, floor=0)
    m.ops = []
    x = torch.zeros(1, N_PREFIX + gh * gw, D)
    x[0, N_PREFIX:] = 1.0                                            # identical patches: every free window merges
    interest = torch.zeros(1, gh * gw)
    interest[0, 0] = 1.0                                             # high: window (0, 1, 4, 5) must not merge
    interest[0, 2] = 0.5                                             # middle tier in window (2, 3, 6, 7) with low patches
    st = NS(grid=(gh, gw), size=torch.ones(1, gh * gw), coords=torch.zeros(1, gh * gw, 2), interest=interest,
            src=torch.arange(gh * gw)[None], tokens=[], pairs=[])
    assign = m._merge_nonhigh(0, x, st, interest[0] >= 0.75)[1]
    assert assign is not None
    assert len(set(assign[[0, 1, 4, 5]].tolist())) == 4                 # the high patch's window stays unmerged
    assert len(set(assign[[2, 3, 6, 7]].tolist())) == 1                 # middle and low merge by ALGM's rule


def test_tiered_names_parse():
    from src.stage_b.timing import from_name
    a = from_name("titome_t0.98_l0.7_floor64_dil1_cls64", 30)
    b = from_name("titome_t0.98_l0.7_floor64_dil1_cls64_ha1", 30)
    assert a.name == "titome_t0.98_l0.7_floor64_dil1_cls64" and a.high_value == 0.75
    assert b.name == "titome_t0.98_l0.7_floor64_dil1_cls64_ha1" and b.high_value == float("inf")


def test_tome_tiered_never_merges_high_and_matches_protected_tome():
    from types import SimpleNamespace as NS
    from src.stage_b.baselines import ToMeProtect, ToMeTiered
    from src.stage_b.timing import from_name
    assert from_name("ttome_r4_l0.7_floor0_dil1_cls64", 4).name == "ttome_r4_l0.7_floor0_dil1_cls64"
    n, D = 16, 8
    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, 2, N_PREFIX + n, D, generator=g)
    x = torch.randn(1, N_PREFIX + n, D, generator=g)
    interest = torch.zeros(1, n)
    interest[0, :4] = 1.0
    mk = lambda: NS(grid=(4, 4), size=torch.ones(1, n), coords=torch.zeros(1, n, 2), interest=interest.clone(),
                    src=torch.arange(n)[None], tokens=[], pairs=[])
    t = ToMeTiered(4, 0.7)
    t.ops = []
    st1 = mk()
    _, a1 = t._tome_nonhigh(0, x, k, st1, interest[0] >= 0.75)
    p = ToMeProtect(4)
    p.ops = []
    st2 = mk()
    p.after_attn(0, x, k, None, st2)
    assert len(set(a1[:4].tolist())) == 4 and all((a1 == a1[i]).sum() == 1 for i in range(4))   # high never merged
    assert torch.equal(st1.src, st2.src)                                                         # = protected ToMe


def test_floor_mismatch_reduces_to_cosine_and_tolerates():
    from src.stage_b.surprise import MEAN, STD, pixel_features, unit
    gh, gw = 6, 8
    g = torch.Generator().manual_seed(4)
    img = torch.rand(3, 16 * gh, 16 * gw, generator=g)
    img2 = torch.rand(3, 16 * gh, 16 * gw, generator=g)
    norm = lambda im: (im - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
    a, b = norm(img), norm(img2)
    fa, fb = unit(pixel_features(a)).reshape(gh * gw, -1), unit(pixel_features(b)).reshape(gh * gw, -1)
    e0, border0 = ex.mismatch(fb @ fa.T, gh, gw, (0, 1), 0)
    e1, border1, _ = ex.floor_mismatch(ex.pooled_pixels(b), ex.pooled_pixels(a), gh, gw, (0, 1), 0.0, 0)
    assert torch.equal(border0, border1) and float((e0 - e1).abs().max()) < 1e-2   # sigma 0, tol 0: 1 − cosine (up to the eps)
    flat = torch.full((3, 16 * gh, 16 * gw), 0.5)
    n1, n2 = flat + 0.04 * torch.randn(flat.shape, generator=g), flat + 0.04 * torch.randn(flat.shape, generator=g)
    ef0 = ex.floor_mismatch(ex.pooled_pixels(norm(n2)), ex.pooled_pixels(norm(n1)), gh, gw, (0, 0), 0.0, 0)[0]
    ef = ex.floor_mismatch(ex.pooled_pixels(norm(n2)), ex.pooled_pixels(norm(n1)), gh, gw, (0, 0), 0.1, 0)[0]
    assert float(ef0.mean()) > 0.5 and float(ef.max()) < 0.05                       # flat + noise: fires only unfloored
    moved = torch.roll(img, 4, dims=2)                                               # content moved 4 pixels right
    et0 = ex.floor_mismatch(ex.pooled_pixels(norm(moved)), ex.pooled_pixels(norm(img)), gh, gw, (0, 0), 0.0, 0)[0]
    et1 = ex.floor_mismatch(ex.pooled_pixels(norm(moved)), ex.pooled_pixels(norm(img)), gh, gw, (0, 0), 0.0, 1)[0]
    inner = torch.zeros(gh, gw, dtype=torch.bool); inner[:, 1:] = True
    assert float(et0[inner.flatten()].mean()) > 0.3 and float(et1[inner.flatten()].max()) < 0.05   # (frame means differ slightly)
    for n in ("exp-pix-mot-rl-r0-e0.24-ring-srch-f0.05-o1", "exp-pix-mot-rl-r0-e0.3-ring-srch-f0.02"):
        assert ex.config_from_name(n).name == n


def _tiered_run(name, n_side=40, D=16, seed=0):
    """Token counts (middle, low, high) after each of 12 blocks for a tiered method on random features."""
    from types import SimpleNamespace as NS
    from src.stage_b.timing import from_name
    g = torch.Generator().manual_seed(seed)
    n = n_side * n_side
    t = from_name(name, n_side)
    t.ops = []
    interest = torch.zeros(1, n)
    interest[0, : n // 20] = 1.0                                       # high 5%
    interest[0, n // 20: n // 4] = 0.5                                 # middle 20%
    st = NS(grid=(n_side, n_side), size=torch.ones(1, n), coords=torch.zeros(1, n, 2), interest=interest.clone(),
            src=torch.arange(n)[None], tokens=[], pairs=[], carried=torch.zeros(1, n, D), interest0=interest.clone())
    t.start(st)
    x = torch.randn(1, N_PREFIX + n, D, generator=g)
    out = []
    for b in range(12):
        k = torch.randn(1, 2, N_PREFIX + st.size.shape[1], D, generator=g)
        x = t.after_attn(b, x, k, None, st)
        it = st.interest[0]
        out.append((int(((it >= 0.01) & (it < 0.75)).sum()), int((it < 0.01).sum()), int((it >= 0.75).sum())))
    return t, out


def test_tome_tiered_rate_follows_plain_tome_share():
    from types import SimpleNamespace as NS
    from src.stage_b.timing import from_name
    t = from_name("ttome_r64_l0.7_floor64_dil1_cls64_tr1", 30)
    assert t.name == "ttome_r64_l0.7_floor64_dil1_cls64_tr1" and t.as_rate
    assert not from_name("ttome_r64_l0.7_floor64_dil1_cls64", 30).as_rate
    t.ops = []
    t.start(NS(size=torch.ones(1, 200), interest=None, grid=(10, 20)))
    counts = [t._rate_r(torch.zeros(n, dtype=torch.bool)) for n in (200, 136, 72, 36)]
    assert counts == [64, 64, 36, 18]                                    # no high tier: plain ToMe's count per block
    t.start(NS(size=torch.ones(1, 200), interest=None, grid=(10, 20)))
    half_high = torch.arange(200) < 100
    assert t._rate_r(half_high) == 32                                    # the share 64 / 200 of the 100 non-high tokens


def test_tome_tiered_rate_keeps_the_middle_tier():
    _, fixed = _tiered_run("ttome_r64_l0.7_floor64")
    _, rate = _tiered_run("ttome_r64_l0.7_floor64_tr1")
    mid0 = 1600 // 4 - 1600 // 20
    assert all(h == 80 for _, _, h in fixed + rate)                      # the high tier is never merged
    assert fixed[-1][0] < 0.1 * mid0                                     # R-34: the fixed count collapses the middle tier
    assert rate[-1][0] > 0.45 * mid0                                     # the rate keeps about plain ToMe's share (52%)


def _hold_run(name, frames):
    """Frames of random features: `frames` lists, per frame, the patches whose content is replaced (a surprise)."""
    gh, gw = 12, 16
    cfg = ex.config_from_name(name)
    state = ex.ExpectState()
    w = ex.ExpectMap(_Inner(), state, cfg)
    f = _feat(gh, gw)
    M = torch.zeros(gh, gw, dtype=torch.bool)
    M[1:3, 1:3] = True
    state.objects = [{"id": 1, "M": M, "T": M, "core": M.clone(), "speed": 0.0, "lost": False, "n": 0, "c": (1.5, 1.5),
                      "v": (0.0, 0.0)}]
    out = []
    for i, ch in enumerate(frames):
        f = f.clone()
        for j, (y, x_) in enumerate(ch):
            f[y, x_] = _feat(1, 1, f.shape[-1], 100 + 10 * i + j)[0, 0]
        st, x = _st(f, ex.grow(M, 1))
        w.start(st)
        w.after_attn(0, x, None, None, st)
        out.append(st.interest[0].reshape(gh, gw).clone())
    return out


def test_plan_t_names_and_escalation_and_hold():
    for n in ("exp-x0-mot-rl-r0-e0.5-ring-srch-fh", "exp-x0-mot-rl-r0-e0.5-ring-srch-esc2", "exp-x0-mot-rl-r0-e0.5-ring-srch-hold3",
              "exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1-esc2-hold3"):
        assert ex.config_from_name(n).name == n
    spot = (10, 14)                                                  # far from the object's zone
    base = "exp-x0-rl-r0-e0.5-ring"
    plain = _hold_run(base, [[], [spot], [spot], []])
    assert float(plain[1][spot]) == 0.5 and float(plain[2][spot]) == 0.5 and float(plain[3][spot]) == 0.0
    fh = _hold_run(base + "-fh", [[], [spot]])
    assert float(fh[1][spot]) == 1.0                                 # every fired patch at full resolution
    esc = _hold_run(base + "-esc2", [[], [spot], [spot], [spot]])
    assert float(esc[1][spot]) == 0.5 and float(esc[2][spot]) == 1.0 and float(esc[3][spot]) == 1.0
    hold = _hold_run(base + "-hold2", [[], [spot], [], [], []])
    assert float(hold[1][spot]) == 0.5 and float(hold[2][spot]) == 0.5 and float(hold[3][spot]) == 0.5
    assert float(hold[4][spot]) == 0.0                               # released after two frames without firing
