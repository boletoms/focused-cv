"""R40 plan section L1 (`src/stage_b/expect_dev.py`): the device forms of the map motion, the object-state update, the
map preparation and the expectation step equal the current code (`shift_by_velocity`, `objects_from_predictions`,
the inner method's start and `ExpectMap._step`) on DAVIS sequences, with objects dropped from the prediction for a few
frames so that losses, search regions and re-acquisition occur. CPU, fp32."""
import pytest
import torch

from src.stage_b import expect_dev as ed


def _davis_or_skip():
    from src.stage_b.p2_timing import ROOT
    if not ROOT.exists():
        pytest.skip(f"{ROOT} not available")


def _frames(n_frames=24, n_seqs=3, drop_all=()):
    """Per sequence: per frame (pixels, ids seen before t, id map t − 1, id map t − 2) with one object dropped from the
    'prediction' at frames 6..11 (lost, then found again), the second object shrunk below 16 pixels at 14..15, and
    every object dropped at the frames in `drop_all` (fallback frames)."""
    from src.stage_b.vos.r40_timing import sequences
    out = []
    for seq in sequences("cpu", n_frames, n_seqs):
        maps = [seq[0][3]] + [fr[2] for fr in seq]                         # id maps t − 2 (first), then t − 1 per frame
        objs = sorted(set(int(o) for o in torch.unique(maps[-1]).tolist()) - {0}) or [1]
        drop = objs[0]
        mod = []
        for t, m in enumerate(maps):
            m = m.clone()
            if 6 <= t < 12:
                m[m == drop] = 0
            if t in drop_all:
                m = torch.zeros_like(m)
            if 14 <= t < 16 and len(objs) > 1:
                ys, xs = torch.nonzero(m == objs[1], as_tuple=True)
                keep = torch.zeros_like(m, dtype=torch.bool)
                keep[ys[:8], xs[:8]] = True
                m[(m == objs[1]) & ~keep] = 0
            mod.append(m)
        out.append([(fr[0], fr[1], mod[t + 1], mod[t]) for t, fr in enumerate(seq)])
    return out


def test_shift_and_grow_match_current():
    from src.stage_b.expect import grow, shift
    g = torch.Generator().manual_seed(0)
    m = torch.rand(5, 9, 13, generator=g) > 0.85
    d = torch.tensor([[0, 0], [2, -3], [-9, 1], [4, 13], [-1, -1]])
    r = torch.tensor([0, 1, 2, 5, 20])
    for k in range(5):
        assert torch.equal(ed.shift_dev(m, d)[k], shift(m[k], *d[k].tolist()))
        assert torch.equal(ed.grow_dev(m, r)[k], grow(m[k], int(r[k])) if r[k] > 0 else m[k])
    # R-38: radii at and past H + W + 1 (a search region 2 + n after a long loss), on masks moved out of the grid
    # (empty) and on non-empty ones: empty stays empty, anything else covers the grid, as `grow`
    m2 = torch.stack([ed.shift_dev(m, d)[3], m[1], torch.zeros_like(m[0]), m[2]])
    r2 = torch.tensor([9 + 13 + 1, 9 + 13 + 1, 500, 9 + 13])
    assert not m2[0].any() and m2[1].any()
    for k in range(4):
        assert torch.equal(ed.grow_dev(m2, r2)[k], grow(m2[k], int(r2[k]))), k


def test_moved_union_matches_current():
    _davis_or_skip()
    from src.stage_b.p2_davis import shift_by_velocity
    n = 0
    for seq in _frames(12, 3):
        for x, ids, p1, p2 in seq:
            ref = shift_by_velocity(p1, p2) > 0
            got = ed.moved_union(p1, p2, torch.tensor(ids or [0]))
            assert torch.equal(ref, got)
            n += 1
    assert n > 20


def test_objects_match_current():
    _davis_or_skip()
    from src.stage_b.expect import objects_from_predictions
    seen_lost = 0
    for seq in _frames():
        ost, dev = {}, ed.DevObjects(motion=True)
        for t, (x, ids, p1, p2) in enumerate(seq):
            ref, all_ref = objects_from_predictions(p1, p2 if t else None, ids, ost, True)
            got, all_got = dev.update(p1, p2 if t else None, ids)
            assert bool(all_got) == all_ref, t
            for k, o in enumerate(ref):
                for key in ("M", "T", "core"):
                    assert torch.equal(o[key], got[key][k]), (t, key)
                assert bool(got["lost"][k]) == o["lost"] and int(got["n"][k]) == o["n"], t
                # centroids: exact integer sums here, a float32 mean in the current code (about 1e-6 patch apart)
                assert abs(float(got["speed"][k]) - o["speed"]) < 1e-5, t
                assert torch.allclose(got["v"][k], torch.tensor(o["v"], dtype=torch.float64), atol=1e-5), t
                if o["Mlast"] is None:
                    assert not bool(got["hasM"][k])
                else:
                    assert bool(got["hasM"][k]) and torch.equal(got["Mlast"][k], o["Mlast"])
                seen_lost += o["lost"]
    assert seen_lost > 0, "the dropped objects should be lost on some frames"


def _setting():
    from src.stage_b.expect import config_from_name
    from src.stage_b.vos.r40_compare import S_FINAL
    inner, rest = S_FINAL.split("~")
    fb, exp = rest.split("@")
    return inner, fb, config_from_name(exp)


def test_prepare_map_matches_inner_start():
    _davis_or_skip()
    from src.stage_b.model import State
    from src.stage_b.p2_timing import HW
    from src.stage_b.timing import from_name
    from src.stage_b.vos.r40_timing import cov
    inner_name, _, _ = _setting()
    grid = (HW[0] // 16, HW[1] // 16)
    n = 0
    for seq in _frames(10, 3):
        for x, ids, p1, p2 in seq:
            imap = cov(p1)
            m = from_name(inner_name, grid[0], 12)
            st = State(size=None, coords=None, src=None, carried=None, pairs=[], tokens=[], tiles=[], interest=imap.clone())
            st.grid = grid
            m.start(st)
            ref = (st.interest >= m.theta).float()
            d = ed.dilate_map(imap, m.dilate, grid)
            k = ed.promote_count(int((d[0] >= m.theta).sum()), m.size_class, d.shape[1])
            got = ed.promote_map(d, m.theta, m.size_class, grid, k)
            assert torch.equal(ref, got)
            n += 1
    assert n > 10


def test_expect_step_matches_current():
    """Closed loop per sequence: the current wrappers (`tiered_graph.plan`) against `ExpectDev`, fallback frames when
    every object is lost or the map is empty; the tiered maps must be equal on every frame."""
    _davis_or_skip()
    from src.stage_b.expect import ExpectMap, ExpectState, objects_from_predictions
    from src.stage_b.p2_davis import shift_by_velocity
    from src.stage_b.p2_timing import HW
    from src.stage_b.tiered_graph import plan
    from src.stage_b.timing import from_name
    from src.stage_b.vos.r40_timing import cov
    inner_name, fb_name, cfg = _setting()
    grid = (HW[0] // 16, HW[1] // 16)
    tiers, fallbacks = 0, 0
    for seq in _frames():
        est, ost = ExpectState(), {}
        main = ExpectMap(from_name(inner_name, grid[0], 12), est, cfg)
        fb = ExpectMap(from_name(fb_name, grid[0], 12), est, cfg, passive=True)
        inner = main.inner
        exd, dobj = ed.ExpectDev(cfg, inner.theta, inner.srp_value, grid, "cpu"), ed.DevObjects(cfg.motion)
        for t, (x, ids, p1, p2) in enumerate(seq):
            imap = cov((p1 > 0) | (shift_by_velocity(p1, p2) > 0))
            est.objects, all_lost = objects_from_predictions(p1, p2, ids, ost, cfg.motion)
            use_fb = not bool(imap.any()) or all_lost
            st = plan(fb if use_fb else main, x, imap)
            objs, all_dev = dobj.update(p1, p2, ids)
            imap_d = cov(ed.moved_union(p1, p2, torch.tensor(ids)) | (p1 > 0))
            assert torch.equal(imap, imap_d) and bool(all_dev) == all_lost
            dm = ed.dilate_map(imap_d, inner.dilate, grid)
            k = ed.promote_count(int((dm[0] >= inner.theta).sum()), inner.size_class, dm.shape[1])
            mp = ed.promote_map(dm, inner.theta, inner.size_class, grid, k)
            got, _ = exd.step(x, None if use_fb else mp, objs, use_fb)
            if use_fb:
                fallbacks += 1
                continue
            assert torch.equal(st.interest, got), (t, (st.interest != got).sum())
            last = est.last
            if not last.get("first"):
                assert (int(exd.last[0]), int(exd.last[1]), bool(exd.last[2])) == (last["high"], last["mid"], last["global"])
                tiers += last["high"] + last["mid"]
    assert tiers > 0
