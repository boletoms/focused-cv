"""R40 plan section L1 (`src/stage_b/frame_runner.py`): per frame, the frame runner (device steps, one decision read,
capacities from the previous frame) equals the current path (`objects_from_predictions`, `shift_by_velocity`, the
wrappers and `EffModel`): the fallback decision, token counts, patch assignment, outputs within 1e-4 and counted FLOPs
(on fallback frames without the expectation products, which the frame runner no longer computes there). With a
negative margin most capacities are too small: those frames run again with per-block reads and still agree. CPU."""
import pytest
import torch

from tests.stage_b.test_expect_dev import _davis_or_skip, _frames


@pytest.mark.parametrize("margin", [32, -64])
def test_frame_runner_matches_current(margin):
    _davis_or_skip()
    from src.stage_a.model import load_backbone
    from src.stage_b.expect import ExpectMap, ExpectState, config_from_name, objects_from_predictions
    from src.stage_b.flops import image_flops
    from src.stage_b.frame_runner import FrameRunner
    from src.stage_b.methods import ops_of
    from src.stage_b.model import EffModel
    from src.stage_b.p2_davis import shift_by_velocity
    from src.stage_b.p2_timing import HW
    from src.stage_b.timing import from_name
    from src.stage_b.vos.r40_compare import S_FINAL
    from src.stage_b.vos.r40_timing import cov
    model = EffModel(load_backbone("cpu"))
    blocks = len(model.bb.model.layer)
    inner, rest = S_FINAL.split("~")
    fbn, exp = rest.split("@")
    cfg = config_from_name(exp)
    fr = FrameRunner(model, HW, S_FINAL, margin=margin, fast=False, dtype=torch.float32, graphs=False)
    seen = {"fallback": 0, "main": 0}
    for seq in _frames(14, 2, drop_all=(9, 10)):
        est, ost = ExpectState(), {}
        main = ExpectMap(from_name(inner, HW[0] // 16, blocks), est, cfg)
        fb = ExpectMap(from_name(fbn, HW[0] // 16, blocks), est, cfg, passive=True)
        fr.reset()
        for t, (x, ids, p1, p2) in enumerate(seq):
            imap = cov((p1 > 0) | (shift_by_velocity(p1, p2) > 0))
            est.objects, all_lost = objects_from_predictions(p1, p2, ids, ost, cfg.motion)
            m = fb if (not bool(imap.any()) or all_lost) else main
            sts = []
            ref = model(x, structure=m, dtype=torch.float32, interest=imap, state_out=sts).final
            out, st, ops, use_fb = fr.run(x, ids, p1, p2)
            assert use_fb == (m is fb), t
            assert [tuple(map(int, a)) for a in sts[0].tokens] == [tuple(map(int, a)) for a in st.tokens], t
            assert torch.equal(sts[0].src, st.src), t
            assert (out - ref).abs().max() < 1e-4, (t, float((out - ref).abs().max()))
            N0 = imap.shape[1]
            ref_ops = [o for o in ops_of(m) if not (use_fb and o[1].startswith("expect"))]
            fa, fb_ = image_flops(sts[0], ref_ops, 0, N0)["flops"], image_flops(st, ops, 0, N0)["flops"]
            assert abs(fa - fb_) <= 1e-9 * fa, (t, fa, fb_)
            seen["fallback" if use_fb else "main"] += 1
    assert seen["main"] > 5 and seen["fallback"] > 0, seen
    if margin < 0:
        assert fr.stats["overflow"] > 0
