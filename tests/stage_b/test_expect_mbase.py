"""R40 plan section M (`src/stage_b/expect.py`, `-mb`): a frame whose tracked object moves at the block-matching radius
runs the baseline (the fallback's inner method) on the input a fallback frame gets, so it equals a fallback frame:
outputs, token counts and counted FLOPs; a frame below the radius is unchanged by the option. CPU, fp32."""
import pytest
import torch

from tests.stage_b.test_expect_dev import _davis_or_skip


def test_config_roundtrip():
    from src.stage_b.expect import config_from_name
    name = "exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1-mb"
    c = config_from_name(name)
    assert c.mbase and c.name == name
    assert not config_from_name(name[:-3]).mbase


@pytest.mark.parametrize("fast", [True, False])
def test_mbase_switch_equals_fallback_frame(fast):
    _davis_or_skip()
    from src.stage_a.model import load_backbone
    from src.stage_b.expect import ExpectMap, ExpectState, config_from_name, objects_from_predictions
    from src.stage_b.flops import image_flops
    from src.stage_b.methods import ops_of
    from src.stage_b.model import EffModel
    from src.stage_b.p2_davis import shift_by_velocity
    from src.stage_b.p2_timing import HW
    from src.stage_b.timing import from_name
    from src.stage_b.vos.r40_timing import cov, sequences
    model = EffModel(load_backbone("cpu"))
    blocks = len(model.bb.model.layer)
    inner, fbn = "ttome_r64_l0.7_floor64_dil1_cls64_tr1", "tome_r64_from1_noprop"
    exp = "exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1"
    seq = sequences("cpu", 5, 1)[0]
    runs = {}
    for tag, cfg_name in (("mb", exp + "-mb"), ("plain", exp)):
        cfg = config_from_name(cfg_name)
        est, ost = ExpectState(), {}
        main = ExpectMap(from_name(inner, HW[0] // 16, blocks), est, cfg)
        fb = ExpectMap(from_name(fbn, HW[0] // 16, blocks), est, cfg, passive=True)
        if cfg.mbase:
            main.alt = fb.inner
        outs = []
        for t, (x, ids, p1, p2, _) in enumerate(seq):
            imap = cov((p1 > 0) | (shift_by_velocity(p1, p2) > 0))
            est.objects, _ = objects_from_predictions(p1, p2, ids, ost, cfg.motion)
            last = t == len(seq) - 1
            if last and fast:
                est.objects[0]["speed"] = float(cfg.r_obj)                 # at the radius: the switch fires
            sts = []
            out = model(x, structure=main, dtype=torch.float32, interest=imap, state_out=sts).final
            outs.append((out, sts[0], image_flops(sts[0], ops_of(main), 0, imap.shape[1])["flops"], dict(est.last)))
        runs[tag] = outs
    # the reference for the last frame: the same history, then a fallback frame
    cfg = config_from_name(exp)
    est, ost = ExpectState(), {}
    main = ExpectMap(from_name(inner, HW[0] // 16, blocks), est, cfg)
    fb = ExpectMap(from_name(fbn, HW[0] // 16, blocks), est, cfg, passive=True)
    for t, (x, ids, p1, p2, _) in enumerate(seq):
        imap = cov((p1 > 0) | (shift_by_velocity(p1, p2) > 0))
        est.objects, _ = objects_from_predictions(p1, p2, ids, ost, cfg.motion)
        m = fb if t == len(seq) - 1 else main
        sts = []
        ref = model(x, structure=m, dtype=torch.float32, interest=imap, state_out=sts).final
    ref_st, ref_fl = sts[0], image_flops(sts[0], ops_of(fb), 0, imap.shape[1])["flops"]
    for t in range(len(seq) - 1):                                        # before the last frame the option changes nothing
        assert torch.equal(runs["mb"][t][0], runs["plain"][t][0]), t
    out, st, fl, last = runs["mb"][-1]
    if fast:
        assert last.get("mbase")
        assert torch.equal(out, ref)
        assert [tuple(map(int, a)) for a in st.tokens] == [tuple(map(int, a)) for a in ref_st.tokens]
        assert abs(fl - ref_fl) <= 1e-9 * ref_fl
    else:
        assert not last.get("mbase")
        assert torch.equal(out, runs["plain"][-1][0])
