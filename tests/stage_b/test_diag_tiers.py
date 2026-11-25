"""The diagnostics log records the expectation focus map's final tiers and the share of an object's ground truth inside
them (logging only)."""
import numpy as np
import torch

from src.stage_b.vos.diag import DiagLog


def test_tier_coverage_logged(tmp_path):
    h, w = 64, 96
    gt = torch.zeros(h, w, dtype=torch.long)
    gt[16:48, 32:64] = 1                                             # patches rows 1-2, columns 2-3
    prob = torch.zeros(2, h, w)
    prob[1] = (gt == 1).float()
    gh, gw = 4, 6
    th = torch.zeros(gh, gw, dtype=torch.bool)
    th[1, 2] = th[2, 2] = True                                       # half of the object in the high tier
    tm = torch.zeros_like(th)
    tm[1, 3] = True                                                  # a quarter in the middle tier
    d = DiagLog(tmp_path, "v", "s", 0, [1], with_gt=False)
    args = (0, {1: -1}, gt > 0, prob, gt, {0: 1}, gt, 0.4, False, False, False, False, False, None, {1: (1.0, 1.0)})
    d.frame(*args, tiers=(th, tm), cam=(1, -2))
    d.frame(*args)                                                   # a frame without tiers (fallback, passive)
    d.save()
    z = np.load(tmp_path / "diag" / "v" / "0.npz")
    assert z["tier_h"].shape[0] == 2 and z["tier_m"].shape[0] == 2
    assert abs(z["cov_h"][0, 0] - 0.5) < 1e-6 and abs(z["cov_hm"][0, 0] - 0.75) < 1e-6
    assert z["cam_y"][0] == 1 and z["cam_x"][0] == -2 and z["cam_y"][1] == -999
    assert np.isnan(z["cov_h"][1, 0]) and np.isnan(z["cov_hm"][1, 0])
