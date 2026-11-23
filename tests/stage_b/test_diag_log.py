"""Long-video diagnostics logger (`src/stage_b/vos/diag.py`): grid padding as the encoder's, protected patches as the
methods' (coverage >= 0.01, one-patch margin on the grid), exact coverage shares, round trip through the file. CPU."""
import json

import numpy as np
import torch

from src.stage_b.vos.diag import DiagLog, to_grid


def test_to_grid_pads_symmetrically_like_the_encoder():
    x = torch.zeros(40, 50)                                   # padded to 48 x 64: 4 rows top, 7 columns left
    x[0, 0] = 1.0                                             # lands in patch (0, 0) at offset (4, 7)
    g = to_grid(x)
    assert g.shape == (3, 4) and g[0, 0] == 1 / 256 and g.sum() == 1 / 256
    m = torch.zeros(40, 50)
    m[:, :] = 1
    H, W = 48, 64
    lh, lw = (H - 40) // 2, (W - 50) // 2
    ref = torch.zeros(H, W)
    ref[lh:lh + 40, lw:lw + 50] = 1                           # as dino_encoder builds the focus map
    assert torch.allclose(to_grid(m), torch.nn.functional.avg_pool2d(ref[None, None], 16)[0, 0])


def _frame(log, t, interest, gt, pred, probs):
    first = {1: 0, 2: 5}
    prob = torch.stack([1 - probs.sum(0)] + list(probs))
    jf = {o: (0.5, 0.25) for o in first if t > first[o]}
    log.frame(t, first, interest, prob, pred, {1: 1, 2: 2}, gt, 0.3, False, False, not bool(interest.any()),
              False, t % 5 == 0, None, jf)


def test_coverage_and_round_trip(tmp_path):
    h, w = 64, 96                                             # no padding: 4 x 6 patches
    log = DiagLog(tmp_path, "vid", "dense", 0, [1, 2], with_gt=True)
    interest = torch.zeros(h, w, dtype=torch.bool)
    interest[0:16, 0:16] = True                               # map on patch (0, 0); protected: patches (0..1, 0..1)
    gt = torch.zeros(h, w, dtype=torch.long)
    gt[0:16, 16:32] = 1                                       # object 1 on patch (0, 1): protected (margin)
    gt[48:64, 80:96] = 2                                      # object 2 on patch (3, 5): outside
    pred = torch.zeros(h, w, dtype=torch.long)
    pred[0:16, 16:24] = 1
    probs = torch.zeros(2, h, w)
    probs[0, 0:16, 16:32] = 0.8
    probs[1, 48:64, 80:96] = 0.3
    for t in range(7):
        _frame(log, t, interest, gt, pred, probs)
    log.save()
    d = np.load(tmp_path / "diag" / "vid" / "0.npz")
    assert json.loads((tmp_path / "diag" / "vid" / "settings.json").read_text()) == {"0": "dense"}
    assert d["t"].tolist() == list(range(7)) and d["mem"].tolist() == [t % 5 == 0 for t in range(7)]
    mg = np.unpackbits(d["map_grid"][0])[:24].reshape(4, 6)
    assert mg.sum() == 1 and mg[0, 0] == 1
    assert d["map_cov"][6].tolist() == [1.0, 0.0]
    assert np.isnan(d["j"][3, 1]) and d["j"][6, 1] == 0.5 and np.isnan(d["j"][0, 0]) and d["j"][1, 0] == 0.5
    assert d["gt_px"][6].tolist() == [256, 256] and d["pred_px"][6].tolist() == [128, 0] and d["inter_px"][6, 0] == 128
    assert abs(d["prob_gt"][6, 0] - 0.8) < 1e-6
    assert abs(d["prob_out"][6, 1] - 0.3) < 1e-6 and d["prob_out"][6, 0] == 0.0   # object 1's evidence is protected
    assert d["prob_grid"].shape == (7, 2, 4, 6) and d["prob_grid"][6, 0, 0, 1] == round(0.8 * 255)
    assert d["gt_grid"][6, 1, 3, 5] == 255
    assert tuple(d["gt_c"][6, 0]) == (7.5, 23.5)
