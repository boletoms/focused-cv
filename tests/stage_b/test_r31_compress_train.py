"""R31: focus maps for training with compression in the loop, and the encoder's per-image compressed path."""
import os

import pytest
import torch


def test_maps_from_gt_previous_frame_shift_and_empty():
    from src.stage_b.vos import train as tr
    g = torch.zeros(2, 4, 1, 64, 64, dtype=torch.long)
    g[:, 0, 0, 20:30, 20:30] = 1
    g[:, 1, 0, 30:40, 30:40] = 2
    g[:, 2, 0, 40:50, 40:50] = 1
    gen = torch.Generator().manual_seed(0)
    old = tr.EMPTY
    try:
        tr.EMPTY = 0.0
        m = tr.maps_from_gt(g, gen).view(2, 4, 64, 64)
    finally:
        tr.EMPTY = old
    assert m.shape == (2, 4, 64, 64) and m.dtype == torch.bool
    for b in range(2):
        for t, (y0, src) in enumerate([(20, 0), (20, 0), (30, 1), (40, 2)]):
            ys, xs = torch.nonzero(m[b, t], as_tuple=True)
            assert len(ys) == 100                                         # moved whole (stays inside the frame)
            assert abs(int(ys.min()) - y0) <= tr.SHIFT and abs(int(xs.min()) - y0) <= tr.SHIFT
    gen = torch.Generator().manual_seed(1)
    big = torch.ones(50, 4, 1, 32, 32, dtype=torch.long)
    share = (tr.maps_from_gt(big, gen).flatten(1).sum(1) == 0).float().mean()
    assert 0.08 < float(share) < 0.25                                     # EMPTY 0.15


@pytest.fixture(scope="module")
def encoder():
    os.environ["VOS_COMPRESS_TRAIN"] = "ftome_l0.7_t0.01_floor64_dil1_noprop_cls64~tome_r64_from1_noprop"
    try:
        from src.stage_b.vos.dino_encoder import DinoPixelEncoder
        enc = DinoPixelEncoder({"resnet_model_path": None}).eval()
    except Exception as e:                                                 # ResNet weights not cached: skip
        pytest.skip(f"encoder unavailable: {e}")
    finally:
        os.environ.pop("VOS_COMPRESS_TRAIN", None)
    return enc


def test_encoder_compressed_path_per_image(encoder):
    from src.stage_a.model import N_PREFIX
    from src.stage_b.timing import from_name
    torch.manual_seed(0)
    x = torch.randn(3, 3, 96, 96)
    maps = torch.zeros(3, 96, 96, dtype=torch.bool)
    maps[0, 16:48, 16:48] = True
    maps[1, 48:80, 32:64] = True                                           # image 2: empty map -> fallback
    encoder.train_maps = maps
    with torch.no_grad():
        f16, f8, f4 = encoder(x)
    assert encoder.train_maps is None and f16.shape == (3, 384, 6, 6)
    with torch.no_grad():
        for i in range(3):
            interest = torch.nn.functional.avg_pool2d(maps[i].float()[None, None], 16)[0, 0].flatten()[None]
            m = encoder.structure if bool(interest.any()) else encoder.train_fb
            ref = encoder.vit(x[i:i + 1], structure=m, dtype=torch.bfloat16,
                              interest=interest if m is encoder.structure else None).final[:, N_PREFIX:]
            assert torch.allclose(f16[i], ref[0].float().T.reshape(384, 6, 6), atol=1e-5)
    assert encoder.train_fb is not None and encoder.train_fb.name.startswith("tome_r64")
