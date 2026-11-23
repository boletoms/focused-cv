"""R1 baseline ports (src/stage_b/baselines.py) against the authors' code where it imports only torch/einops
(cloned untouched in /work/third_party), plus invariants on the real backbone. CPU, fp32, 224 pixels."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.stage_a.data import ImageList
from src.stage_a.model import N_PREFIX, load_backbone
from src.stage_b import baselines as bl
from src.stage_b.flops import dense_flops, image_flops
from src.stage_b.methods import ops_of
from src.stage_b.model import EffModel, State

TP = Path("/work/third_party")


def load(path, name):
    if not path.exists():
        pytest.skip(f"{path} not available")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def state(n, gh, gw):
    st = State(size=torch.ones(1, n), coords=torch.rand(1, n, 2), src=torch.arange(n)[None], carried=None,
               pairs=[], tokens=[], tiles=[])
    st.grid = (gh, gw)
    return st


def as_set(x):
    """Tokens as a sorted list of rows (merge order differs between implementations)."""
    return x[0][torch.argsort(x[0][:, 0])]


def test_expedite_clustering_matches_original():
    mod = load(TP / "Expedit-Segmenter/mmseg/ops/cluster.py", "exp_cluster")
    gh = gw = 14
    x = torch.randn(1, N_PREFIX + gh * gw, 32)
    ref_feat, ref_labels = mod.TokenClusteringBlock(num_spixels=(7, 7), n_iters=5, temperture=0.02, window_size=5)(
        x[:, N_PREFIX:].transpose(1, 2).reshape(1, 32, gh, gw))
    m = bl.Expedite(0.25, location=0)
    st = state(gh * gw, gh, gw)
    out = m.before_block(1, x, st)
    assert torch.allclose(out[0, N_PREFIX:], ref_feat[0], atol=1e-4)
    assert torch.equal(st.src[0], ref_labels[0])


def test_expedite_reconstruction_matches_original(monkeypatch):
    mod = load(TP / "Expedit-Segmenter/mmseg/ops/reconstruction.py", "exp_rec")
    feat, sfeat, out = torch.randn(1, 50, 16), torch.randn(1, 9, 16), torch.randn(1, 9, 16)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)    # the original moves a mask to CUDA
    blk = mod.TokenReconstructionBlock(cfg={"k": 4, "temperture": 0.02})
    s = mod.State(blk)
    s.update_state(feat_before_pooling=feat, feat_after_pooling=sfeat)
    ref, _ = s.call(out)
    m = bl.Expedite(0.25, k=4)
    m.before, m.after = feat[0], sfeat[0]
    st = state(50, 5, 10)
    got = m.reconstruct_out(torch.cat([torch.zeros(1, N_PREFIX, 16), out], 1), st)
    assert torch.allclose(got[0, N_PREFIX:], ref[0], atol=1e-5)


@pytest.mark.parametrize("block", [1, 9])                      # bsm half and pitome half
def test_pitome_matches_original(block):
    mod = load(TP / "PiToMe/algo/pitome/merge.py", "pitome_merge")
    n, hd, heads = 60, 16, 2
    k = torch.randn(1, heads, N_PREFIX + n, hd)
    x = torch.randn(1, N_PREFIX + n, 24)
    keep = 0.8
    margin = 0.75 - 0.75 * block / 12
    metric = k[:, :, N_PREFIX:].mean(1)
    merge = mod.pitome_vision(metric=metric, ratio=keep, margin=margin, class_token=False,
                              use_bsm_pitome=block < 6)
    ref = merge(x[:, N_PREFIX:], mode="mean")
    m = bl.PiToMe(keep)
    got = m.after_attn(block, x, k, None, state(n, 6, 10))
    assert got.shape[1] - N_PREFIX == ref.shape[1]
    assert torch.allclose(as_set(got[:, N_PREFIX:]), as_set(ref), atol=1e-5)


def test_algm_matches_original(monkeypatch):
    lm = load(TP / "algm-segmenter/algm/algm/local_merge.py", "algm_local")
    gm = load(TP / "algm-segmenter/algm/algm/global_merge.py", "algm_global")
    gh = gw = 8
    n = gh * gw
    base = torch.randn(1, 1 + n, 16)
    base[:, 1:] += 3 * torch.randn(1, 1, 16)                    # similar tokens so some windows merge
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    merge = lm.conditional_pooling(base, 0.5, (2, 2))
    ref, size = lm.merge_wavg(merge, base)
    m = bl.ALGM(0.5)
    x = torch.cat([torch.zeros(1, N_PREFIX - 1, 16), base], 1)  # our prefix: CLS + registers
    st = state(n, gh, gw)
    got = m.after_attn(0, x, None, None, st)
    assert torch.allclose(as_set(got[:, N_PREFIX:]), as_set(ref[:, 1:]), atol=1e-5)
    assert torch.allclose(torch.sort(st.size[0]).values, torch.sort(size[0, 1:, 0]).values)
    # global matching without a class token: same tokens merged
    y = torch.randn(1, n, 16)
    src = torch.eye(n)[None]
    gmerge = gm.turbo_matching(y, 5, src, class_token=False)
    ref2, _ = lm.merge_wavg(gmerge, y)
    got2 = m.after_attn(4, torch.cat([torch.zeros(1, N_PREFIX, 16), y], 1), None, None, state(n, gh, gw))
    assert torch.allclose(as_set(got2[:, N_PREFIX:]), as_set(ref2), atol=1e-5)


def _g2tm_ref():
    """The authors' core package (utils, connected components, BFS and FastSV merges import only torch); its
    __init__ imports Segmenter, so the package is registered bare."""
    import importlib
    import types
    root = TP / "g2tm/g2tm/g2tm"
    if not (root / "utils.py").exists():
        pytest.skip(f"{root} not available (branch segm)")
    pkg = types.ModuleType("g2tm_ref")
    pkg.__path__ = [str(root)]
    sys.modules["g2tm_ref"] = pkg
    return importlib.import_module("g2tm_ref.bfs_merge"), importlib.import_module("g2tm_ref.fast_sv_merge")


@pytest.mark.parametrize("seed", range(5))
def test_g2tm_matches_original(seed):
    """Square grid, CLS only in the original: same components, same merged tokens and sizes, same order, with both
    of the authors' labelling paths (BFS and FastSV)."""
    bfs, fsv = _g2tm_ref()
    g = torch.Generator().manual_seed(seed)
    gh = gw = 8
    n = gh * gw
    base = torch.randn(1, 1 + n, 16, generator=g)
    off = 2.5 * torch.randn(2, 16, generator=g)                       # two regions (left / right half) of similar tokens
    base[0, 1:] += off[(torch.arange(n) % gw >= gw // 2).long()]
    ref, size, _, unmerge = bfs.bfs_merge(base, 0.8)
    ref2, size2, _, _ = fsv.fast_sv_merge(base, 0.8)
    assert torch.allclose(ref, ref2) and torch.equal(size, size2)
    m = bl.G2TM(0.8, at=1)
    x = torch.cat([torch.zeros(1, N_PREFIX - 1, 16), base], 1)       # our prefix: CLS + registers
    st = state(n, gh, gw)
    got = m.after_attn(0, x, None, None, st)
    assert 1 < ref.shape[1] - 1 < n, "test input should merge some tokens"
    assert torch.allclose(got[:, N_PREFIX:], ref[:, 1:], atol=1e-5)
    assert torch.equal(st.size[0].long(), size[0, 1:])
    assert torch.equal(st.src[0], unmerge[0])                         # copy-back index onto the grid


def test_g2tm_rectangular_grid_components():
    """Non-square grids (our 28 x 49): components equal a reference BFS on the same edges."""
    bfs, _ = _g2tm_ref()
    from g2tm_ref.connected_components import connected_components_bfs
    g = torch.Generator().manual_seed(7)
    gh, gw = 5, 9
    feat = torch.randn(gh * gw, 8, generator=g) + 2 * torch.randn(1, 8, generator=g)
    right, bottom = bl._neighbour_masks(feat, 0.7, gh, gw)
    lab = bl._fastsv_labels(right, bottom, gw)[0]
    mean_i, red_i, mrg_i = connected_components_bfs(right[0], bottom[0], gh * gw, gw, "cpu")
    ref = torch.arange(gh * gw)
    if red_i.numel():
        ref[red_i] = mean_i[mrg_i.reshape(-1)]
    assert torch.equal(lab, ref)


def test_g2tm_threshold_above_one_is_dense():
    gh = gw = 6
    x = torch.randn(1, N_PREFIX + gh * gw, 16)
    st = state(gh * gw, gh, gw)
    m = bl.G2TM(1.01, at=1)
    assert torch.equal(m.after_attn(0, x, None, None, st), x)
    assert st.size.shape[1] == gh * gw


def test_g2tm_protected_cuts_region_edges():
    gh = gw = 8
    n = gh * gw
    x = torch.cat([torch.zeros(1, N_PREFIX, 16), torch.ones(1, n, 16) + 0.01 * torch.randn(1, n, 16)], 1)
    st = state(n, gh, gw)
    st.interest = torch.zeros(1, n)
    st.interest[0, 18:21] = 1.0
    m = bl.G2TMProtect(0.5, at=1)
    m.after_attn(0, x, None, None, st)
    src = st.src[0]
    for i in range(18, 21):
        assert (src == src[i]).sum() == 1, "a protected patch was merged"
    assert len(torch.unique(src)) == 4                                # three region singletons + one component


@pytest.fixture(scope="module")
def model_and_image():
    model = EffModel(load_backbone("cpu"))
    ds = ImageList("calibration", 224)
    return model, ds[0][0][None]


@pytest.mark.parametrize("name", ["pitome_k0.9", "algm_t0.6", "ailurus_m0.39_at2", "expedite_m0.49_at4_t0.1", "g2tm_t0.95_at2"])
def test_baselines_on_backbone(model_and_image, name):
    from src.stage_b.timing import from_name
    model, x = model_and_image
    m = from_name(name, 14)
    sts = []
    with torch.no_grad():
        out = model(x, structure=m, state_out=sts).final
    st = sts[0]
    assert out.shape == (1, N_PREFIX + 196, 384) and torch.isfinite(out).all()
    fl = image_flops(st, ops_of(m), 0, 196)["flops"] / dense_flops(196)
    assert 0.2 < fl < 1.0, fl
    dense = model(x).final
    cos = F.cosine_similarity(out[0, N_PREFIX:], dense[0, N_PREFIX:], dim=-1).mean()
    assert cos > 0.5, cos


@pytest.mark.parametrize("name", ["ptome_r20", "palgm_t0.6", "ptome_r20_dil1", "pg2tm_t0.5_at2", "pg2tm_t0.5_at2_dil1"])
def test_protected_baselines_keep_region(model_and_image, name):
    """R6: no protected patch shares a token with any other patch; without a map they equal the plain baseline."""
    from src.stage_b.timing import from_name
    model, x = model_and_image
    interest = torch.zeros(1, 196)
    interest[0, 30:40] = 1.0
    sts = []
    with torch.no_grad():
        model(x, structure=from_name(name, 14), state_out=sts, interest=interest)
    st = sts[0]
    prot = torch.zeros(196, dtype=torch.bool)
    prot[30:40] = True
    if "dil1" in name:
        prot = F.max_pool2d(prot.float().view(1, 1, 14, 14), 3, 1, 1).flatten().bool()
    src = st.src[0]
    for t in src[prot].tolist():
        assert (src == t).sum() == 1, "a protected patch was merged"
    plain = from_name(name.replace("ptome", "tome").replace("palgm", "algm").replace("pg2tm", "g2tm").replace("_dil1", "") + ("_from1_noprop" if "ptome" in name else ""), 14)
    a, b_ = model(x, structure=from_name(name, 14)).final, model(x, structure=plain).final
    assert torch.allclose(a, b_, atol=1e-4)


def _energy(model, x, interest, at=0):
    """Keys of block `at` on the dense model, PiToMe energy (as Novelty)."""
    from src.stage_b.methods import Method

    class Grab(Method):
        def after_attn(self, b, x_, k, p, st):
            if b == at:
                self.k = k
            return x_
    g = Grab()
    with torch.no_grad():
        model(x, structure=g, interest=interest)
    m = F.normalize(g.k[0, :, N_PREFIX:].float().mean(0), dim=-1)
    return F.elu(m @ m.T - 0.75, alpha=1.0).mean(-1)


@pytest.mark.parametrize("inner", ["palgm_t0.6_dil1", "pg2tm_t0.95_at1_dil1", "ftome_l0.5_t0.01_floor16_dil1_noprop"])
def test_novelty_protects_lowest_energy(model_and_image, inner):
    """R19: the fraction q of unprotected patches with the lowest energy at block 0 is never merged (stays a
    singleton), on top of the region; q = 0 equals the inner method."""
    from src.stage_b.timing import from_name
    model, x = model_and_image
    interest = torch.zeros(1, 196)
    interest[0, 30:40] = 1.0
    prot = F.max_pool2d((interest > 0).float().view(1, 1, 14, 14), 3, 1, 1).flatten().bool()
    q = 0.1
    e = _energy(model, x, interest)
    free = torch.nonzero(~prot)[:, 0]
    novel = free[torch.argsort(e[free], stable=True)[:int(q * len(free))]]
    sts = []
    with torch.no_grad():
        out = model(x, structure=from_name(f"nov{q}:{inner}", 14), state_out=sts, interest=interest).final
    src = sts[0].src[0]
    for i in torch.cat([novel, torch.nonzero(prot)[:, 0]]).tolist():
        assert (src == src[i]).sum() == 1, f"patch {i} was merged"
    assert torch.isfinite(out).all()
    sts0 = []
    with torch.no_grad():
        model(x, structure=from_name(inner, 14), state_out=sts0, interest=interest)
    s0 = sts0[0].src[0]
    assert any((s0 == s0[i]).sum() > 1 for i in novel.tolist()), "without the wrapper some novel patches merge"
    a = model(x, structure=from_name(f"nov0:{inner}", 14), interest=interest).final
    b_ = model(x, structure=from_name(inner, 14), interest=interest).final
    assert torch.allclose(a, b_, atol=1e-5)

