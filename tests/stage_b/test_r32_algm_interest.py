"""R32 `ALGMInterest` (`aitome_*`, src/stage_b/baselines.py): ALGM inside the protected set, our budget outside, classes
never mixed; the outside budget is exactly InterestToMe's; surprise-protected tokens join the ALGM class. CPU, fp32,
224 pixels (14 x 14 patches)."""
import pytest
import torch
import torch.nn.functional as F

from src.stage_a.data import ImageList
from src.stage_a.model import N_PREFIX, load_backbone
from src.stage_b.flops import dense_flops, image_flops
from src.stage_b.methods import InterestToMe, ops_of
from src.stage_b.model import EffModel


@pytest.fixture(scope="module")
def model_and_image():
    model = EffModel(load_backbone("cpu"))
    ds = ImageList("calibration", 224)
    return model, ds[0][0][None]


def _map():
    m = torch.zeros(14, 14)
    m[2:10, 3:11] = 1.0                                               # 8 x 8 region: whole 2 x 2 windows inside
    return m.flatten()[None]


def _run(model, x, m, interest):
    sts = []
    with torch.no_grad():
        out = model(x, structure=m, state_out=sts, interest=interest).final
    return out, sts[0]


def test_name_round_trip():
    from src.stage_b.timing import from_name
    for n in ("aitome_t0.98_l0.7_floor64_dil1_cls64", "aitome_t0.9_l0.5_floor16"):
        assert from_name(n, 14).name == n


@pytest.mark.parametrize("name", ["aitome_t0.5_l0.5_floor16", "aitome_t0.98_l0.7_floor16"])
def test_classes_never_mix_and_region_merges(model_and_image, name):
    from src.stage_b.timing import from_name
    model, x = model_and_image
    out, st = _run(model, x, from_name(name, 14), _map())
    assert torch.isfinite(out).all()
    prot = _map()[0] >= 0.01
    src = st.src[0]
    for t in torch.unique(src).tolist():
        members = prot[src == t]
        assert bool(members.all()) or not bool(members.any()), "a token holds protected and out-of-focus patches"
    n_prot_tokens = len(torch.unique(src[prot]))
    assert n_prot_tokens < int(prot.sum()), "ALGM merged nothing inside the region"


def test_outside_budget_is_interest_tome(model_and_image):
    """Out-of-focus tokens follow InterestToMe exactly in count (the region's merges never touch them)."""
    from src.stage_b.timing import from_name
    model, x = model_and_image
    prot = _map()[0] >= 0.01
    _, st_a = _run(model, x, from_name("aitome_t0.5_l0.5_floor16", 14), _map())
    _, st_i = _run(model, x, InterestToMe(0.5, theta=0.01, floor=16, prop=False), _map())
    low_a = len(torch.unique(st_a.src[0][~prot]))
    low_i = len(torch.unique(st_i.src[0][~prot]))
    assert low_a == low_i >= 16
    assert len(torch.unique(st_i.src[0][prot])) == int(prot.sum())   # InterestToMe never merges the region


def test_cheaper_than_region_dense_and_close_to_dense(model_and_image):
    from src.stage_b.timing import from_name
    model, x = model_and_image
    m = from_name("aitome_t0.5_l0.5_floor16", 14)
    out, st = _run(model, x, m, _map())
    it = InterestToMe(0.5, theta=0.01, floor=16, prop=False)
    _, st_i = _run(model, x, it, _map())
    fa = image_flops(st, ops_of(m), 0, 196)["flops"] / dense_flops(196)
    fi = image_flops(st_i, ops_of(it), 0, 196)["flops"] / dense_flops(196)
    assert fa < fi, (fa, fi)
    dense = model(x).final
    assert F.cosine_similarity(out[0, N_PREFIX:], dense[0, N_PREFIX:], dim=-1).mean() > 0.5


def test_surprise_protected_tokens_join_the_algm_class(model_and_image):
    """protect_tokens raises interest (no `sched`): the surprised tokens are merged only among protected tokens."""
    from src.stage_b.baselines import protect_tokens
    from src.stage_b.timing import from_name
    model, x = model_and_image
    m = from_name("aitome_t0.5_l0.5_floor16", 14)
    new = torch.tensor([14 * 12 + 2, 14 * 12 + 3, 14 * 13 + 2, 14 * 13 + 3])   # a 2 x 2 window outside the region

    class Wrap:
        per_image, proportional = True, False

        def __getattr__(self, a):
            return getattr(m, a)

        def after_attn(self, b, x_, k, p, st):
            if b == 0:
                x_, k = protect_tokens(m, b, x_, k, st, new)
            return m.after_attn(b, x_, k, p, st)
    _, st = _run(model, x, Wrap(), _map())
    prot = _map()[0] >= 0.01
    prot[new] = True
    src = st.src[0]
    for t in torch.unique(src).tolist():
        members = prot[src == t]
        assert bool(members.all()) or not bool(members.any())


def test_r35_without_surprise_equals_interest_tome(model_and_image):
    from src.stage_b.timing import from_name
    model, x = model_and_image
    m = from_name("sitome_t0.5_l0.5_floor16", 14)
    assert m.name == "sitome_t0.5_l0.5_floor16"
    a, _ = _run(model, x, m, _map())
    b, _ = _run(model, x, InterestToMe(0.5, theta=0.01, floor=16, prop=False), _map())
    assert torch.allclose(a, b, atol=1e-5)


def test_r35_region_whole_surprise_merged_among_itself(model_and_image):
    from src.stage_b.baselines import protect_tokens
    from src.stage_b.timing import from_name
    model, x = model_and_image
    m = from_name("sitome_t0.5_l0.5_floor16", 14)
    new = torch.tensor([14 * r + c for r in (10, 11, 12, 13) for c in (0, 1, 2, 3)])   # 4 x 4 block of 2 x 2 windows

    class Wrap:
        per_image, proportional = True, False

        def __getattr__(self, a):
            return getattr(m, a)

        def after_attn(self, b, x_, k, p, st):
            if b == 0:
                x_, k = protect_tokens(m, b, x_, k, st, new)
            return m.after_attn(b, x_, k, p, st)
    _, st = _run(model, x, Wrap(), _map())
    region = _map()[0] >= 0.01
    srp = torch.zeros(196, dtype=torch.bool)
    srp[new] = True
    src = st.src[0]
    for t in src[region].tolist():
        assert (src == t).sum() == 1, "a region patch was merged"
    for t in torch.unique(src).tolist():
        cls = {("r" if region[i] else "s" if srp[i] else "l") for i in torch.nonzero(src == t)[:, 0].tolist()}
        assert len(cls) == 1, cls
    assert len(torch.unique(src[srp])) < int(srp.sum()), "ALGM merged nothing in the surprise area"
