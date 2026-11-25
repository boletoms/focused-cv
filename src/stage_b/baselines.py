"""R1 baselines from the literature, ported from the authors' code into our framework (same frozen backbone, task
heads and FLOP counter). Sources are cloned, untouched, in ~/ml_work/third_party; each port keeps the original logic
and names the function it follows. None uses an focus map. Deviations needed by our setting are listed per class.

- PiToMe (NeurIPS 2024), hchautran/PiToMe @ 550b5de, algo/pitome/merge.py `pitome_vision`, `pitome`, `pitome_bsm`;
  patch/deit.py, patch/timm.py: metric = keys averaged over heads; energy = mean_j ELU(cos_ij - margin), margin =
  0.75 - 0.75 l / L; tokens sorted by energy; first ceil(L / 2) blocks: bipartite soft matching over the energy-sorted
  order (`pitome_bsm`), later blocks: the 2r highest-energy tokens split alternately into A and B, every A token merges
  into its most similar B token (`pitome`); r = floor(T - T ratio) of the current count; size-weighted means
  (`merge_wavg`); no proportional attention (the patched block does not pass sizes to attention).
- ALGM (CVPR 2024), tue-mps/algm-segmenter @ 674eaa2, algm/local_merge.py `conditional_pooling`, global_merge.py
  `turbo_matching`, patch/algm_segmenter_patch.py: selected layers 1 and 5 (ViT-L: 1 and 7), between attention and
  MLP, on the tokens after attention. Layer 1 (CLAP): 2 x 2 windows whose mean pairwise cosine similarity (self
  excluded) reaches `threshold` merge into one token. Later layer (GBM): bipartite matching (alternate tokens),
  threshold per image = mean + std / layer of the A tokens' best similarities, every A token at or above it merges.
  Size-weighted means; attention ignores sizes. The released thresholds (0.88 to 0.955) are for supervised ViTs; for
  DINOv3 we set it by the paper's rule (mean + std of pairwise cosine similarities after attention on held-out
  training images, `calibrate`) and sweep around it. Deviation: grids not divisible by the window leave the last
  row or column unmerged (the code asserts a square, divisible grid); the CLS token is not in the bipartite split.
- AiluRus (NeurIPS 2023), caddyless/ailurus @ 7765be2, mmseg/models/backbones/ailurus.py `AdaptiveResolution`,
  `AilurusViT`: after block `start_loc` (0-based; config: 2), tokens standardised per token, distances / sqrt(C)
  times spatial weights (linspace(alpha + (1 - alpha) / lambda, 1) over the lambda nearest positions including the
  token itself, 10 elsewhere; lambda = 50, alpha = 0.9), density exp(-mean d^2) over the k = 1 nearest, delta = min
  distance to a denser token (else the maximum), the M largest delta x density are centres, every token joins its
  nearest centre; plain means; unfolded by copying. The snapshot does not implement the paper's size re-weighting of
  attention, so neither do we. Config: 400 of 1,024 tokens (0.39).
- Expedite (NeurIPS 2022), Expedit-LargeScale-Vision-Transformer/Expedit-Segmenter @ 812e24b, mmseg/ops/cluster.py
  `TokenClusteringBlock`, mmseg/ops/reconstruction.py `TokenReconstructionBlock`, backbones/hourglass_vit.py: after
  block `location` (ADE20K ViT-L config: 9 of 24), centres = adaptive average pooling of the raw tokens to h x w
  (config: 784 of 1,600, 0.49), then 5 rounds: affinity = softmax over centres of -temperature x squared distance,
  restricted to centres within window // 2 cells of the token's initial cell (window 5, temperature 0.02); centres =
  affinity-weighted means. Reconstruction after the final norm: weights exp(-temperature x squared distance) between
  the tokens and the centres at clustering time, top k = 20 kept, L2-normalised, applied to the centres' outputs
  (k 20, temperature 0.02). Deviation: the Segmenter config never calls reconstruction (its decoder reads the
  cluster grid); the paper reconstructs after the last layer, which our per-patch heads need.
- G2TM (2026), vbercy/g2tm, branch segm @ 963093e (Apache-2.0), g2tm/g2tm/utils.py `cosine_similarity_masks`,
  `merge_from_labels`, `g2tm_merge`, connected_components.py `connected_components_labels` (tensorised FastSV, ported
  verbatim as `_fastsv_labels`), patch/graph_segmenter_patch.py `G2TMBlock`: one merge in the selected block (1-based;
  authors' settings layer 1 at 0.95, layer 2 at 0.88 for supervised ViTs), between attention and MLP, on the tokens
  after attention; edges between 4-neighbours with cosine similarity strictly above the threshold; each connected
  component becomes one token, the plain (unweighted) mean, kept at the component's lowest index; no proportional
  attention (the authors' default; they find it hurts segmentation). Deviations: the grid need not be square (the code
  asserts one; we pass height and width); CLS and all register tokens are kept out of the graph (the code protects one
  CLS token). The supervised-ViT thresholds do not transfer to DINOv3; see `calibrate_g2tm`.
Usage: from_name("pitome_k0.9", ...), "algm_t0.88", "ailurus_m0.39_at3", "expedite_m0.49_at5", "g2tm_t0.9_at2"; ALGM
threshold for DINOv3: python -m src.stage_b.baselines calibrate <res> [backbone]; G2TM:
python -m src.stage_b.baselines calibrate_g2tm <res> <layer> [backbone]."""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.stage_a.model import N_PREFIX

from .methods import Method

TAU_DIR = Path("/work/stage_b/algm_tau")


def merge_by_assignment(x, st, assign, m, ops=None, b=None, weighted=True):
    """Merge patch tokens by an assignment (1, n) -> 0..m-1: size-weighted (or plain) means of features, positions and
    interest; sizes add; the source map follows. Batch 1. Returns the new sequence."""
    P = N_PREFIX
    assert x.shape[0] == 1, "baselines run per image"
    a, D, n = assign[0], x.shape[-1], st.size.shape[1]
    w = st.size[0] if weighted else torch.ones_like(st.size[0])
    tot = torch.zeros(m, device=x.device).index_add_(0, a, w)
    feat = torch.zeros(m, D, device=x.device).index_add_(0, a, x[0, P:].float() * w[:, None]) / tot[:, None]
    st.coords = (torch.zeros(m, 2, device=x.device).index_add_(0, a, st.coords[0] * w[:, None]) / tot[:, None])[None]
    if st.interest is not None:
        st.interest = (torch.zeros(m, device=x.device).index_add_(0, a, st.interest[0] * w) / tot)[None]
    st.size = torch.zeros(m, device=x.device).index_add_(0, a, st.size[0])[None]
    live = st.src >= 0
    st.src = torch.where(live, a[st.src.clamp_min(0)], st.src)
    if ops is not None:
        ops.append((b, "merge", 2 * n * D))
    return torch.cat([x[:, :P], feat[None].to(x.dtype)], 1)


def _assign_pairs(n, src, dst, device):
    """Assignment where tokens `src` join tokens `dst` (disjoint sets), relabelled 0..m-1 in index order. No host
    sync (m = n - len(src) is known from shapes), so CUDA graphs can capture it."""
    a = torch.arange(n, device=device)
    a[src] = dst
    keep = torch.ones(n, dtype=torch.bool, device=device)
    keep.index_fill_(0, src, False)                          # no CPU scalar copy (graph capture)
    new = torch.cumsum(keep.long(), 0) - 1
    return new[a][None], n - len(src)


# SPDX-License-Identifier: CC-BY-NC-4.0 (derived from hchautran/PiToMe @ 550b5de; see NOTICE) -- begin
class PiToMe(Method):
    """keep: share of the current tokens kept per block (the repo's `ratio`)."""
    proportional = False

    def __init__(self, keep, blocks=12):
        super().__init__()
        self.keep, self.blocks = keep, blocks
        self.name = f"pitome_k{keep:g}"

    def after_attn(self, b, x, k, p, st):
        T = st.size.shape[1]
        r = math.floor(T - T * self.keep)
        if r <= 0:
            return x
        metric = F.normalize(k[0, :, N_PREFIX:].float().mean(0), p=2, dim=-1)          # keys, mean over heads
        sim = metric @ metric.T
        margin = 0.75 - 0.75 * (b / self.blocks)
        energy = F.elu(sim - margin, alpha=1.0).mean(-1)
        indices = torch.argsort(energy, descending=True)
        self.ops.append((b, "pitome_energy", 2 * T * T * metric.shape[-1]))
        if b < math.ceil(self.blocks * 0.5):                                       # pitome_bsm
            a_idx, b_idx = indices[::2], indices[1::2]
            r = min(r, len(a_idx))
            s = sim[a_idx][:, b_idx]
            node_max, node_idx = s.max(-1)
            src = node_max.argsort(descending=True)[:r]
            assign, m = _assign_pairs(T, a_idx[src], b_idx[node_idx[src]], x.device)
        else:                                                                       # pitome
            r = min(r, T // 2)
            merge_idx = indices[:2 * r]
            a_idx, b_idx = merge_idx[::2], merge_idx[1::2]
            dst = sim[a_idx][:, b_idx].max(-1).indices
            assign, m = _assign_pairs(T, a_idx, b_idx[dst], x.device)
        return merge_by_assignment(x, st, assign, m, self.ops, b)


# CC-BY-NC-4.0 -- end


class ALGM(Method):
    proportional = False

    def __init__(self, threshold, layers=(1, 5), window=2):
        super().__init__()
        self.threshold, self.layers, self.window = threshold, tuple(layers), window
        self.name = f"algm_t{threshold:g}" + ("" if tuple(layers) == (1, 5) else "_l" + "-".join(map(str, layers)))

    def after_attn(self, b, x, k, p, st):
        layer = b + 1                                                               # the repo's 1-based layers
        if layer not in self.layers:
            return x
        T = st.size.shape[1]
        if layer == self.layers[0]:                                                 # conditional_pooling
            gh, gw = st.grid
            ws = self.window
            hh, ww = gh // ws * ws, gw // ws * ws
            feat = x[0, N_PREFIX:].float().view(gh, gw, -1)[:hh, :ww]
            win = feat.reshape(hh // ws, ws, ww // ws, ws, -1).permute(0, 2, 1, 3, 4).reshape(-1, ws * ws, feat.shape[-1])
            sims = F.cosine_similarity(win[:, :, None, :], win[:, None, :, :], dim=-1)
            sims = sims * (1 - torch.eye(ws * ws, device=x.device))
            sim_map = sims.sum((-1, -2)) / ((ws * ws) * (ws * ws - 1))
            merge = sim_map >= self.threshold
            self.ops.append((b, "clap", 2 * T * ws * ws * feat.shape[-1]))
            ids = torch.arange(T, device=x.device).view(gh, gw)
            sub = ids[:hh, :ww].reshape(hh // ws, ws, ww // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
            dst = sub[:, -1:].expand(-1, ws * ws)                                   # the window's last token
            a = torch.arange(T, device=x.device)
            a[sub[merge].flatten()] = dst[merge].flatten()
            u, inv = torch.unique(a, return_inverse=True)
            return merge_by_assignment(x, st, inv[None], len(u), self.ops, b)
        metric = F.normalize(x[0, N_PREFIX:].float(), dim=-1)                       # turbo_matching
        a_m, b_m = metric[::2], metric[1::2]
        scores = a_m @ b_m.T
        node_max, node_idx = scores.max(-1)
        thr = node_max.mean() + node_max.std() / layer
        src = torch.nonzero(node_max >= thr)[:, 0]
        self.ops.append((b, "gbm", 2 * len(a_m) * len(b_m) * metric.shape[-1]))
        if len(src) == 0:
            return x
        A = torch.arange(0, T, 2, device=x.device)
        Bi = torch.arange(1, T, 2, device=x.device)
        assign, m = _assign_pairs(T, A[src], Bi[node_idx[src]], x.device)
        return merge_by_assignment(x, st, assign, m, self.ops, b)


class AiluRus(Method):
    proportional = False

    def __init__(self, ratio, start_loc=2, lam=50, k=1, alpha=0.9):
        super().__init__()
        self.ratio, self.start_loc, self.lam, self.k, self.alpha = ratio, start_loc, lam, k, alpha
        self.name = f"ailurus_m{ratio:g}_at{start_loc}"

    def before_block(self, b, x, st):
        if b != self.start_loc + 1:                                                 # after layer start_loc
            return x
        T, gh, gw = st.size.shape[1], *st.grid
        assert T == gh * gw, "AiluRus clusters the full grid"
        M = max(1, int(round(self.ratio * T)))
        y, xx = torch.meshgrid(torch.arange(gh, device=x.device), torch.arange(gw, device=x.device), indexing="ij")
        coords = torch.stack((y, xx), -1).reshape(-1, 2).float()
        order = torch.cdist(coords, coords).argsort(dim=1)
        nn_ = min(self.lam, T)
        sw = torch.full((T, T), 10.0, device=x.device)
        lw = torch.linspace(self.alpha + (1 - self.alpha) / nn_, 1.0, steps=nn_, device=x.device)
        sw.scatter_(1, order[:, :nn_], lw.expand(T, -1))
        tok = x[0, N_PREFIX:].float()
        z = (tok - tok.mean(-1, keepdim=True)) / tok.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        d = torch.cdist(z, z) / math.sqrt(tok.shape[-1]) * sw
        near = d.topk(k=self.k + 1, dim=-1, largest=False).values[:, 1:]
        density = (-(near ** 2).mean(-1)).exp()
        density = density + (T - torch.arange(T, device=x.device, dtype=density.dtype)) * (1e-6 / T)
        higher = density[None, :] > density[:, None]
        delta = torch.where(higher, d, d.max()).amin(-1)
        centres = (delta * density).topk(M).indices
        assign = d[centres].argmin(0)                                               # (T,) nearest centre
        assign[centres] = torch.arange(M, device=x.device)
        self.ops.append((b, "ailurus_dpc", 2 * T * T * tok.shape[-1]))
        return merge_by_assignment(x, st, assign[None], M, self.ops, b, weighted=False)


class Expedite(Method):
    proportional = False

    def __init__(self, ratio, location=4, n_iters=5, window=5, temperature=0.02, k=20, rec_temperature=None):
        super().__init__()
        self.ratio, self.location, self.n_iters, self.window = ratio, location, n_iters, window
        self.temp, self.k = temperature, k
        self.rec_temp = temperature if rec_temperature is None else rec_temperature
        self.name = f"expedite_m{ratio:g}_at{location}" + ("" if temperature == 0.02 else f"_t{temperature:g}")

    def before_block(self, b, x, st):
        if b != self.location + 1:                                                  # after layer `location`
            return x
        T, gh, gw = st.size.shape[1], *st.grid
        assert T == gh * gw, "Expedite clusters the full grid"
        f = math.sqrt(self.ratio)
        h, w = max(1, round(gh * f)), max(1, round(gw * f))
        pix = x[0, N_PREFIX:].float().T.reshape(1, -1, gh, gw)                     # (1, C, H, W)
        C = pix.shape[1]
        sp = F.adaptive_avg_pool2d(pix, (h, w))                                     # calc_init_centroid
        labels = torch.arange(h * w, device=x.device).reshape(1, 1, h, w).float()
        init = F.interpolate(labels, size=(gh, gw), mode="nearest").reshape(-1).long()
        sp = sp.reshape(1, C, -1)
        num = torch.arange(h * w, device=x.device)[:, None]
        r = self.window // 2
        mask = ((init[None] % w - num % w).abs() <= r) & ((init[None] // w - num // w).abs() <= r)   # (h*w, T)
        mask_dist = (~mask) * 1e16
        pf = pix.reshape(1, C, -1)
        ppf = pf.permute(0, 2, 1)
        pd = lambda f1, f2: ((f1 * f1).sum(1).unsqueeze(1) + (f2 * f2).sum(1).unsqueeze(2)
                             - 2 * torch.einsum("bcm,bcn->bmn", f2, f1))
        for _ in range(self.n_iters):
            dist = pd(pf, sp) + mask_dist
            aff = (-dist * self.temp).softmax(1)
            sp = torch.bmm(aff, ppf)
            sp = sp / aff.sum(2, keepdim=True).clamp_(min=1e-16)
            sp = sp.permute(0, 2, 1)
        hard = pd(pf, sp).argmin(1)[0]                                              # (T,)
        centres = sp.permute(0, 2, 1)[0]                                            # (m, C)
        self.ops.append((b, "expedite_cluster", (self.n_iters + 1) * 3 * T * h * w * C))
        self.before, self.after = pf[0].T, centres                                  # reconstruction state
        m = centres.shape[0]
        wt = aff[0]                                                                 # (m, T) last affinity
        den = wt.sum(1).clamp_min(1e-16)
        st.coords = ((wt @ st.coords[0]) / den[:, None])[None]
        if st.interest is not None:
            st.interest = ((wt @ st.interest[0]) / den)[None]
        st.size = den[None]
        st.src = hard[None].clone()
        return torch.cat([x[:, :N_PREFIX], centres[None].to(x.dtype)], 1)

    def reconstruct_out(self, out, st):
        """TokenReconstructionBlock on the normed outputs: exp(-t d^2), top k, L2-normalised, weighted sum."""
        if not hasattr(self, "before"):
            return out
        feat, sfeat = self.before, self.after
        ds = (feat * feat).sum(1)[:, None] + (sfeat * sfeat).sum(1)[None, :] - 2 * feat @ sfeat.T
        weight = torch.exp(-self.rec_temp * ds)
        if self.k >= 0:
            kk = min(self.k, weight.shape[1])
            mink = weight.topk(kk, dim=1).values.min(-1, keepdim=True).values
            weight = torch.where(weight >= mink, weight, torch.zeros_like(weight))
        att = F.normalize(weight, dim=1)
        patches = att @ out[0, N_PREFIX:].float()
        self.ops.append((99, "expedite_reconstruct", 2 * att.numel() * out.shape[-1]))
        n0 = patches.shape[0]
        st.src = torch.arange(n0, device=out.device)[None]
        del self.before, self.after
        return torch.cat([out[:, :N_PREFIX], patches[None].to(out.dtype)], 1)


@torch.no_grad()
def _neighbour_masks(feat, threshold, gh, gw):
    """G2TM `cosine_similarity_masks`, batch 1: (n,) masks of kept edges to the right and bottom neighbours."""
    grid = feat.view(gh, gw, -1)
    right = F.cosine_similarity(grid[:, 1:], grid[:, :-1], dim=-1) > threshold
    bottom = F.cosine_similarity(grid[1:], grid[:-1], dim=-1) > threshold
    right = torch.cat([right, torch.zeros(gh, 1, dtype=torch.bool, device=feat.device)], 1).reshape(1, -1)
    bottom = torch.cat([bottom, torch.zeros(1, gw, dtype=torch.bool, device=feat.device)], 0).reshape(1, -1)
    return right, bottom


def _fastsv_labels(right_mask, bottom_mask, base_grid_w, num_iters=None):
    """G2TM `connected_components_labels` (tensorised FastSV), verbatim: (b, n) smallest node index of each node's
    connected component."""
    b, n = right_mask.shape
    device = right_mask.device
    if num_iters is None:
        num_iters = math.ceil(math.log(max(n, 2)) / math.log(1.5)) + 2
    f = torch.arange(n, device=device, dtype=torch.int64).unsqueeze(0).expand(b, n)
    sentinel = torch.full((b, n), n, device=device, dtype=torch.int64)
    pad_col, pad_row = sentinel[:, :1], sentinel[:, :base_grid_w]
    for _ in range(num_iters):
        gf = f.gather(1, f)
        from_right = torch.where(right_mask, torch.cat((gf[:, 1:], pad_col), dim=1), sentinel)
        from_left = torch.cat((pad_col, torch.where(right_mask, gf, sentinel)[:, :-1]), dim=1)
        from_bottom = torch.where(bottom_mask, torch.cat((gf[:, base_grid_w:], pad_row), dim=1), sentinel)
        from_top = torch.cat((pad_row, torch.where(bottom_mask, gf, sentinel)[:, :-base_grid_w]), dim=1)
        ngf = torch.minimum(torch.minimum(from_right, from_left), torch.minimum(from_bottom, from_top))
        hooked = f.scatter_reduce(1, f, ngf, reduce="amin", include_self=True)
        f = torch.minimum(torch.minimum(hooked, f), torch.minimum(ngf, gf))
    return f


class G2TM(Method):
    proportional = False

    def __init__(self, threshold, at=2):
        super().__init__()
        self.threshold, self.at = threshold, at
        self.name = f"g2tm_t{threshold:g}_at{at}"

    def _prot(self, st):
        return None

    def after_attn(self, b, x, k, p, st):
        if b + 1 != self.at:                                                        # the repo's 1-based layers
            return x
        gh, gw = st.grid
        T = st.size.shape[1]
        assert T == gh * gw, "G2TM runs on the full patch grid"
        feat = x[0, N_PREFIX:].float()
        right, bottom = _neighbour_masks(feat, self.threshold, gh, gw)
        prot = self._prot(st)
        if prot is not None:                                                        # edges touching the region are cut
            right = right & ~prot[None] & ~torch.cat([prot[1:], prot[:1]])[None]
            bottom = bottom & ~prot[None] & ~torch.cat([prot[gw:], prot[:gw]])[None]
        labels = _fastsv_labels(right, bottom, gw)[0]
        self.ops.append((b, "g2tm_graph", 2 * 2 * T * feat.shape[-1]))
        u, inv = torch.unique(labels, return_inverse=True)                          # representatives in index order
        if len(u) == T:
            return x
        return merge_by_assignment(x, st, inv[None], len(u), self.ops, b, weighted=False)


def calibrate_g2tm(res, layer, backbone=None, images=200):
    """G2TM threshold scale for DINOv3, by ALGM's rule applied to G2TM's graph: mean + std of the cosine similarities
    between 4-neighbour patch tokens after attention in block `layer` (1-based), on ADE20K training images (held out
    from every evaluation)."""
    from src.stage_a.model import load_backbone

    from .ade import ADE
    from .model import EffModel
    res, layer = int(res), int(layer)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EffModel(load_backbone(dev, backbone) if backbone else load_backbone(dev)).to(dev)

    class Probe(Method):
        def after_attn(self, b, x, k, p, st):
            if b + 1 == layer:
                gh, gw = st.grid
                g = x[0, N_PREFIX:].float().view(gh, gw, -1)
                s = torch.cat([F.cosine_similarity(g[:, 1:], g[:, :-1], dim=-1).flatten(),
                               F.cosine_similarity(g[1:], g[:-1], dim=-1).flatten()]).double()
                self.acc += [s.sum().item(), (s ** 2).sum().item(), s.numel()]
            return x
    import numpy as np
    pr = Probe()
    pr.acc = np.zeros(3)
    ds = ADE("training", res, limit=int(images), seed=1)
    with torch.no_grad():
        for i in range(len(ds)):
            model(ds[i][0][None].to(dev), structure=pr, dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    mu = pr.acc[0] / pr.acc[2]
    sigma = math.sqrt(pr.acc[1] / pr.acc[2] - mu ** 2)
    tag = (backbone or "facebook/dinov3-vits16").split("/")[-1].split("-")[1]
    TAU_DIR.mkdir(parents=True, exist_ok=True)
    (TAU_DIR / f"g2tm_{tag}_{res}_at{layer}.json").write_text(json.dumps(
        {"mu": mu, "sigma": sigma, "threshold": mu + sigma, "layer": layer, "images": int(images),
         "source": "ADE20K training, seed 1", "device": dev}) + "\n")
    print(f"g2tm {tag} {res} layer {layer}: mu {mu:.4f} sigma {sigma:.4f} threshold {mu + sigma:.4f}", flush=True)


def calibrate(res, backbone=None, images=200):
    """ALGM's threshold by the paper's rule: mean + std of pairwise cosine similarities of the patch tokens after
    attention, over all blocks, on ADE20K training images (held out from every evaluation)."""
    from src.stage_a.model import load_backbone

    from .ade import ADE
    from .model import EffModel
    res = int(res)
    model = EffModel(load_backbone("cuda", backbone) if backbone else load_backbone("cuda")).cuda()

    class Probe(Method):
        def after_attn(self, b, x, k, p, st):
            z = F.normalize(x[0, N_PREFIX:].float(), dim=-1)
            s = z @ z.T
            off = s[~torch.eye(len(z), dtype=torch.bool, device=z.device)].double()
            self.acc += [off.sum().item(), (off ** 2).sum().item(), off.numel()]
            return x
    pr = Probe()
    import numpy as np
    pr.acc = np.zeros(3)
    ds = ADE("training", res, limit=int(images), seed=1)
    for i in range(len(ds)):
        model(ds[i][0][None].cuda(), structure=pr, dtype=torch.bfloat16)
    mu = pr.acc[0] / pr.acc[2]
    sigma = math.sqrt(pr.acc[1] / pr.acc[2] - mu ** 2)
    tag = (backbone or "facebook/dinov3-vits16").split("/")[-1].split("-")[1]
    TAU_DIR.mkdir(parents=True, exist_ok=True)
    (TAU_DIR / f"{tag}_{res}.json").write_text(json.dumps({"mu": mu, "sigma": sigma, "threshold": mu + sigma,
                                                           "images": int(images), "source": "ADE20K training, seed 1"}) + "\n")
    print(f"{tag} {res}: mu {mu:.4f} sigma {sigma:.4f} threshold {mu + sigma:.4f}", flush=True)


if __name__ == "__main__":
    {"calibrate": calibrate, "calibrate_g2tm": calibrate_g2tm}[sys.argv[1]](*sys.argv[2:])


class _Protected:
    """R6 map-aware baselines: the same protected set as our budget (patches with interest >= theta after a margin of
    `dilate` patches) is never merged; everything else follows the baseline unchanged. Separates "protect the region"
    (StructSAM's prompt-aware variant, prompted pruning) from the rest of our design (rate schedule, floor, closed loop)."""
    per_image = True

    def _protected(self, st):
        return None if st.interest is None else st.interest[0] >= self.theta

    def start(self, st):
        super().start(st)
        from .methods import dilate_interest
        dilate_interest(st, self.dilate)


class ToMeProtect(_Protected, Method):
    proportional = False

    def __init__(self, r, theta=0.01, dilate=0):
        super().__init__()
        self.r, self.theta, self.dilate = r, theta, dilate
        self.name = f"ptome_r{r}" + ("" if dilate == 0 else f"_dil{dilate}")

    def after_attn(self, b, x, k, p, st):
        n = st.size.shape[1]
        r = min(self.r, n // 2)
        if r <= 0:
            return x
        metric = F.normalize(k[0, :, N_PREFIX:].float().mean(0), dim=-1)
        A, Bi = torch.arange(0, n, 2, device=x.device), torch.arange(1, n, 2, device=x.device)
        scores = metric[A] @ metric[Bi].T
        prot = self._protected(st)
        if prot is not None:
            scores = scores.masked_fill(prot[A][:, None] | prot[Bi][None, :], -math.inf)
        node_max, node_idx = scores.max(-1)
        order = torch.sort(node_max, descending=True, stable=True).indices
        r = int(torch.isfinite(node_max[order[:r]]).sum())                # protected pairs never merge
        self.ops.append((b, "tome_match", 2 * len(A) * len(Bi) * metric.shape[-1]))
        if r == 0:
            return x
        src, unm = order[:r], order[r:]
        # ToMe's token order after merging: unmerged A tokens (in score order), then the B tokens
        nu = len(unm)
        assign = torch.empty(n, dtype=torch.long, device=x.device)
        assign[A[unm]] = torch.arange(nu, device=x.device)
        assign[Bi] = nu + torch.arange(len(Bi), device=x.device)
        assign[A[src]] = nu + node_idx[src]
        return merge_by_assignment(x, st, assign[None], nu + len(Bi), self.ops, b)


class ALGMProtect(_Protected, ALGM):
    def __init__(self, threshold, layers=(1, 5), theta=0.01, dilate=0):
        ALGM.__init__(self, threshold, layers)
        self.theta, self.dilate = theta, dilate
        self.name = f"palgm_t{threshold:g}" + ("" if dilate == 0 else f"_dil{dilate}")

    def after_attn(self, b, x, k, p, st):
        prot = self._protected(st)
        if prot is None or (b + 1) not in self.layers:
            return ALGM.after_attn(self, b, x, k, p, st)
        return self._merge_unprotected(b, x, st, prot)

    def _merge_unprotected(self, b, x, st, prot):
        T = st.size.shape[1]
        layer = b + 1
        if layer == self.layers[0]:
            gh, gw = st.grid
            ws = self.window
            hh, ww = gh // ws * ws, gw // ws * ws
            feat = x[0, N_PREFIX:].float().view(gh, gw, -1)[:hh, :ww]
            win = feat.reshape(hh // ws, ws, ww // ws, ws, -1).permute(0, 2, 1, 3, 4).reshape(-1, ws * ws, feat.shape[-1])
            sims = F.cosine_similarity(win[:, :, None, :], win[:, None, :, :], dim=-1) * (1 - torch.eye(ws * ws, device=x.device))
            sim_map = sims.sum((-1, -2)) / ((ws * ws) * (ws * ws - 1))
            ids = torch.arange(T, device=x.device).view(gh, gw)
            sub = ids[:hh, :ww].reshape(hh // ws, ws, ww // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
            merge = (sim_map >= self.threshold) & ~prot[sub].any(-1)
            a = torch.arange(T, device=x.device)
            a[sub[merge].flatten()] = sub[:, -1:].expand(-1, ws * ws)[merge].flatten()
            u, inv = torch.unique(a, return_inverse=True)
            return merge_by_assignment(x, st, inv[None], len(u), self.ops, b)
        metric = F.normalize(x[0, N_PREFIX:].float(), dim=-1)
        A, Bi = torch.arange(0, T, 2, device=x.device), torch.arange(1, T, 2, device=x.device)
        scores = metric[A] @ metric[Bi].T
        node_max, node_idx = scores.max(-1)
        thr = node_max.mean() + node_max.std() / layer                     # same threshold rule as ALGM
        scores = scores.masked_fill(prot[A][:, None] | prot[Bi][None, :], -math.inf)
        node_max, node_idx = scores.max(-1)
        src = torch.nonzero(node_max >= thr)[:, 0]
        if len(src) == 0:
            return x
        assign, m = _assign_pairs(T, A[src], Bi[node_idx[src]], x.device)
        return merge_by_assignment(x, st, assign, m, self.ops, b)


class G2TMProtect(_Protected, G2TM):
    def __init__(self, threshold, at=2, theta=0.01, dilate=0):
        G2TM.__init__(self, threshold, at)
        self.theta, self.dilate = theta, dilate
        self.name = f"pg2tm_t{threshold:g}_at{at}" + ("" if dilate == 0 else f"_dil{dilate}")

    def _prot(self, st):
        return self._protected(st)


# SPDX-License-Identifier: CC-BY-NC-4.0 (derived from hchautran/PiToMe @ 550b5de; see NOTICE) -- begin
class Novelty(Method):
    """R19 novelty map (E4): at block `at` (0-based; the first merge block of ours and of protected ALGM), PiToMe's
    energy (`pitome_vision`: keys averaged over heads, energy = mean_j ELU(cos_ij - margin), margin = 0.75 - 0.75 at /
    blocks) is computed for every patch token. High energy marks redundant tokens (PiToMe merges those first), so the
    fraction q of the unprotected tokens with the LOWEST energy (unlike their surroundings) join the protected set for
    this frame. Wraps a map-aware method: protected baselines read the raised interest at their merge; ours (whose
    protected-first order and merge schedule are fixed in `prepare`) gets the new tokens moved into its protected head
    and its schedule recomputed. Without a map (empty: the evaluator's fallback runs instead) nothing changes."""
    per_image = True
    proportional = False

    def __init__(self, inner, q, at=0, blocks=12):
        super().__init__()
        self.inner, self.q, self.at, self.blocks = inner, q, at, blocks
        self.theta = getattr(inner, "theta", 0.01)
        self.needs_cls = getattr(inner, "needs_cls", True)
        self.proportional = getattr(inner, "proportional", False)
        self.name = f"nov{q:g}:{inner.name}"

    def start(self, st):
        self.inner.start(st)
        self.ops = self.inner.ops                                                # one shared list (FLOPs)

    def prepare(self, x, st):
        return self.inner.prepare(x, st) if hasattr(self.inner, "prepare") else x

    def after_attn(self, b, x, k, p, st):
        if b == self.at and st.interest is not None and self.q > 0:
            x, k = self._protect_novel(b, x, k, st)
        return self.inner.after_attn(b, x, k, p, st)

    def _protect_novel(self, b, x, k, st):
        P = N_PREFIX
        T = st.size.shape[1]
        metric = F.normalize(k[0, :, P:].float().mean(0), p=2, dim=-1)
        energy = F.elu(metric @ metric.T - (0.75 - 0.75 * b / self.blocks), alpha=1.0).mean(-1)
        self.ops.append((b, "novelty_energy", 2 * T * T * metric.shape[-1]))
        free = st.interest[0] < self.theta
        n_new = int(self.q * int(free.sum()))
        if n_new == 0:
            return x, k
        cand = torch.nonzero(free)[:, 0]
        new = cand[torch.argsort(energy[cand], stable=True)[:n_new]]
        return protect_tokens(self.inner, b, x, k, st, new)


# CC-BY-NC-4.0 -- end


class ALGMInterest(ALGMProtect):
    """R32 (owner 2026-11-20: "do we not use ALGM or ToMe in the interesting or surprise areas? and only make it more
    aggressive in the low interest areas?"): the protected set (interest >= theta after the margin, size classes as
    ours, plus anything surprise protects) is merged by ALGM's rules among itself (layer-1 windows lying wholly inside
    it; turbo matching with both sides inside it, ALGM's threshold rule), instead of never; the out-of-focus tokens
    follow our budget (InterestToMe: a share `rate` per block among themselves, down to `floor`). The classes never mix.
    At an ALGM layer ALGM merges first (its windows need the full grid) and our matching then uses the keys averaged
    over the ALGM groups. Empty map: no merging here (the evaluator's fallback runs). Per image."""
    proportional = False

    def __init__(self, threshold, rate, floor=0, dilate=0, size_class=0, theta=0.01, layers=(1, 5)):
        ALGMProtect.__init__(self, threshold, layers, theta, dilate)
        from .methods import InterestToMe
        self.rate, self.floor, self.size_class = rate, floor, size_class
        self.tail = InterestToMe(rate, theta=theta, floor=floor, prop=False)
        self.name = f"aitome_t{threshold:g}_l{rate:g}_floor{floor}" + ("" if dilate == 0 else f"_dil{dilate}") + \
            ("" if not size_class else f"_cls{size_class}")

    def start(self, st):
        ALGMProtect.start(self, st)
        from .methods import promote
        if self.size_class and st.interest is not None:
            st.interest, _ = promote(st.interest, self.theta, self.size_class, st.grid)
            st.interest0 = st.interest
        self.tail.ops = self.ops

    def after_attn(self, b, x, k, p, st):
        if st.interest is None:
            return x
        prot = self._protected(st)
        if (b + 1) in self.layers and bool(prot.any()):
            P = N_PREFIX
            m0, w = F.normalize(k[0, :, P:].float().mean(0), dim=-1), st.size[0]
            x, assign = self._merge_inside(b, x, st, prot)
            if assign is not None:
                m = torch.zeros(int(assign.max()) + 1, m0.shape[-1], device=x.device).index_add_(0, assign, m0 * w[:, None])
                self.tail._metric = F.normalize(m, dim=-1)[None]
        try:
            return self.tail.after_attn(b, x, k, p, st)
        finally:
            self.tail._metric = None

    def _merge_inside(self, b, x, st, prot):
        """ALGM's merge restricted to the protected set; returns (x, assignment or None)."""
        T = st.size.shape[1]
        layer = b + 1
        if layer == self.layers[0]:
            gh, gw = st.grid
            ws = self.window
            hh, ww = gh // ws * ws, gw // ws * ws
            feat = x[0, N_PREFIX:].float().view(gh, gw, -1)[:hh, :ww]
            win = feat.reshape(hh // ws, ws, ww // ws, ws, -1).permute(0, 2, 1, 3, 4).reshape(-1, ws * ws, feat.shape[-1])
            sims = F.cosine_similarity(win[:, :, None, :], win[:, None, :, :], dim=-1) * (1 - torch.eye(ws * ws, device=x.device))
            sim_map = sims.sum((-1, -2)) / ((ws * ws) * (ws * ws - 1))
            self.ops.append((b, "clap", 2 * T * ws * ws * feat.shape[-1]))
            ids = torch.arange(T, device=x.device).view(gh, gw)
            sub = ids[:hh, :ww].reshape(hh // ws, ws, ww // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
            merge = (sim_map >= self.threshold) & prot[sub].all(-1)
            if not bool(merge.any()):
                return x, None
            a = torch.arange(T, device=x.device)
            a[sub[merge].flatten()] = sub[:, -1:].expand(-1, ws * ws)[merge].flatten()
            u, inv = torch.unique(a, return_inverse=True)
            return merge_by_assignment(x, st, inv[None], len(u), self.ops, b), inv
        metric = F.normalize(x[0, N_PREFIX:].float(), dim=-1)
        A, Bi = torch.arange(0, T, 2, device=x.device), torch.arange(1, T, 2, device=x.device)
        scores = metric[A] @ metric[Bi].T
        self.ops.append((b, "gbm", 2 * len(A) * len(Bi) * metric.shape[-1]))
        node_max, node_idx = scores.max(-1)
        thr = node_max.mean() + node_max.std() / layer                     # same threshold rule as ALGM
        scores = scores.masked_fill(~(prot[A][:, None] & prot[Bi][None, :]), -math.inf)
        node_max, node_idx = scores.max(-1)
        src = torch.nonzero(node_max >= thr)[:, 0]
        if len(src) == 0:
            return x, None
        assign, m = _assign_pairs(T, A[src], Bi[node_idx[src]], x.device)
        return merge_by_assignment(x, st, assign, m, self.ops, b), assign[0]


class ALGMSurprise(ALGMInterest):
    """R35 (after R32: merging inside the map costs the closed loop on long videos, −5.55 against ours with the same
    detector): the map's region (interest >= theta, size classes as ours) is never merged, as ours; only the patches
    surprise protects (marked `srp_value` by `protect_tokens`) are merged by ALGM's rules among themselves; the
    out-of-focus tokens follow our budget. Three classes, never mixed. Without surprise it equals InterestToMe."""
    srp_value = 0.5

    def __init__(self, threshold, rate, floor=0, dilate=0, size_class=0, theta=0.01, layers=(1, 5)):
        super().__init__(threshold, rate, floor, dilate, size_class, theta, layers)
        self.name = "s" + self.name[1:]

    def _protected(self, st):
        return (st.interest[0] - self.srp_value).abs() < 1e-3


class ALGMTiered(ALGMInterest):
    """R40 (the expectation focus map's three tiers, `src/stage_b/expect.py`; replaces ALGMSurprise there after the R40 smoke run: ALGM's
    rules restricted to a scattered middle tier merge almost nothing, since a layer-1 window must lie wholly inside it).
    High interest (>= 0.75: the binary map and promotions to high) is never merged. Every other token is merged by ALGM's
    own rules, as plain ALGM would outside the map (layer-1 windows with no high patch, ALGM's threshold; layer-5 matching
    between non-high tokens, ALGM's threshold rule). The low tokens (interest below theta: neither the map nor the middle
    tier, marked 0.5 by the wrapper) are then also merged by our budget among themselves. A merge of middle and low
    tokens stays middle (its size-weighted interest is at least theta), so our budget never reaches the middle tier.
    Without a middle tier it is ALGM outside the map plus our budget on top. Per image."""
    high_value, srp_value = 0.75, 0.5

    def __init__(self, threshold, rate, floor=0, dilate=0, size_class=0, theta=0.01, layers=(1, 5), high_algm=False):
        super().__init__(threshold, rate, floor, dilate, size_class, theta, layers)
        self.name = "t" + self.name[1:] + ("_ha1" if high_algm else "")
        if high_algm:                                    # ablation (owner 2026-11-21): the high tier also by ALGM's rules
            self.high_value = math.inf                   # (no token is exempt from ALGM; the low tokens still get our budget)

    def after_attn(self, b, x, k, p, st):
        if st.interest is None:
            return x
        if (b + 1) in self.layers:
            high = st.interest[0] >= self.high_value
            P = N_PREFIX
            m0, w = F.normalize(k[0, :, P:].float().mean(0), dim=-1), st.size[0]
            mid = (st.interest[0] >= self.theta) & ~high
            x, assign = self._merge_nonhigh(b, x, st, high)
            if assign is not None:
                m = torch.zeros(int(assign.max()) + 1, m0.shape[-1], device=x.device).index_add_(0, assign, m0 * w[:, None])
                self.tail._metric = F.normalize(m, dim=-1)[None]
                has_mid = torch.zeros(m.shape[0], device=x.device).index_add_(0, assign, mid.float()) > 0
                st.interest = torch.where(has_mid & (st.interest[0] < self.high_value), self.srp_value, st.interest[0])[None]
        try:
            return self.tail.after_attn(b, x, k, p, st)
        finally:
            self.tail._metric = None

    def _merge_nonhigh(self, b, x, st, high):
        """ALGM's merge of every token outside `high` (as ALGMProtect._merge_unprotected); returns (x, assignment or None)."""
        T = st.size.shape[1]
        layer = b + 1
        if layer == self.layers[0]:
            gh, gw = st.grid
            ws = self.window
            hh, ww = gh // ws * ws, gw // ws * ws
            feat = x[0, N_PREFIX:].float().view(gh, gw, -1)[:hh, :ww]
            win = feat.reshape(hh // ws, ws, ww // ws, ws, -1).permute(0, 2, 1, 3, 4).reshape(-1, ws * ws, feat.shape[-1])
            sims = F.cosine_similarity(win[:, :, None, :], win[:, None, :, :], dim=-1) * (1 - torch.eye(ws * ws, device=x.device))
            sim_map = sims.sum((-1, -2)) / ((ws * ws) * (ws * ws - 1))
            self.ops.append((b, "clap", 2 * T * ws * ws * feat.shape[-1]))
            ids = torch.arange(T, device=x.device).view(gh, gw)
            sub = ids[:hh, :ww].reshape(hh // ws, ws, ww // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
            merge = (sim_map >= self.threshold) & ~high[sub].any(-1)
            if not bool(merge.any()):
                return x, None
            a = torch.arange(T, device=x.device)
            a[sub[merge].flatten()] = sub[:, -1:].expand(-1, ws * ws)[merge].flatten()
            u, inv = torch.unique(a, return_inverse=True)
            return merge_by_assignment(x, st, inv[None], len(u), self.ops, b), inv
        metric = F.normalize(x[0, N_PREFIX:].float(), dim=-1)
        A, Bi = torch.arange(0, T, 2, device=x.device), torch.arange(1, T, 2, device=x.device)
        scores = metric[A] @ metric[Bi].T
        self.ops.append((b, "gbm", 2 * len(A) * len(Bi) * metric.shape[-1]))
        node_max, node_idx = scores.max(-1)
        thr = node_max.mean() + node_max.std() / layer                     # same threshold rule as ALGM
        scores = scores.masked_fill(high[A][:, None] | high[Bi][None, :], -math.inf)
        node_max, node_idx = scores.max(-1)
        src = torch.nonzero(node_max >= thr)[:, 0]
        if len(src) == 0:
            return x, None
        assign, m = _assign_pairs(T, A[src], Bi[node_idx[src]], x.device)
        return merge_by_assignment(x, st, assign, m, self.ops, b), assign[0]


class ToMeTiered(ALGMTiered):
    """R40 second baseline (owner 2026-11-21: ToMe): as ALGMTiered with ToMe's rules on every token outside the high tier
    (bipartite soft matching of the keys averaged over heads, r tokens per block from block 0, size-weighted means; as
    `ToMeProtect`, the protected form of `tome_r{r}_from1_noprop`); the low tokens are then also merged by our budget. A
    merge of middle and low tokens stays middle. Without a middle tier it is protected ToMe plus our budget on top.
    `as_rate` (`_tr1`, plan section I, review flag R-34): ToMe's rule as a rate, round(r_b / N_b x the non-high tokens)
    per block, where plain ToMe merges r_b of its N_b tokens at that block, so the middle tier keeps plain ToMe's share
    once our budget has thinned the low tokens (with the fixed count it collapsed)."""

    def __init__(self, r, rate, floor=0, dilate=0, size_class=0, theta=0.01, as_rate=False):
        super().__init__(1.0, rate, floor, dilate, size_class, theta, layers=())
        self.r, self.as_rate = r, as_rate
        self.name = f"ttome_r{r}_l{rate:g}_floor{floor}" + ("" if dilate == 0 else f"_dil{dilate}") + \
            ("" if not size_class else f"_cls{size_class}") + ("_tr1" if as_rate else "")

    def start(self, st):
        """`as_rate` (R40 plan section I, review flag R-34): plain ToMe's token count, followed block by block from the
        frame's patch count, sets the share of the non-high tokens merged in each block."""
        super().start(st)
        self._plain_n = st.size.shape[1]

    def _rate_r(self, high):
        """Tokens to merge outside `high` in this block: plain ToMe's share at this block times the non-high count."""
        n_plain = self._plain_n
        r_plain = min(self.r, n_plain // 2)
        self._plain_n = n_plain - r_plain
        return int(round(r_plain / n_plain * int((~high).sum()))) if n_plain > 0 else 0

    def after_attn(self, b, x, k, p, st):
        if st.interest is None:
            return x
        high = st.interest[0] >= self.high_value
        mid = (st.interest[0] >= self.theta) & ~high
        m0, w = F.normalize(k[0, :, N_PREFIX:].float().mean(0), dim=-1), st.size[0]
        x, assign = self._tome_nonhigh(b, x, k, st, high, self._rate_r(high) if self.as_rate else None)
        if assign is not None:
            m = torch.zeros(int(assign.max()) + 1, m0.shape[-1], device=x.device).index_add_(0, assign, m0 * w[:, None])
            self.tail._metric = F.normalize(m, dim=-1)[None]
            has_mid = torch.zeros(m.shape[0], device=x.device).index_add_(0, assign, mid.float()) > 0
            st.interest = torch.where(has_mid & (st.interest[0] < self.high_value), self.srp_value, st.interest[0])[None]
        try:
            return self.tail.after_attn(b, x, k, p, st)
        finally:
            self.tail._metric = None

    def _tome_nonhigh(self, b, x, k, st, high, r=None):
        """ToMe's merge of r tokens (default `self.r`) among the tokens outside `high` (as ToMeProtect.after_attn);
        (x, assignment or None)."""
        n = st.size.shape[1]
        r = min(self.r if r is None else r, n // 2)
        if r <= 0:
            return x, None
        metric = F.normalize(k[0, :, N_PREFIX:].float().mean(0), dim=-1)
        A, Bi = torch.arange(0, n, 2, device=x.device), torch.arange(1, n, 2, device=x.device)
        scores = (metric[A] @ metric[Bi].T).masked_fill(high[A][:, None] | high[Bi][None, :], -math.inf)
        node_max, node_idx = scores.max(-1)
        order = torch.sort(node_max, descending=True, stable=True).indices
        r = int(torch.isfinite(node_max[order[:r]]).sum())
        self.ops.append((b, "tome_match", 2 * len(A) * len(Bi) * metric.shape[-1]))
        if r == 0:
            return x, None
        src, unm = order[:r], order[r:]
        nu = len(unm)
        assign = torch.empty(n, dtype=torch.long, device=x.device)
        assign[A[unm]] = torch.arange(nu, device=x.device)
        assign[Bi] = nu + torch.arange(len(Bi), device=x.device)
        assign[A[src]] = nu + node_idx[src]
        return merge_by_assignment(x, st, assign[None], nu + len(Bi), self.ops, b), assign


def protect_tokens(inner, b, x, k, st, new):
    """Protect the unprotected patch tokens `new` (token indices, before any merge of this frame) from block b on.
    Protected baselines read the raised interest at their merge; ours (protected-first order and merge schedule fixed
    in `prepare`) gets the tokens moved into its protected head and its schedule recomputed from block b."""
    P, T = N_PREFIX, st.size.shape[1]
    if len(new) == 0:
        return x, k
    if not hasattr(inner, "sched"):                                              # protected baselines read the map
        st.interest = st.interest.clone()
        st.interest[0, new] = getattr(inner, "srp_value", 1.0)                  # R35 marks surprise tokens
        return x, k
    h = inner.n_high                                                             # ours: tokens are protected-first
    assert bool((new >= h).all()), "tokens to protect come from the unprotected tail"
    is_new = torch.zeros(T, dtype=torch.bool, device=x.device)
    is_new[new] = True
    tail = torch.arange(h, T, device=x.device)
    order = torch.cat([torch.arange(h, device=x.device), tail[is_new[h:]], tail[~is_new[h:]]])
    inv = torch.empty_like(order)
    inv[order] = torch.arange(T, device=x.device)
    st.src = inv[st.src.clamp_min(0)].where(st.src >= 0, st.src)
    st.interest = st.interest.clone()
    st.interest[0, new] = 1.0
    st.size, st.coords, st.interest = st.size[:, order], st.coords[:, order], st.interest[:, order]
    x = torch.cat([x[:, :P], x[:, P:][:, order]], 1)
    k = torch.cat([k[:, :, :P], k[:, :, P:][:, :, order]], 2)
    inner.n_high = h + int(is_new.sum())
    n_low, inner.sched = T - inner.n_high, {}
    for bb in range(max(b, inner.first), min(inner.last, 63) + 1):              # as FastInterestToMe.prepare
        r = min(int(inner.rate * n_low), max(0, n_low - inner.floor), n_low // 2)
        if r <= 0:
            break
        inner.sched[bb] = r
        n_low -= r
    return x, k

