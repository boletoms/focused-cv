"""P2: focus maps from the previous video frame (owner direction; docs/Future Work - Interest-Conditioned Compute.md).

DAVIS 2017 val (30 videos), semi-supervised: the first frame's object masks are given. Each frame's patch features come
from the frozen backbone under a method; labels are propagated as in DINO's video protocol (first frame plus the last
7 frames as context, top 5 matches within 12 patches, temperature 0.1). For the focus budget (InterestToMe), the
focus map of frame t is the model's own predicted object mask at t - 1 (closed loop; optional: moved by each object's velocity, a margin widened when the map's area jumps,
plain ToMe as the fallback for an empty map; `keyframe` K: every K frames the full frame is processed (an I-frame, as in
MPEG), refreshing the map from a full-quality prediction; the first frame uses the given
mask); "@gt" runs use the true mask of frame t instead (a perfect prior, for reference). If a predicted map is empty, the
frame runs dense (fallback). Methods keep one instance per video, so temporal reuse ("reuse...:" settings) carries its
cache from frame to frame. Scores: J (region IoU) and F (boundary F-measure) per object, at the processing resolution,
over all frames but the first; FLOPs per frame against dense. N8 long-term memory (`long_every`, `long_keep`,
`global_long`): a frame stored every `long_every` frames joins the context, and with `global_long` the first frame and
the stored frames are matched over the whole frame (objects that leave and come back). Usage: python -m src.stage_b.p2_davis [config]"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.ablate.runinfo import git_dirty, git_hash
from src.stage_a.data import MEAN, STD
from src.stage_a.model import N_PREFIX, load_backbone

from .flops import dense_flops, image_flops
from .methods import ops_of
from .model import EffModel
from .timing import from_name


def load_frame(path, hw, dev):
    img = Image.open(path).convert("RGB").resize((hw[1], hw[0]), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]).to(dev)


class LazyMasks:
    """Ground-truth masks read from disk when indexed (long videos)."""
    def __init__(self, paths, hw, dev):
        self.paths, self.hw, self.dev, self._cache = paths, hw, dev, {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if i == 0:
            if 0 not in self._cache:
                self._cache[0] = load_mask(self.paths[0], self.hw, self.dev)
            return self._cache[0]
        return load_mask(self.paths[i], self.hw, self.dev)


def load_mask(path, hw, dev):
    m = Image.open(path).resize((hw[1], hw[0]), Image.NEAREST)
    return torch.from_numpy(np.asarray(m, dtype=np.int64).copy()).to(dev)


def shift_by_velocity(cur, prev):
    """Moves each object's mask in `cur` by its centroid displacement from `prev` to `cur` (constant velocity)."""
    out = torch.zeros_like(cur)
    H, W = cur.shape
    for o in torch.unique(cur):
        o = int(o)
        if o == 0:
            continue
        a, b = (cur == o).nonzero().float(), (prev == o).nonzero().float()
        if len(a) == 0 or len(b) == 0:
            out[cur == o] = o
            continue
        dy, dx = (a.mean(0) - b.mean(0)).round().long().tolist()
        m = torch.roll(cur == o, shifts=(dy, dx), dims=(0, 1))
        if dy > 0: m[:dy] = False
        if dy < 0: m[dy:] = False
        if dx > 0: m[:, :dx] = False
        if dx < 0: m[:, dx:] = False
        out[m | (cur == o)] = o                                          # keep the current mask too (a safe union)
    return out


def boundary(m):
    """Boundary pixels of a binary mask (H, W) bool."""
    er = -F.max_pool2d(-m.float()[None, None], 3, 1, 1)[0, 0]
    return m & (er < 0.5)


def j_and_f(pred, gt, tol):
    if not gt.any() and not pred.any():
        return 1.0, 1.0
    inter, union = (pred & gt).sum().item(), (pred | gt).sum().item()
    j = inter / union if union else 1.0
    bp, bg = boundary(pred), boundary(gt)
    k = 2 * tol + 1
    dil = lambda b: F.max_pool2d(b.float()[None, None], k, 1, tol)[0, 0] > 0
    p = (bp & dil(bg)).sum().item() / max(bp.sum().item(), 1)
    r = (bg & dil(bp)).sum().item() / max(bg.sum().item(), 1)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return j, f


def propagate(feat, ctx_feats, ctx_segs, gh, gw, cfg, local=None):
    """feat (N, D) normalized; ctx_feats list of (N, D); ctx_segs list of (N, K); local: per context frame, whether
    matches are limited to `radius` (default all; N8 long-term memory frames match globally). Returns (N, K)."""
    N = feat.shape[0]
    ys, xs = torch.meshgrid(torch.arange(gh, device=feat.device), torch.arange(gw, device=feat.device), indexing="ij")
    pos = torch.stack([ys.flatten(), xs.flatten()], 1)
    near = (pos[:, None, :] - pos[None, :, :]).abs().amax(-1) <= cfg["radius"]          # (N, N)
    cf, cs = torch.cat(ctx_feats), torch.cat(ctx_segs)                                   # (M N, D), (M N, K)
    local = local or [True] * len(ctx_feats)
    out = []
    for q0 in range(0, N, cfg.get("query_chunk", N)):          # N8 high resolution: query rows in chunks (memory)
        q1 = min(N, q0 + cfg.get("query_chunk", N))
        logits = (feat[q0:q1] @ cf.T) / cfg["temperature"]
        nr = near[q0:q1]
        allow = torch.cat([nr if l else torch.ones_like(nr) for l in local], 1)
        logits = logits.masked_fill(~allow, float("-inf"))
        top, idx = logits.topk(cfg["topk"], dim=1)
        out.append((top.softmax(1)[..., None] * cs[idx]).sum(1))
    return torch.cat(out)


def run_sequence(seq, names, gt_names, model, cfg, dev):
    root = Path(cfg["davis"])
    sub = "" if cfg.get("layout") == "lvos" else "480p"         # LVOS: JPEGImages/<seq>, Annotations/<seq>
    frames = sorted((root / "JPEGImages" / sub / seq).glob("*.jpg"))[: cfg.get("max_frames")]
    annos = sorted((root / "Annotations" / sub / seq).glob("*.png"))[: len(frames)]
    hw = tuple(cfg["size"])
    gh, gw = hw[0] // 16, hw[1] // 16
    N = gh * gw
    width = model.bb.config.hidden_size
    d_flops = dense_flops(N, width=width, blocks=len(model.bb.model.layer))
    if cfg.get("pingpong"):            # N3 long-video proxy: play forward and backward `pingpong` times (labels stay valid)
        T = len(frames)
        idx = list(range(T))
        for rep in range(1, cfg["pingpong"]):
            idx += list(range(T - 2, -1, -1)) if rep % 2 else list(range(1, T))
        frames, annos = [frames[i] for i in idx], [annos[i] for i in idx]
    gts = LazyMasks(annos, hw, dev)    # loaded per frame (long videos would not fit on the GPU)
    K = int(gts[0].max()) + 1                                                            # background + objects
    present = set(torch.unique(gts[0]).tolist()) - {0}                  # objects given in the first frame
    tol = max(1, round(cfg["boundary_th"] * (hw[0] ** 2 + hw[1] ** 2) ** 0.5))
    dtype = getattr(torch, cfg["dtype"])
    out = []
    for name in ([] if cfg.get("skip_dense") else ["dense"]) + names + [n + "@gt" for n in gt_names]:   # skip_dense: from another run
        use_gt = name.endswith("@gt")
        m_name = name[:-3] if use_gt else name
        prev_pred, prev_prev, prev_area, widened, keyframes = gts[0], None, None, 0, 0
        ctx_feats, ctx_segs, flops, js, fs, fallbacks = [], [], [], [[] for _ in range(K - 1)], [[] for _ in range(K - 1)], 0
        long_mem = []                    # N8: (feat, seg) every `long_every` frames, the last `long_keep` kept
        m_video = from_name(m_name, gh, len(model.bb.model.layer))                                  # one instance per video (temporal reuse keeps a cache)
        for t, fpath in enumerate(frames):
            x = load_frame(fpath, hw, dev)[None]
            m = m_video
            interest = None
            key = cfg.get("keyframe")                   # I-frames: every `keyframe` frames, process the full frame
            if key and t > 0 and t % key == 0 and m is not None:
                m = None
                keyframes += 1
            if m is not None and getattr(m, "per_image", False):
                src = gts[t] if (use_gt or t == 0) else prev_pred
                if cfg.get("motion") == "velocity" and not use_gt and t >= 2 and prev_prev is not None:
                    src = shift_by_velocity(prev_pred, prev_prev)
                wmode = cfg.get("weight_mode")
                if wmode:                      # graded (Q2): object 1 weight 1, other objects 1 / 0.5 / 0, background 0
                    other = {"both": 1.0, "graded": 0.5, "top": 0.0}[wmode]
                    touch = lambda b: F.max_pool2d(b.float()[None, None], 16)[0, 0].flatten()
                    interest = torch.maximum(touch(src == 1), other * touch(src > 1))[None]
                else:
                    interest = F.avg_pool2d((src > 0).float()[None, None], 16)[0, 0].flatten()[None]
                if cfg.get("reliability") and not use_gt and t >= 1 and prev_area is not None:
                    area = float(interest.gt(0).float().mean())
                    if prev_area > 0 and not (1 / cfg["reliability"] <= area / prev_area <= cfg["reliability"]):
                        interest = F.max_pool2d(interest.view(1, 1, gh, gw), 5, 1, 2).flatten()[None]   # widen by 2
                        widened += 1
                if interest.sum() == 0:
                    fallbacks += 1
                    if cfg.get("fallback"):                      # plain ToMe instead of dense
                        m, interest = from_name(cfg["fallback"], gh, len(model.bb.model.layer)), None
                    else:
                        m, interest = None, None
                prev_area = float(interest.gt(0).float().mean()) if interest is not None else prev_area
            sts = []
            with torch.no_grad():
                f = model(x, structure=m, dtype=dtype, state_out=sts, interest=interest).final[0, N_PREFIX:]
            feat = F.normalize(f.float(), dim=-1)
            flops.append((d_flops if m is None else image_flops(sts[0], ops_of(m), 0, N, width=width)["flops"]) / d_flops)
            if t == 0:
                seg = F.avg_pool2d(F.one_hot(gts[0], K).permute(2, 0, 1).float()[None], 16)[0].flatten(1).T   # (N, K)
                first = (feat, seg)
                continue
            short = list(zip(ctx_feats, ctx_segs))[-cfg["n_last"]:]
            ctx = [first] + long_mem + short
            loc = [not cfg.get("global_long")] * (1 + len(long_mem)) + [True] * len(short)
            seg = propagate(feat, [c[0] for c in ctx], [c[1] for c in ctx], gh, gw, cfg, loc)
            up = F.interpolate(seg.T.reshape(1, K, gh, gw), size=hw, mode="bilinear", align_corners=False)[0]
            pred = up.argmax(0)
            prev_prev, prev_pred = prev_pred, pred
            ctx_feats.append(feat)
            ctx_segs.append(seg)
            ctx_feats, ctx_segs = ctx_feats[-cfg["n_last"]:], ctx_segs[-cfg["n_last"]:]   # keep only what is used
            if cfg.get("long_every") and t % cfg["long_every"] == 0:
                long_mem = (long_mem + [(feat, seg)])[-cfg["long_keep"]:]
            if t < len(annos):
                gt_t = gts[t]
                for o in range(1, K):
                    if o not in present:                       # objects appearing later are not given: not scored
                        continue
                    j, fm = j_and_f(pred == o, gt_t == o, tol)
                    js[o - 1].append(j)
                    fs[o - 1].append(fm)
        out.append({"sequence": seq, "setting": name, "objects": K - 1, "frames": len(frames),
                    "flops_ratio": float(np.mean(flops)), "flops_ratio_after_first": float(np.mean(flops[1:])),
                    "J": [float(np.mean(v)) for v in js if v], "F": [float(np.mean(v)) for v in fs if v], "fallback_frames": fallbacks,
                    "widened_frames": widened, "keyframes": keyframes})
    return out


def main(cfg_path="configs/stage_b/p2_davis.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(cfg["out"])
    (out / "sequences").mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"run_id": cfg["run_id"], "config": cfg, "git": git_hash(),
                                               "dirty": git_dirty()}, indent=1) + "\n")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EffModel(load_backbone(dev, cfg["backbone"]) if cfg.get("backbone") else load_backbone(dev)).to(dev)
    if cfg.get("adapters"):                                          # Q10: trained adapters
        from .model import load_adapters
        load_adapters(model, cfg["adapters"])
    model.fast = bool(cfg.get("fast"))     # N8 high resolution: bf16 SDPA attention for every setting (never fp16)
    if cfg.get("layout") == "lvos":
        seqs = sorted(p.name for p in (Path(cfg["davis"]) / "JPEGImages").iterdir())[: cfg.get("max_sequences")]
    else:
        split = cfg.get("split", "val")                         # R2: "train" for tuning on held-out videos
        seqs = (Path(cfg["davis"]) / f"ImageSets/2017/{split}.txt").read_text().split()[: cfg.get("max_sequences")]
    if cfg.get("min_objects"):                                     # Q2: videos with several objects
        seqs = [q for q in seqs if int(np.asarray(Image.open(sorted((Path(cfg["davis"]) / "Annotations/480p" / q)
                                                                     .glob("*.png"))[0])).max()) >= cfg["min_objects"]]
    for seq in seqs:
        f = out / "sequences" / f"{seq}.json"
        if f.exists():
            continue
        try:
            os.close(os.open(out / "sequences" / f"{seq}.claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        rows = run_sequence(seq, cfg["methods"], cfg.get("gt_maps", []), model, cfg, dev)
        f.write_text(json.dumps(rows))
        print(f"{seq} done", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
