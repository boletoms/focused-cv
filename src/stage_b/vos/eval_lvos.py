"""R12 long videos inside Cutie: LVOS V1 val (50 videos, 31,158 annotated frames at 1280 x 720, every fifth frame of
the source video), with the R10 Cutie + frozen DINOv3 ViT-S model and Cutie's LVOS protocol (eval_config `lvos-val`:
shortest side 480, long-term memory, memory every 5 frames; each object's mask given at its first appearance, as
Cutie's `use_all_masks` with `Annotations_first_only`). Output probabilities are upsampled to 1280 x 720 before the
argmax (as Cutie's saver) and each object is scored on every frame after its first appearance (our J&F, boundary
tolerance 0.8% of the diagonal). The focus map of frame t is the previous prediction (at the internal size) plus
any object given at t. Empty map: `setting` alone falls back to dense; `setting~fallback` uses that method instead.
R13 suffixes (re-acquisition after the target leaves the map): `@key{k}` a dense I-frame every k frames; `@hold{h}`
the map is the union of the last h predictions (a lost object's region stays protected for h frames); `@motion` each
object's last mask moved by its centroid displacement over the last two predictions (P2 `shift_by_velocity`, kept
in union with the unshifted mask).
R15 surprise suffixes: `@memkey` a dense frame exactly when Cutie writes working memory (a mask is given, or mem_every
frames since the last memory frame); `@oraclemap` / `@oraclekey` (ceiling only, uses ground truth): when a tracked
object has at least one patch of area (256 px at the internal size) and half of it outside the protected area (the
map dilated by one patch), its true mask is added to the map / the frame runs dense, on the same frame; `@surpAmap` /
`@surpAkey`: pixels outside the protected area whose summed object probability exceeds 0.2 (at least one patch) are
added to the next frame's map / make the next frame dense; `@surpBkey`: an object's predicted area below half its
mean over its last 5 frames, or its mean probability on its predicted pixels below 0.7, makes the next frame dense
(it keeps firing while the recent mean stays above one patch, so a lost object buys up to 5 dense frames).
Rows also count trigger frames, memory frames and memory frames that ran dense.
R27 correction of signal A (`@surpA2map` / `@surpA2key`; signal A kept unchanged for the R15, R24 and R26 rows): signal A
tests object probability against the map used for frame t, so it fires on ordinary motion and soft mask edges that the
next frame's map already covers, misses faint evidence inside the old map that the next map drops, and counts scattered
pixels as "one patch". A2 tests against the next frame's own map (prediction, plus motion and hold when set) dilated by
one patch, and counts evidence per patch of the encoder's grid: a patch fires when at least half its pixels have summed
object probability above 0.2 (`patch_evidence`). `empty_map_frames` counts frames whose map is empty whatever ran on them
(`fallback_frames` misses empty-map frames that a trigger made dense).
R23 map sources (diagnostic, separating the mechanism from the closed loop): `@gtmap` the map is the ground truth of the
previous frame (a perfect tracker; the model's own previous prediction where that frame has no annotation);
`@densemap` the map is the dense model's previous prediction (stored, bit-packed, by the dense pass that runs first on
each video). Both are united with any object given at t; fallback, FLOPs and scoring unchanged.
R33 map quality (`@gtmap@drop{p}`, `@shift{d}`, `@grow{d}`, combinable): the ground-truth map degraded on purpose
(`degrade`; seeded per video and frame, so every setting sees the same draws).
R28 surprise (`@srp_<detector>`, `src/stage_b/surprise.py`, name as `SurpriseDetector.name`): unexpected change in the
low-focus area, detected from the block-0 features, is reprocessed at full resolution on the same frame; the method
and the empty-map fallback are wrapped (plain ToMe / ALGM fallbacks become their protected forms, identical while
nothing fires), and dense frames run an observer so the detector sees every frame. Rows count frames with surprise,
the mean share of patches it reprocessed, and global frames.
R40 redesign (`@exp-...`, `src/stage_b/expect.py`, name as `ExpectConfig.name`; with `sitome_...~algm_t...`): the
map expects each tracked object at its centroid-shifted position; on the current frame (block 0) objects are
relocalised by block matching, a speed-scaled ring and lost objects' search discs take the baseline's rules, and
surprise (deviation from the previous frame's features moved by the camera) promotes out-of-focus patches (high when
near an object, else the baseline's rules); when every object that appeared is lost the frame runs the fallback
(the baseline), which observes passively. Rows add the mean shares of patches promoted to high and to the middle tier,
global frames and fallbacks caused by losses. `VOS_SKIP_DENSE=1` skips the dense pass when its rows exist elsewhere.
Determinism (review flag R-29; default from 2026-11-20, `VOS_DETERMINISTIC=0` to disable): with the default kernels
the reduced methods are not reproducible run to run (CUDA reductions add in varying order) and long videos amplify
the difference; with deterministic kernels runs are bitwise identical (checked on LVOS train video PEyHFztC: dense,
ALGM 0.98 and ours, three runs and a logged run).
R30 (`VOS_OFFICIAL=small|base`): the official Cutie network with its own ResNet pixel encoder and the given official
weights, same protocol and scoring; only `dense` (FLOPs reported as 1).
FLOPs of the ViT per frame against dense. The ViT runs on the bf16 SDPA path (FLOPs counted as in the eager path; N8).
Usage (ml-base:vos): python -m src.stage_b.vos.eval_lvos <weights.pth> <out_dir> <setting,...> [max videos] [split]"""
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor

from src.stage_b.vos.eval import build

LVOS = Path("/work/data/lvos")
LVOS2 = Path("/work/data/lvos_v2")                # LVOS V2 (R38, diagnostics): 420 train, 140 valid videos
SIZE = 480
PATCH = 16
PATCH_PX = PATCH * PATCH


def dilate(mask, px=16):
    """The protected area: the map dilated by one patch (as the methods' margin 1)."""
    return F.max_pool2d(mask.float()[None, None], 2 * px + 1, 1, px)[0, 0] > 0


DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _shift(m, dy, dx):
    """Move a (h, w) map by (dy, dx) pixels, zero fill."""
    out = torch.zeros_like(m)
    H, W = m.shape
    sy, ty = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    sx, tx = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[ty, tx] = m[sy, sx]
    return out


def degrade(lab, mods, rng):
    """R33 map quality: a ground-truth label map (h, w at the evaluation size, 0 background) degraded on purpose, in
    this order: `drop{p}` each object left out with probability p; `shift{d}` the map moved d patches along each axis
    of one of 8 directions drawn per frame (zero fill); `grow{d}` dilated by d patches. Returns the boolean map."""
    for x in mods:
        if x.startswith("drop"):
            gone = [o for o in torch.unique(lab).tolist() if o and rng.random() < float(x[4:])]
            if gone:
                lab = torch.where(torch.isin(lab, torch.tensor(gone, device=lab.device)), 0, lab)
    m = lab > 0
    for x in mods:
        if x.startswith("shift"):
            dy, dx = DIRS[int(rng.integers(8))]
            d = int(x[5:]) * PATCH
            m = _shift(m, dy * d, dx * d)
        elif x.startswith("grow"):
            m = dilate(m, int(x[4:]) * PATCH)
    return m


def patch_evidence(prob, protected, thr=0.2):
    """Signal A2: the patches of the encoder's grid (padded as Cutie's pad_divide_by: symmetric, to multiples of 16)
    outside `protected` dilated by one patch where at least half the pixels have summed object probability above
    `thr`, as a pixel mask of the frame (empty if none)."""
    h, w = protected.shape
    ev = (((1 - prob[0]) > thr) & ~dilate(protected)).float()
    ph, pw = (-h) % PATCH, (-w) % PATCH
    lh, lw = ph // 2, pw // 2
    g = F.avg_pool2d(F.pad(ev[None, None], (lw, pw - lw, lh, ph - lh)), PATCH)[0, 0] >= 0.5
    return g.repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)[lh:lh + h, lw:lw + w]


def build_official(weights, size, dataset):
    """The official Cutie (its ResNet pixel encoder; no DINOv3), as verify_official (R26 gate 1)."""
    import cutie.model.cutie as cm
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from cutie.inference.utils.args_utils import get_dataset_cfg
    from src.stage_b.vos.eval import CUTIE
    with initialize_config_dir(version_base="1.3.2", config_dir=f"{CUTIE}/cutie/config", job_name="eval_config"):
        cfg = compose(config_name="eval_config", overrides=[f"model={size}", f"dataset={dataset}"])
    with open_dict(cfg):
        cfg["weights"] = weights
    get_dataset_cfg(cfg)
    net = cm.CUTIE(cfg).cuda().eval()
    net.load_weights(torch.load(weights, map_location="cuda", weights_only=True))
    return net, cfg


def deterministic():
    """R-29 (owner 2026-11-20, adopted for every evaluation): CUDA reductions in the merge paths add in varying order and
    long videos amplify the difference, so evaluations use deterministic kernels (bitwise reproducible, about 10% slower,
    no method change). `VOS_DETERMINISTIC=0` restores the default kernels (runs before 2026-11-20 used them)."""
    if os.environ.get("VOS_DETERMINISTIC", "1") != "0":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


@torch.inference_mode()
def main(weights, out_dir, settings, max_seq=None, split="valid"):
    from cutie.inference.inference_core import InferenceCore
    from src.stage_b.flops import dense_flops, image_flops
    from src.stage_b.methods import ops_of
    from src.stage_b.p2_davis import j_and_f, shift_by_velocity
    from src.stage_b.timing import from_name
    from src.stage_b.vos import masks
    deterministic()
    official = os.environ.get("VOS_OFFICIAL")
    if official:
        assert settings == "dense", "official Cutie: dense only"
        net, cfg = build_official(weights, official, "lvos-val")
        enc, width, blocks, settings = None, None, 12, ""
    else:
        net, cfg = build(weights, "lvos-val")
        enc = net.pixel_encoder
        enc.vit.fast = True
        width, blocks = enc.vit.bb.config.hidden_size, len(enc.vit.bb.model.layer)
    out = Path(out_dir)
    (out / "sequences").mkdir(parents=True, exist_ok=True)
    root = LVOS2 / split[3:] if split.startswith("v2_") else LVOS / split      # v2_train, v2_valid: LVOS V2
    seqs = sorted(p.name for p in (root / "JPEGImages").iterdir())[: int(max_seq) if max_seq and max_seq != "-" else None]
    if os.environ.get("VOS_SEQS"):                                             # a fixed list of videos (diagnostics)
        keep = set(Path(os.environ["VOS_SEQS"]).read_text().split())
        seqs = [x for x in seqs if x in keep]
    diag = os.environ.get("VOS_DIAG") == "1"
    for seq in seqs:
        f = out / "sequences" / f"{seq}.json"
        if f.exists():
            continue
        try:
            os.close(os.open(out / "sequences" / f"{seq}.claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        frames = sorted((root / "JPEGImages" / seq).glob("*.jpg"))
        annos = {p.stem: p for p in (root / "Annotations" / seq).glob("*.png")}
        # ground truth read per frame when needed (holding a long video's masks costs 1 to 2 GB of host memory)
        gt_at = lambda t: np.array(Image.open(annos[frames[t].stem])) if frames[t].stem in annos else None
        first, H, W = {}, None, None                                 # object -> frame of first appearance
        for t in range(len(frames)):
            g = gt_at(t)
            if g is not None:
                H, W = g.shape
                for o in np.unique(g).tolist():
                    if o and o not in first:
                        first[o] = t
        tol = max(1, round(0.008 * (H ** 2 + W ** 2) ** 0.5))
        pal = masks.palette(next(iter(annos.values()))) if masks.ON else None
        rows = []
        need_dense_maps = any("@densemap" in x for x in settings.split(","))
        dense_pred = {}                                              # t -> bit-packed dense prediction (R23)
        skip_dense = os.environ.get("VOS_SKIP_DENSE") == "1" and not need_dense_maps and not diag   # dense rows run
        for spec_i, spec in enumerate(([] if skip_dense else ["dense"]) + [x for x in settings.split(",") if x]):
            base, *mods = spec.split("@")
            name, _, fb_name = base.partition("~")
            smalls = []                                                  # VOS_SAVE_SMALL (figures; reads only)
            key = next((int(x[3:]) for x in mods if x.startswith("key")), None)
            hold = next((int(x[4:]) for x in mods if x.startswith("hold")), 1)
            history = []
            motion = "motion" in mods
            memkey = "memkey" in mods
            oracle = next((x[6:] for x in mods if x.startswith("oracle")), None)      # "map" | "key"
            sig_a = next((x[5:] for x in mods if x in ("surpAmap", "surpAkey")), None)       # "map" | "key"
            sig_a2 = next((x[6:] for x in mods if x in ("surpA2map", "surpA2key")), None)    # R27 correction
            sig_b = "surpBkey" in mods
            mapsrc = "gt" if "gtmap" in mods else "dense" if "densemap" in mods else None
            deg = [x for x in mods if x.startswith(("drop", "shift", "grow"))]                 # R33
            assert not deg or mapsrc == "gt", "map degradations apply to @gtmap only"
            srp = next((x for x in mods if x.startswith("srp_")), None)
            expm = next((x for x in mods if x.startswith("exp-")), None)          # R40 redesign (`src/stage_b/expect.py`)
            membase = "membase" in mods                    # R40 M: frames Cutie writes to memory run the baseline (fallback)
            dense_next, extra_next, areas = False, None, {}
            triggers, mem_frames, mem_dense, empty = 0, 0, 0, 0
            idx_prev = idx_prev2 = None                                 # last two predicted index masks
            proc = InferenceCore(net, cfg=cfg)
            proc.max_internal_size = -1                              # resizing done here, as Cutie's video reader
            m_video = None if name == "dense" else from_name(name, SIZE // 16, blocks)
            fb = from_name(fb_name, SIZE // 16, blocks) if fb_name else None
            det, observer, srp_frames, srp_share, srp_global = None, None, 0, 0.0, 0
            if srp:
                from src.stage_b.surprise import SurpriseMap, detector_from_name, protected_form
                det = detector_from_name(srp)
                m_video = SurpriseMap(m_video, det) if m_video is not None else None
                fb = SurpriseMap(from_name(protected_form(fb_name), SIZE // 16, blocks), det) if fb_name else None
                observer = SurpriseMap(None, det)
            estate, ostate, exp_high, exp_mid, exp_global, exp_lostfb = None, {}, 0.0, 0.0, 0, 0
            if expm:
                from src.stage_b.expect import ExpectMap, ExpectState, config_from_name, objects_from_predictions
                assert m_video is not None and fb is not None and not srp, "@exp needs a reduced method and its fallback"
                ecfg, estate = config_from_name(expm), ExpectState()
                m_video, fb = ExpectMap(m_video, estate, ecfg), ExpectMap(fb, estate, ecfg, passive=True)
                motion = motion or ecfg.motion                         # the centroid shift enters the map itself
                if ecfg.mbase:                                         # R40 M: motion at the search radius -> baseline
                    m_video.alt = fb.inner
            assert not membase or (estate is not None and fb is not None), "@membase needs @exp and a fallback"
            membase_frames, mbase_frames = 0, 0
            graph2 = os.environ.get("VOS_GRAPH") == "2" and estate is not None   # R40 plan L: the frame runner does
            if graph2:                                                 # the map motion, object states and decision
                assert enc is not None and not (key or hold > 1 or memkey or oracle or mapsrc or sig_a or sig_a2 or sig_b
                                                or diag or membase or ecfg.mbase), "VOS_GRAPH=2 covers the redesign's own settings only"
                enc.frame_setting, g2_reset = spec, True
            prev, fl, fallbacks = None, [], 0
            dlog = None
            if diag:
                from src.stage_b.vos.diag import DiagLog
                dlog = DiagLog(out, seq, spec, spec_i, sorted(first), with_gt=spec == "dense")
            js, fs = {o: [] for o in first}, {o: [] for o in first}
            started = False

            def base_map():                                          # the map before triggers: prediction, motion, hold
                b = prev
                if motion and idx_prev is not None and idx_prev2 is not None:
                    b = b | (shift_by_velocity(idx_prev, idx_prev2) > 0)
                for hm in history[-(hold - 1):] if hold > 1 else []:
                    b = b | hm
                return b
            for t, fp in enumerate(frames):
                new = [o for o, t0 in first.items() if t0 == t]
                if not started and not new:
                    continue
                started = True
                img = to_tensor(Image.open(fp).convert("RGB")).cuda()
                img = resize(img, SIZE, InterpolationMode.BILINEAR, antialias=True)
                h, w = img.shape[-2:]
                mask = None
                if new:
                    g = torch.from_numpy(gt_at(t).astype(np.int64)).cuda()
                    g = torch.where(torch.isin(g, torch.tensor(new, device=g.device)), g, 0)
                    mask = resize(g[None], [h, w], InterpolationMode.NEAREST)[0]
                    prev = (mask > 0) if prev is None else (prev | (mask > 0))
                end = t == len(frames) - 1
                will_mem = ((proc.curr_ti + 1 - proc.last_mem_ti) >= proc.mem_every or mask is not None) and not end
                m = m_video
                if m is not None and key and t % key == 0:
                    m = None
                if m is not None and (memkey and will_mem or dense_next):
                    m = None
                interest_mask = prev if graph2 else base_map()         # graph2: the motion is added in the runner
                if extra_next is not None:
                    interest_mask = interest_mask | extra_next
                if mapsrc is not None:                                 # R23: replace the closed-loop map
                    src, g_prev = None, gt_at(t - 1) if mapsrc == "gt" and t > 0 else None
                    if g_prev is not None:
                        lab = resize(torch.from_numpy(g_prev.astype(np.int64)).cuda()[None], [h, w],
                                     InterpolationMode.NEAREST)[0]
                        src = degrade(lab, deg, np.random.default_rng([zlib.crc32(seq.encode()), t])) if deg else lab > 0
                    elif mapsrc == "dense" and (t - 1) in dense_pred:
                        src = torch.from_numpy(np.unpackbits(dense_pred[t - 1])[: h * w].reshape(h, w).astype(bool)).cuda()
                    if src is None:
                        src = prev if mapsrc == "gt" else torch.zeros_like(prev)
                    interest_mask = src | (mask > 0) if mask is not None else src
                gt_t = gt_at(t)
                oracle_now, fb_now = False, False                         # diagnostics only
                if m is not None and oracle and gt_t is not None:
                    g = resize(torch.from_numpy(gt_t.astype(np.int64)).cuda()[None], [h, w],
                               InterpolationMode.NEAREST)[0]
                    cover = dilate(interest_mask)
                    hit = [g == o for o, t0 in first.items() if t > t0 and (g == o).sum() >= PATCH_PX
                           and ((g == o) & ~cover).sum() * 2 >= (g == o).sum()]
                    if hit:
                        triggers += 1
                        oracle_now = True
                        if oracle == "key":
                            m = None
                        else:
                            interest_mask = interest_mask | torch.stack(hit).any(0)
                empty += not graph2 and not bool(interest_mask.any())
                all_lost = False
                if graph2:
                    ids = [o for o, t0 in first.items() if t0 < t]
                    enc.frame_inputs = (("reset",) if g2_reset else ()) + (ids, idx_prev, idx_prev2, interest_mask, bool(new))
                    g2_reset = False
                elif estate is not None:                                # run-time object states from t − 1 and t − 2
                    ids = [o for o, t0 in first.items() if t0 < t]
                    estate.objects, all_lost = objects_from_predictions(idx_prev, idx_prev2, ids, ostate, ecfg.motion) \
                        if idx_prev is not None else ([], False)
                    all_lost = all_lost and not new                    # a newly given mask is confidently tracked
                if not graph2 and m is not None and getattr(m, "per_image", False) and (not interest_mask.any() or all_lost):
                    m, fallbacks = fb, fallbacks + 1                    # nothing confidently tracked: the baseline
                    fb_now = True
                    exp_lostfb += all_lost and bool(interest_mask.any())
                if membase and will_mem and m is m_video:          # R40 M: memory from the baseline's features
                    m = fb
                    membase_frames += 1
                if will_mem:
                    mem_frames += 1
                    mem_dense += m is None
                if observer is not None and m is None:                 # dense frame: the detector still sees it
                    m = observer
                dense_now = m is None or (observer is not None and m is observer)
                if enc is not None:
                    enc.structure, enc.prev_mask, enc.interest = m, None if graph2 else interest_mask, None
                prob = proc.step(img, mask, new if new else None, end=end)
                if enc is None:                                          # official Cutie (R30): dense only
                    fl.append(1.0)
                elif graph2:
                    st, ops, fb_now, frn = enc.frame_out
                    any_map, lost_now = frn.flags
                    empty += not any_map
                    fallbacks += fb_now
                    exp_lostfb += lost_now and any_map
                    fl.append(image_flops(st, ops, 0, frn.N0, width=width)["flops"] / dense_flops(frn.N0, width=width, blocks=blocks))
                else:
                    st = enc.states[-1]
                    N = st.coords0.shape[1]
                    fl.append(1.0 if m is None else image_flops(st, ops_of(m), 0, N, width=width)["flops"]
                              / dense_flops(N, width=width, blocks=blocks))
                small = proc.output_prob_to_mask(prob)
                if masks.SMALL:
                    smalls.append((t, small.to(torch.uint8).cpu().numpy()))
                if estate is not None and estate.last:
                    exp_high += estate.last.get("high", 0) / N
                    exp_mid += estate.last.get("mid", 0) / N
                    exp_global += bool(estate.last.get("global"))
                    mbase_frames += bool(estate.last.get("mbase"))
                    estate.last = {}
                if det is not None:
                    n_srp, g_srp = m.fired
                    srp_frames += n_srp > 0 or g_srp
                    srp_share += n_srp / N
                    srp_global += g_srp
                dense_next, extra_next, fired = False, None, False
                if sig_a:
                    ev = ((1 - prob[0]) > 0.2) & ~dilate(interest_mask)
                    if ev.sum() >= PATCH_PX:
                        fired = True
                        dense_next, extra_next = (True, None) if sig_a == "key" else (False, ev)
                if sig_b:
                    lost = False
                    # prob channels are Cutie's temporary ids, `small` holds object ids (identical on LVOS
                    # train and val, where objects are numbered in order of first appearance; mapped anyway)
                    for i, ob in proc.object_manager.tmp_id_to_obj.items():
                        sel = small == ob.id
                        area, hist = int(sel.sum()), areas.setdefault(ob.id, [])
                        if hist and np.mean(hist[-5:]) >= PATCH_PX and (
                                area < 0.5 * np.mean(hist[-5:]) or float(prob[i][sel].mean()) < 0.7):
                            lost = True
                        hist.append(area)
                    if lost:
                        fired, dense_next = True, True
                if hold > 1:
                    history = (history + [prev])[-(hold - 1):]
                prev = small > 0
                if spec == "dense" and need_dense_maps:
                    dense_pred[t] = np.packbits(prev.flatten().cpu().numpy())
                idx_prev2, idx_prev = idx_prev, small
                if sig_a2:                                               # against the next frame's own map
                    ev = patch_evidence(prob, base_map())
                    if ev.any():
                        fired = True
                        if sig_a2 == "key":
                            dense_next = True
                        else:
                            extra_next = ev
                triggers += fired
                jf_now = {}
                if gt_t is not None:
                    up = F.interpolate(prob[None], (H, W), mode="bilinear", align_corners=False)[0]
                    pred = proc.output_prob_to_mask(up)
                    if masks.ON:                                         # plan O: official metrics (reads only)
                        masks.save(out, spec, seq, frames[t].stem, pred, pal)
                    gt = torch.from_numpy(gt_t.astype(np.int64)).cuda()
                    for o, t0 in first.items():
                        if t > t0:
                            j, fm = j_and_f(pred == o, gt == o, tol)
                            js[o].append(j)
                            fs[o].append(fm)
                            jf_now[o] = (float(j), float(fm))
                if dlog is not None:                                     # diagnostics: reads only, changes nothing
                    g_eval = (resize(torch.from_numpy(gt_t.astype(np.int64)).cuda()[None], [h, w],
                                     InterpolationMode.NEAREST)[0] if gt_t is not None else None)
                    dlog.frame(t, first, interest_mask, prob, small,
                               {i: ob.id for i, ob in proc.object_manager.tmp_id_to_obj.items()}, g_eval, fl[-1],
                               fb_now, dense_now, not bool(interest_mask.any()), oracle_now, will_mem,
                               getattr(m, "fired_mask", None) if (det is not None or estate is not None) else None,
                               jf_now)
            if dlog is not None:
                dlog.save()
            if graph2 and enc.frame_out is not None:                  # the frame runner's tier counts (device sums)
                frn = enc.frame_out[3]
                exp_high, exp_mid = int(frn.exd.sum_high) / frn.N0, int(frn.exd.sum_mid) / frn.N0
                exp_global = int(frn.exd.n_global)
                enc.frame_inputs = enc.frame_out = None
            if masks.SMALL:
                masks.save_small(out, spec, seq, smalls)
            obj = [o for o in first if js[o]]
            rows.append({"sequence": seq, "setting": spec, "frames": len(fl), "flops_ratio": float(np.mean(fl)),
                         "objects": obj, "J": [float(np.mean(js[o])) for o in obj],
                         "F": [float(np.mean(fs[o])) for o in obj], "fallback_frames": fallbacks,
                         "triggers": triggers, "mem_frames": mem_frames, "mem_dense": mem_dense,
                         "empty_map_frames": empty, "srp_frames": srp_frames,
                         "srp_share": srp_share / max(1, len(fl)), "srp_global": srp_global,
                         "exp_high_share": exp_high / max(1, len(fl)), "exp_mid_share": exp_mid / max(1, len(fl)),
                         "exp_global": exp_global, "exp_lost_fallbacks": exp_lostfb,
                         "membase_frames": membase_frames, "mbase_frames": mbase_frames})
            print(f"{seq} {spec}: J&F {50 * (np.mean(rows[-1]['J']) + np.mean(rows[-1]['F'])):.1f} "
                  f"FLOPs {rows[-1]['flops_ratio']:.3f} fallback {fallbacks}", flush=True)
            del proc                                                 # return the cached GPU memory between settings,
            torch.cuda.empty_cache()                                 # so 4 workers fit the 16 GB GPU (R14 OOM)
        f.write_text(json.dumps(rows))
    print("done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
