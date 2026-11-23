"""R10 evaluation: Cutie with the DINOv3 pixel encoder on DAVIS 2017 val, Cutie's InferenceCore and DAVIS settings
(480p, memory every 5 frames, top-k 30, no flip), every Stage B method applied to the ViT. The focus map of frame t
is the given mask at t = 0 and Cutie's own prediction at t - 1 afterwards (closed loop); empty map -> dense frame;
optional I-frames every `keyframe` frames. J and F per object (objects given in the first frame), our implementation
at 480p (boundary tolerance 0.8% of the diagonal, as in P2). FLOPs of the ViT per frame against dense.
R24 suffixes (as `eval_lvos.py`, short clips): `@hold{h}`, `@motion`, `@surpAmap` / `@surpAkey`, `@surpBkey`; rows count
trigger frames. R27: `@surpA2map` / `@surpA2key` (signal A corrected, see `eval_lvos.py`); rows count empty-map frames
whatever ran on them. R28: `@srp_<detector>` (surprise, as `eval_lvos.py`; the empty-map fallback here is dense, so
those frames run the observer).
R40: `@exp-...` (the redesign, as `eval_lvos.py`; needs `setting~fallback`). R29: `setting~fallback` runs that method on empty-map frames instead of dense (as `eval_lvos.py`); environment
VOS_FAST=1 runs the ViT on the fast bf16 SDPA path of the long-video evaluator instead of eager fp32 attention.
Usage (ml-base:vos): python -m src.stage_b.vos.eval <weights.pth> <out_dir> <setting,setting,...> [keyframe|-] [max videos]"""
import os
import sys

os.environ.setdefault("LOCAL_RANK", "0")
CUTIE = "/work/third_party/Cutie"
sys.path.insert(0, CUTIE)
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

DAVIS = Path("/work/data/davis/DAVIS")


def build(weights, dataset="d17-val"):
    import cutie.model.big_modules as bm
    import cutie.model.cutie as cm
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from cutie.inference.utils.args_utils import get_dataset_cfg
    from src.stage_b.vos.dino_encoder import VIT_WIDTH, DinoPixelEncoder
    bm.PixelEncoder = cm.PixelEncoder = DinoPixelEncoder
    with initialize_config_dir(version_base="1.3.2", config_dir=f"{CUTIE}/cutie/config", job_name="eval_config"):
        cfg = compose(config_name="eval_config", overrides=["model=small", f"model.pixel_encoder.ms_dims=[{VIT_WIDTH},128,64]",
                                                            f"dataset={dataset}"])
    with open_dict(cfg):
        cfg["weights"] = weights
    get_dataset_cfg(cfg)
    net = cm.CUTIE(cfg).cuda().eval()
    net.load_weights(torch.load(weights, map_location="cuda"))
    return net, cfg


@torch.inference_mode()
def main(weights, out_dir, settings, keyframe=None, max_seq=None):
    from src.stage_b.vos.eval_lvos import deterministic
    deterministic()                                   # R-29: deterministic kernels (default from 2026-11-20)
    from cutie.inference.inference_core import InferenceCore
    from src.stage_b.flops import dense_flops, image_flops
    from src.stage_b.methods import ops_of
    from src.stage_b.p2_davis import j_and_f, shift_by_velocity
    from src.stage_b.vos.eval_lvos import PATCH_PX, dilate, patch_evidence
    from src.stage_b.timing import from_name
    from src.stage_b.vos import masks
    keyframe = int(keyframe) if keyframe and keyframe != "-" else None
    net, cfg = build(weights)
    enc = net.pixel_encoder
    if os.environ.get("VOS_FAST") == "1":
        enc.vit.fast = True
    width, blocks = enc.vit.bb.config.hidden_size, len(enc.vit.bb.model.layer)
    out = Path(out_dir)
    (out / "sequences").mkdir(parents=True, exist_ok=True)
    seqs = (DAVIS / "ImageSets/2017/val.txt").read_text().split()[: int(max_seq) if max_seq else None]
    for seq in seqs:
        f = out / "sequences" / f"{seq}.json"
        if f.exists():
            continue
        try:
            os.close(os.open(out / "sequences" / f"{seq}.claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        frames = sorted((DAVIS / "JPEGImages/480p" / seq).glob("*.jpg"))
        annos = sorted((DAVIS / "Annotations/480p" / seq).glob("*.png"))
        gt = [torch.from_numpy(np.array(Image.open(a))).long().cuda() for a in annos]
        objects = sorted(set(torch.unique(gt[0]).tolist()) - {0})
        H, W = gt[0].shape
        tol = max(1, round(0.008 * (H ** 2 + W ** 2) ** 0.5))
        pal = masks.palette(annos[0]) if masks.ON else None
        rows = []
        for name in ["dense"] + settings.split(","):
            spec, *mods = name.split("@")
            base, _, fb_name = spec.partition("~")
            hold = next((int(x[4:]) for x in mods if x.startswith("hold")), 1)
            motion = "motion" in mods
            sig_a = next((x[5:] for x in mods if x in ("surpAmap", "surpAkey")), None)       # "map" | "key"
            sig_a2 = next((x[6:] for x in mods if x in ("surpA2map", "surpA2key")), None)    # R27 correction
            sig_b = "surpBkey" in mods
            srp = next((x for x in mods if x.startswith("srp_")), None)
            expm = next((x for x in mods if x.startswith("exp-")), None)          # R40 redesign (as `eval_lvos.py`)
            history, idx_prev, idx_prev2, dense_next, extra_next, areas, triggers = [], None, None, False, None, {}, 0
            empty = 0
            proc = InferenceCore(net, cfg=cfg)
            proc.max_internal_size = -1
            m_video = None if base == "dense" else from_name(base, (H + 15) // 16, blocks)
            fb = from_name(fb_name, (H + 15) // 16, blocks) if fb_name else None
            det, observer, srp_frames, srp_share, srp_global = None, None, 0, 0.0, 0
            if srp:
                from src.stage_b.surprise import SurpriseMap, detector_from_name
                det = detector_from_name(srp)
                m_video = SurpriseMap(m_video, det) if m_video is not None else None
                if fb_name:
                    from src.stage_b.surprise import protected_form
                    fb = SurpriseMap(from_name(protected_form(fb_name), (H + 15) // 16, blocks), det)
                observer = SurpriseMap(None, det)
            estate, ostate, exp_high, exp_mid, exp_global, exp_lostfb = None, {}, 0.0, 0.0, 0, 0
            if expm:
                from src.stage_b.expect import ExpectMap, ExpectState, config_from_name, objects_from_predictions
                assert m_video is not None and fb is not None and not srp, "@exp needs a reduced method and its fallback"
                ecfg, estate = config_from_name(expm), ExpectState()
                m_video, fb = ExpectMap(m_video, estate, ecfg), ExpectMap(fb, estate, ecfg, passive=True)
                motion = motion or ecfg.motion
            graph2 = os.environ.get("VOS_GRAPH") == "2" and estate is not None   # R40 plan L (as `eval_lvos.py`)
            if graph2:
                assert not (keyframe or hold > 1 or sig_a or sig_a2 or sig_b), "VOS_GRAPH=2 covers the redesign's own settings only"
                enc.frame_setting, g2_reset = name, True
            prev, js, fs, fl, fallbacks = gt[0] > 0, {o: [] for o in objects}, {o: [] for o in objects}, [], 0

            def base_map():                                      # the map before triggers: prediction, motion, hold
                b = prev
                if motion and idx_prev is not None and idx_prev2 is not None:
                    b = b | (shift_by_velocity(idx_prev, idx_prev2) > 0)
                for hm in history[-(hold - 1):] if hold > 1 else []:
                    b = b | hm
                return b
            for t, fp in enumerate(frames):
                img = to_tensor(Image.open(fp).convert("RGB")).cuda().float()
                m = m_video
                if m is not None and keyframe and t > 0 and t % keyframe == 0:
                    m = None
                if m is not None and dense_next:
                    m = None
                interest = prev if graph2 else base_map()               # graph2: the motion is added in the runner
                if extra_next is not None:
                    interest = interest | extra_next
                empty += not graph2 and not bool(interest.any())
                all_lost = False
                if graph2:
                    enc.frame_inputs = (("reset",) if g2_reset else ()) + (list(objects), idx_prev, idx_prev2, interest, False)
                    g2_reset = False
                elif estate is not None:                                # run-time object states from t − 1 and t − 2
                    estate.objects, all_lost = objects_from_predictions(idx_prev, idx_prev2, objects, ostate, ecfg.motion) \
                        if idx_prev is not None else ([], False)
                if not graph2 and m is not None and getattr(m, "per_image", False) and (not interest.any() or all_lost):
                    m, fallbacks = fb, fallbacks + 1
                    exp_lostfb += all_lost and bool(interest.any())
                if observer is not None and m is None:                 # dense frame: the detector still sees it
                    m = observer
                enc.structure, enc.prev_mask, enc.interest = m, None if graph2 else interest, None
                prob = proc.step(img, gt[0], objects=objects) if t == 0 else proc.step(img)
                pred = proc.output_prob_to_mask(prob)
                if masks.ON:                                             # plan O: official metrics (reads only)
                    masks.save(out, name, seq, frames[t].stem, pred, pal)
                if estate is not None and estate.last:
                    N0 = enc.states[-1].coords0.shape[1]
                    exp_high += estate.last.get("high", 0) / N0
                    exp_mid += estate.last.get("mid", 0) / N0
                    exp_global += bool(estate.last.get("global"))
                    estate.last = {}
                if det is not None:
                    n_srp, g_srp = m.fired
                    srp_frames += n_srp > 0 or g_srp
                    srp_share += n_srp / enc.states[-1].coords0.shape[1]
                    srp_global += g_srp
                dense_next, extra_next, fired = False, None, False
                if sig_a:                                                # as eval_lvos (R15)
                    ev = ((1 - prob[0]) > 0.2) & ~dilate(interest)
                    if ev.sum() >= PATCH_PX:
                        fired = True
                        dense_next, extra_next = (True, None) if sig_a == "key" else (False, ev)
                if sig_b:
                    lost = False
                    for i, ob in proc.object_manager.tmp_id_to_obj.items():
                        sel = pred == ob.id
                        area, hist = int(sel.sum()), areas.setdefault(ob.id, [])
                        if hist and np.mean(hist[-5:]) >= PATCH_PX and (
                                area < 0.5 * np.mean(hist[-5:]) or float(prob[i][sel].mean()) < 0.7):
                            lost = True
                        hist.append(area)
                    if lost:
                        fired, dense_next = True, True
                if hold > 1:
                    history = (history + [prev])[-(hold - 1):]
                idx_prev2, idx_prev = idx_prev, pred
                prev = pred > 0
                if sig_a2:                                           # against the next frame's own map
                    ev = patch_evidence(prob, base_map())
                    if ev.any():
                        fired = True
                        if sig_a2 == "key":
                            dense_next = True
                        else:
                            extra_next = ev
                triggers += fired
                st = enc.states[-1]
                N = st.coords0.shape[1]
                if graph2:
                    st, ops, fb_now, frn = enc.frame_out
                    any_map, lost_now = frn.flags
                    empty += not any_map
                    fallbacks += fb_now
                    exp_lostfb += lost_now and any_map
                    fl.append(image_flops(st, ops, 0, N, width=width)["flops"] / dense_flops(N, width=width, blocks=blocks))
                else:
                    fl.append(1.0 if m is None else image_flops(st, ops_of(m), 0, N, width=width)["flops"]
                              / dense_flops(N, width=width, blocks=blocks))
                if t > 0 and t < len(gt):
                    for o in objects:
                        j, fm = j_and_f(pred == o, gt[t] == o, tol)
                        js[o].append(j)
                        fs[o].append(fm)
            if graph2 and enc.frame_out is not None:                  # the frame runner's tier counts (device sums)
                frn = enc.frame_out[3]
                exp_high, exp_mid = int(frn.exd.sum_high) / frn.N0, int(frn.exd.sum_mid) / frn.N0
                exp_global = int(frn.exd.n_global)
                enc.frame_inputs = enc.frame_out = None
            rows.append({"sequence": seq, "setting": name, "frames": len(frames), "flops_ratio": float(np.mean(fl)),
                         "J": [float(np.mean(js[o])) for o in objects], "F": [float(np.mean(fs[o])) for o in objects],
                         "fallback_frames": fallbacks, "keyframe": keyframe, "triggers": triggers,
                         "empty_map_frames": empty, "srp_frames": srp_frames,
                         "srp_share": srp_share / max(1, len(fl)), "srp_global": srp_global,
                         "exp_high_share": exp_high / max(1, len(fl)), "exp_mid_share": exp_mid / max(1, len(fl)),
                         "exp_global": exp_global, "exp_lost_fallbacks": exp_lostfb})
        f.write_text(json.dumps(rows))
        print(f"{seq} done", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
