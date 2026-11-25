"""Closed-loop timing inside a real long-video evaluation (audits 2026-11-23; the paper's Table 5 times three DAVIS clips
whose maps come from ground-truth masks, so no frame falls back). Here the evaluator runs Cutie on LVOS V2 valid videos
with the method on the timed implementation (`VOS_GRAPH=2 ...`), so maps come from Cutie's own predictions and frames
whose objects are all lost run the fallback (plain ALGM on the same implementation). CUDA events time, per frame, the
pixel encoder (the frozen ViT with the tier planning, plus Cutie's ResNet stem for the stride-8 and stride-4 skips) and
Cutie's whole step (encoder, memory readout, decoder, memory update). The first video warms up and is not counted.
Writes `r40_e2e_timing_<tag>.json`: medians and means per frame type (main, fallback) and the encoder's share of the
step. Usage (idle GPU, one process): python -m src.stage_b.vos.r40_e2e_timing <weights> <out dir> <setting> <videos file> <tag>"""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

from .r40_compare import Q


def main(weights, out_dir, setting, videos, tag):
    sys.path.insert(0, "/work/third_party/Cutie")                    # as eval.py: Cutie's package
    from cutie.inference.inference_core import InferenceCore
    from .dino_encoder import DinoPixelEncoder
    rec, cur = [], {}
    f_enc, f_step = DinoPixelEncoder.forward, InferenceCore.step

    def enc(self, *a, **k):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); out = f_enc(self, *a, **k); e.record()
        cur["enc"] = (s, e)
        cur["fb"] = bool(self.frame_out[2]) if self.frame_out is not None else None
        return out

    def step(self, *a, **k):
        cur.clear()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); out = f_step(self, *a, **k); e.record()
        torch.cuda.synchronize()
        if "enc" in cur:
            rec.append((None, cur["fb"], cur["enc"][0].elapsed_time(cur["enc"][1]), s.elapsed_time(e)))
        return out

    DinoPixelEncoder.forward, InferenceCore.step = enc, step
    sys.argv = ["eval_lvos", weights, out_dir, setting, "-", "v2_valid"]
    import os
    os.environ["VOS_SEQS"] = videos

    runpy.run_module("src.stage_b.vos.eval_lvos", run_name="__main__")
    rows = {p.stem: json.loads(p.read_text()) for p in sorted((Path(out_dir) / "sequences").glob("*.json"))}
    counts = [r[0]["frames"] for s_, r in sorted(rows.items())]
    first = counts[0] if counts else 0
    body = rec[first:]                                     # first video: graph capture and compilation
    enc_ms = np.array([r[2] for r in body]); step_ms = np.array([r[3] for r in body])
    fb = np.array([bool(r[1]) for r in body])
    idx = np.concatenate([np.arange(c) for c in counts[1:]]) if len(counts) > 1 else np.zeros(0, int)
    assert len(idx) == len(body), (len(idx), len(body))              # frames of each video in processing order
    steady = idx >= 30                                              # after each video's start (graph capture, buffers)
    q = lambda x, m=None: ({"median": float(np.median(x)), "mean": float(np.mean(x)), "p90": float(np.percentile(x, 90)),
                            "p99": float(np.percentile(x, 99)), "frames": int(len(x)),
                            "mean_after_30": float(np.mean(x[m])) if m is not None and m.any() else None} if len(x) else None)
    out = {"setting": setting, "videos": sorted(rows)[1:], "warmup_video": sorted(rows)[0] if rows else None,
           "fallback_share": float(fb.mean()) if len(fb) else None,
           "encoder_ms": {"all": q(enc_ms, steady), "main": q(enc_ms[~fb], steady[~fb]), "fallback": q(enc_ms[fb], steady[fb])},
           "step_ms": {"all": q(step_ms, steady), "main": q(step_ms[~fb], steady[~fb]), "fallback": q(step_ms[fb], steady[fb])},
           "encoder_share_of_step": float(enc_ms.sum() / step_ms.sum()) if len(step_ms) else None}
    out["peak_memory_mb"] = {"allocated": float(torch.cuda.max_memory_allocated() / 2 ** 20),
                             "reserved": float(torch.cuda.max_memory_reserved() / 2 ** 20)}   # whole process, including the warm-up video
    (Q / f"r40_e2e_timing_{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
