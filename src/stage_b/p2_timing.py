"""Per-frame wall clock on DAVIS frames (batch 1, the video setting): real kernels (fast path), bf16, idle GPU only.
Focus maps for per-image methods are each frame's true object mask (realistic coverage; the map's source does not
change the cost). Median over frames after warm-up. Usage: python -m src.stage_b.p2_timing <out.json> [backbone|-] [setting,setting,...]"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.stage_a.model import load_backbone

from .model import EffModel
from .p2_davis import load_frame, load_mask
from .timing import from_name, gpu_idle

ROOT = Path("/work/data/davis/DAVIS")
HW = (448, 784)
SEQS = ["bike-packing", "dogs-jump", "parkour"]
SETTINGS = ["dense", "tome_r64_from1", "tome_r80_from1", "tome_r64_from1_noprop", "tome_r80_from1_noprop",
            "itome_l0.3_t0.01_floor128_dil2", "itome_l0.5_t0.01_floor128_dil2", "itome_l0.5_t0.01_floor128_dil2_noprop",
            "iprune_k0.1_at3_t0.01_dil2", "iprune_k0.25_at3_t0.01_dil2"]
WARMUP = 10


@torch.no_grad()
def main(out_path, backbone=None, settings=None):
    """settings: optional comma-separated list of setting names (default SETTINGS); backbone "-" for the default."""
    if backbone == "-":
        backbone = None
    names = settings.split(",") if settings else SETTINGS
    if not gpu_idle():
        sys.exit("p2_timing: another process is using the GPU; wall clock needs an idle GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = EffModel(load_backbone("cuda", backbone) if backbone else load_backbone("cuda")).cuda()
    model.fast = True
    frames = []
    for seq in SEQS:
        imgs = sorted((ROOT / "JPEGImages/480p" / seq).glob("*.jpg"))[:40]
        annos = sorted((ROOT / "Annotations/480p" / seq).glob("*.png"))[:40]
        for i, a in zip(imgs, annos):
            x = load_frame(i, HW, "cuda")[None]
            m = load_mask(a, HW, "cuda")
            frames.append((x, F.avg_pool2d((m > 0).float()[None, None], 16)[0, 0].flatten()[None]))
    res = []
    for name in names:
        times = []
        m_video = from_name(name, HW[0] // 16)                         # one instance (temporal reuse keeps its cache)
        for k, (x, interest) in enumerate(frames[:WARMUP] + frames):
            m = m_video
            per_image = m is not None and getattr(m, "per_image", False)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            model(x, structure=m, dtype=torch.bfloat16, interest=interest if per_image else None)
            e.record()
            torch.cuda.synchronize()
            if k >= WARMUP:
                times.append(s.elapsed_time(e))
        r = {"setting": name, "ms_per_frame": float(np.median(times)),
             "ms_iqr": [float(np.percentile(times, 25)), float(np.percentile(times, 75))], "frames": len(times)}
        res.append(r)
        print(f"{name:40s} {r['ms_per_frame']:.2f} ms/frame", flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
