"""R8 wall clock of every method at its validated operating point (DAVIS val, R2 and R4: the setting that reaches
dense-level J&F, or the best a method reaches), batch 1, DAVIS frames 448 x 784, bf16, idle GPU, median over 120
frames after warm-up. Eager for all; CUDA graphs for every method whose shapes are fixed per frame (ToMe, PiToMe,
AiluRus, Expedite: one graph) and for ours (one graph per protected-set size class); ALGM's and G2TM's token counts depend on the
image (as with the R19 novelty map), so they have no graph. Usage: python -m src.stage_b.r8_timing <out.json> <vits|vitb|vitl> [setting,...]"""
import json
import sys
from pathlib import Path

import torch

from src.stage_a.model import load_backbone

from .graphs import GraphRunner
from .model import EffModel
from .n7_timing import frames, timed
from .p2_timing import HW
from .timing import from_name, gpu_idle

REPO = {"vits": None, "vitb": "facebook/dinov3-vitb16-pretrain-lvd1689m", "vitl": "facebook/dinov3-vitl16-pretrain-lvd1689m"}
POINTS = {   # (setting, J&F change on DAVIS val, FLOPs) from R2 / R4
    "vits": ["ftome_l0.7_t0.01_floor64_dil1_noprop_cls64", "tome_r64_from1_noprop", "ailurus_m0.5_at3", "algm_t0.98",
             "pitome_k0.97", "expedite_m0.49_at5_t0.1"],
    "vitb": ["ftome_l0.7_t0.01_floor128_dil2_noprop_cls64", "ftome_l0.3_t0.01_floor128_dil2_noprop_cls64",
             "tome_r64_from1_noprop", "ailurus_m0.5_at3", "algm_t0.97", "pitome_k0.97", "expedite_m0.49_at5_t0.05"],
    "vitl": ["ftome_l0.7_t0.01_floor128_dil1_noprop_cls64", "tome_r24_from1_noprop", "ailurus_m0.5_at4", "algm_t0.99",
             "pitome_k0.98", "expedite_m0.49_at9_t0.05"],
}


@torch.no_grad()
def main(out_path, bb, names=None):
    """names: optional comma-separated settings replacing POINTS[bb] (dense is always timed)."""
    if not gpu_idle():
        sys.exit("r8_timing: another process is using the GPU; wall clock needs an idle GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = EffModel(load_backbone("cuda", REPO[bb]) if REPO[bb] else load_backbone("cuda")).cuda()
    model.fast = True
    blocks = len(model.bb.model.layer)
    fr = frames()
    res = []

    def add(name, mode, fn, **extra):
        r = {"setting": name, "mode": mode, **timed(fn, fr), **extra}
        res.append(r)
        print(f"{bb} {name:48s} {mode:6s} {r['ms_per_frame']:.2f} ms/frame", flush=True)

    add("dense", "eager", lambda x, i: model(x, dtype=torch.bfloat16))
    gd = GraphRunner(model, "dense", HW)
    add("dense", "graph", lambda x, i: gd.run(x))
    for name in (names.split(",") if names else POINTS[bb]):
        per_image = getattr(from_name(name, HW[0] // 16, blocks), "per_image", False)
        add(name, "eager", lambda x, i: model(x, structure=from_name(name, HW[0] // 16, blocks), dtype=torch.bfloat16,
                                              interest=i if per_image else None))
        if name.startswith(("algm", "palgm", "g2tm", "pg2tm", "nov")):  # token count depends on the image
            continue
        try:
            gr = GraphRunner(model, name, HW)
            for x, i in fr:
                gr.run(x, i)
            add(name, "graph", lambda x, i: gr.run(x, i), classes=len(gr.graphs))
        except Exception as e:                                           # report, do not hide, a capture failure
            print(f"{bb} {name}: graph capture failed: {type(e).__name__}: {e}", flush=True)
            res.append({"setting": name, "mode": "graph", "error": f"{type(e).__name__}: {e}"})
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
