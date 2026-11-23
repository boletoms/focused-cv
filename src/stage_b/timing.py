"""Wall-clock timing (plan step B2): whole-model forward time per image on an idle GPU, with the fast attention path
(SDPA, FlexAttention), bf16 linear layers and bf16 attention. Median over timed passes after warm-up; mask building
(the method's clustering plus block-mask creation) is reported separately. Refuses to run if another process holds the
GPU. Usage: python -m src.stage_b.timing <methods.json> <out.json>, where methods.json lists
{"resolution": 896, "batch": 8, "methods": ["dense", "tome_r64_from1", ...]} entries (names as in b1_calib)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.stage_a.data import ImageList
from src.stage_a.model import load_backbone

from .methods import Group, Prune, ToMe, Windows
from .model import EffModel

WARMUP, REPEATS = 10, 30


def from_name(name, grid, blocks=12):
    """Rebuild a method from its b1 setting name (dense, tome_r{r}_from{f}, prune_k{k}_at{a-b-c},
    group_k{K}_m{m}_tau{t}, windows_w{w}); combos are 'a+b'."""
    if name == "dense":
        return None
    if name.startswith("dsim-") and ":" in name:  # R21 combination: our budget with DTEM's metric
        from .dtem import load_similarity
        return load_similarity(name, from_name(name.split(":", 1)[1], grid, blocks), blocks)
    if name.startswith("dtem_"):                # R21: learned merging metric (src/stage_b/dtem.py)
        from .dtem import load
        return load(name, blocks)
    if name.startswith("nov") and ":" in name:  # R19 novelty map: nov{q}:<map-aware method>
        from . import baselines as bl
        head, inner = name.split(":", 1)
        return bl.Novelty(from_name(inner, grid, blocks), float(head[3:]), blocks=blocks)
    if ":" in name:                            # temporal reuse wrapper: reuse{delta}a{age}:<inner>
        from .methods import Reuse
        head, inner = name.split(":", 1)
        delta, rest = head[5:].split("a")
        age, theta = (rest.split("t") + ["0.01"])[:2]
        return Reuse(from_name(inner, grid, blocks), delta=float(delta), max_age=int(age), theta=float(theta))
    if "+" in name:
        from .methods import Combo
        a, b = name.split("+")
        return Combo(from_name(a, grid), from_name(b, grid))
    f = name.split("_")
    if f[0] == "iprune":
        from .methods import InterestPrune
        at = tuple(int(a) - 1 for a in f[2][2:].split("-"))
        dil = next((int(x[3:]) for x in f[3:] if x.startswith("dil")), 0)
        rest = next((int(x[7:]) - 1 for x in f[3:] if x.startswith("restore")), None)
        return InterestPrune(float(f[1][1:]), at=at, theta=float(f[3][1:]), dilate=dil, restore_at=rest)
    if f[0] in ("ptome", "palgm", "pg2tm"):                                # R6 map-aware baselines
        from . import baselines as bl
        dil = next((int(x[3:]) for x in f[2:] if x.startswith("dil")), 0)
        if f[0] == "pg2tm":
            return bl.G2TMProtect(float(f[1][1:]), at=int(f[2][2:]), dilate=dil)
        if f[0] == "ptome":
            return bl.ToMeProtect(int(f[1][1:]), dilate=dil)
        return bl.ALGMProtect(float(f[1][1:]), layers=(1, 7) if blocks == 24 else (1, 5), dilate=dil)
    if f[0] == "ttome":                                 # R40 second baseline: ToMe's rules outside the map, our budget on top
        from . import baselines as bl
        g = lambda key, d=0: next((int(x[len(key):]) for x in f[3:] if x.startswith(key)), d)
        return bl.ToMeTiered(int(f[1][1:]), float(f[2][1:]), floor=g("floor"), dilate=g("dil"), size_class=g("cls"),
                             as_rate=bool(g("tr")))
    if f[0] in ("aitome", "sitome", "titome"):          # R32: ALGM inside the region; R35: inside surprise areas only;
        from . import baselines as bl                   # R40: ALGM's rules on every token outside the map, our budget on top
        g = lambda key, d=0: next((int(x[len(key):]) for x in f[3:] if x.startswith(key)), d)
        if f[0] == "titome":
            return bl.ALGMTiered(float(f[1][1:]), float(f[2][1:]), floor=g("floor"), dilate=g("dil"), size_class=g("cls"),
                                 layers=(1, 7) if blocks == 24 else (1, 5), high_algm=bool(g("ha")))
        return {"aitome": bl.ALGMInterest, "sitome": bl.ALGMSurprise}[f[0]](float(f[1][1:]), float(f[2][1:]), floor=g("floor"), dilate=g("dil"), size_class=g("cls"),
                               layers=(1, 7) if blocks == 24 else (1, 5))
    if f[0] in ("pitome", "algm", "ailurus", "expedite", "g2tm"):          # R1 literature baselines
        from . import baselines as bl
        if f[0] == "g2tm":
            return bl.G2TM(float(f[1][1:]), at=int(f[2][2:]))
        if f[0] == "pitome":
            return bl.PiToMe(float(f[1][1:]), blocks=blocks)
        if f[0] == "algm":
            return bl.ALGM(float(f[1][1:]), layers=(1, 7) if blocks == 24 else (1, 5))
        at = int(f[2][2:])
        if f[0] == "ailurus":
            return bl.AiluRus(float(f[1][1:]), start_loc=at)
        tmp = next((float(x[1:]) for x in f[3:] if x.startswith("t")), 0.02)
        return bl.Expedite(float(f[1][1:]), location=at, temperature=tmp)
    if f[0] == "ftome":
        from .methods import FastInterestToMe
        floor = next((int(x[5:]) for x in f[3:] if x.startswith("floor")), 0)
        dil = next((int(x[3:]) for x in f[3:] if x.startswith("dil")), 0)
        cls = next((int(x[3:]) for x in f[3:] if x.startswith("cls")), 0)
        return FastInterestToMe(float(f[1][1:]), theta=float(f[2][1:]), floor=floor, dilate=dil, prop="noprop" not in f,
                                size_class=cls)
    if f[0] == "otome":
        from .methods import OneShotToMe
        at = int(f[2][2:]) - 1
        floor = next((int(x[5:]) for x in f[3:] if x.startswith("floor")), 0)
        dil = next((int(x[3:]) for x in f[3:] if x.startswith("dil")), 0)
        return OneShotToMe(float(f[1][1:]), at=at, theta=float(f[3][1:]), floor=floor, dilate=dil, prop="noprop" not in f)
    if f[0] == "gtome":
        from .methods import GradedToMe
        floor = next((int(x[5:]) for x in f[2:] if x.startswith("floor")), 0)
        dil = next((int(x[3:]) for x in f[2:] if x.startswith("dil")), 0)
        return GradedToMe(float(f[1][1:]), floor=floor, dilate=dil, prop="noprop" not in f)
    if f[0] == "itome":
        from .methods import InterestToMe
        first = next((int(x[4:]) - 1 for x in f[3:] if x.startswith("from")), 0)
        floor = next((int(x[5:]) for x in f[3:] if x.startswith("floor")), 0)
        dil = next((int(x[3:]) for x in f[3:] if x.startswith("dil")), 0)
        return InterestToMe(float(f[1][1:]), theta=float(f[2][1:]), first=first, floor=floor, dilate=dil,
                            prop="noprop" not in f)
    if f[0] == "tome":
        return ToMe(int(f[1][1:]), first=int(f[2][4:]) - 1, prop="noprop" not in f)
    if f[0] == "prune":
        restore = int(f[3][7:]) - 1 if len(f) > 3 else None
        return Prune(float(f[1][1:]), at=tuple(int(a) - 1 for a in f[2][2:].split("-")), restore_at=restore)
    if f[0] == "group":
        tau = None if f[3] == "taunone" else float(f[3][3:])
        blocks = tuple(range(12)) if len(f) < 5 else tuple(range(int(f[4][1:].split("-")[0]) - 1, int(f[4].split("-")[1])))
        return Group(int(f[1][1:]), m=int(f[2][1:]), tau=tau, blocks=blocks)
    if f[0] in ("graph", "blockoracle"):        # oracles; the caller sets .trees or .current
        from .methods import BlockOracle, Graph
        return Graph(name.split("_", 1)[1], grid) if f[0] == "graph" else BlockOracle(name.split("_", 1)[1])
    if f[0] == "oracle":                       # needs .current set by the caller (partitions from a2g-calib)
        from .methods import Oracle
        return Oracle(name.split("_", 1)[1])
    if f[0] == "windows":
        return Windows(int(f[1][1:]), grid)
    raise ValueError(name)


def gpu_idle():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return all(int(l.split(",")[1].split()[0]) < 512 for l in out.strip().splitlines() if l.strip())


@torch.no_grad()
def time_one(model, x, method):
    model.events = []
    for _ in range(WARMUP):
        model(x, structure=method, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    total, build = [], []
    for _ in range(REPEATS):
        model.events = []
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        model(x, structure=method, dtype=torch.bfloat16)
        e.record()
        torch.cuda.synchronize()
        total.append(s.elapsed_time(e))
        build.append(sum(a.elapsed_time(b) for a, b in model.events))
    n = x.shape[0]
    return {"ms_per_image": float(np.median(total)) / n, "ms_per_image_iqr": [float(np.percentile(total, 25)) / n,
            float(np.percentile(total, 75)) / n], "mask_build_ms_per_image": float(np.median(build)) / n}


def main(spec_path, out_path):
    if not gpu_idle():
        sys.exit("timing: another process is using the GPU; wall clock needs an idle GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = EffModel(load_backbone("cuda")).cuda()
    model.fast, model.compiled = True, True
    results = []
    for entry in json.loads(Path(spec_path).read_text()):
        res, bs = entry["resolution"], entry["batch"]
        ds = ImageList("calibration", res)
        x = torch.stack([ds[i][0] for i in range(bs)]).cuda()
        for name in entry["methods"]:
            r = time_one(model, x, from_name(name, res // 16))
            results.append({"resolution": res, "batch": bs, "setting": name, **r})
            print(res, name, f"{r['ms_per_image']:.3f} ms/image (mask build {r['mask_build_ms_per_image']:.3f})", flush=True)
    Path(out_path).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
