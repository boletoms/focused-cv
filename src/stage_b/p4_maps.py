"""Q4 maps without ground truth (queue core gate; docs/records/STAGE_B_ABLATION_QUEUE.md). VOC images; the region of
interest is the largest object class; the model gets an focus map from one of:
- gt: the true region (reference);
- box: the region's bounding box (what a user would draw);
- click: one point inside the region (its most interior pixel), grown to a region by feature similarity in a cheap
  224-pixel pass (patches whose cosine similarity to the clicked patch exceeds `click_sim`, or, if "adaptive", halfway
  between 1 and the image's median similarity);
- saliency: no user input; the `saliency_top` share of patches CLS attends to most in the same cheap pass.
The cheap pass's FLOPs are added (relative to the dense pass at the working resolution). Scores: pixel accuracy of the
dense probe inside and outside the true region; FLOPs. Usage: python -m src.stage_b.p4_maps [config] [res]"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.ablate.runinfo import git_dirty, git_hash
from src.stage_a.data import ImageList
from src.stage_a.evaluate import probe_logits
from src.stage_a.model import N_PREFIX, load_backbone

from .b1_calib import CLASSES, IGNORE, mask_of
from .flops import dense_flops, image_flops
from .methods import ops_of
from .model import EffModel
from .timing import from_name

CHEAP = 224


class CaptureCLS:
    """Records the last block's CLS attention to the patches (mean over heads) in a dense pass."""
    def after_attn(self, b, x, k, p, st):
        if b == 11:
            self.cls = p[0, :, 0, N_PREFIX:].mean(0)
        return x


def interior_point(region):
    """Pixel of the region farthest from its border (iterated erosion), as (y, x)."""
    m = region.float()[None, None]
    last = m
    while m.sum() > 0:
        last = m
        m = -F.max_pool2d(-m, 3, 1, 1)
    ys, xs = last[0, 0].nonzero(as_tuple=True)
    return int(ys[len(ys) // 2]), int(xs[len(xs) // 2])


def maps_for(model, x, region, g, cfg, dtype):
    """Focus maps (flattened g x g) for each source, plus the cheap pass's cost in dense-pass units at size g."""
    touch = lambda m: F.max_pool2d(m.float()[None, None], 16)[0, 0].flatten()
    out = {"gt": touch(region)}
    ys, xs = region.nonzero(as_tuple=True)
    box = torch.zeros_like(region)
    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    out["box"] = touch(box)
    xc = F.interpolate(x, size=(CHEAP, CHEAP), mode="bilinear", align_corners=False)
    cap = CaptureCLS()
    with torch.no_grad():
        f = model(xc, structure=cap, dtype=dtype).final[0, N_PREFIX:]
    gc = CHEAP // 16
    feat = F.normalize(f.float(), dim=-1)
    py, px = interior_point(region)
    H = region.shape[0]
    idx = (py * gc // H) * gc + (px * gc // H)
    sim = (feat @ feat[idx]).view(1, 1, gc, gc)
    thr = (1.0 + sim.median()) / 2 if cfg["click_sim"] == "adaptive" else cfg["click_sim"]   # halfway to the median
    click = (F.interpolate(sim, size=(g, g), mode="bilinear", align_corners=False)[0, 0] > thr).float()
    out["click"] = click.flatten()
    # N2 sources: looser click growth, three clicks, DINO-style foreground (first principal component)
    up = lambda m: F.interpolate(m.view(1, 1, gc, gc).float(), size=(g, g), mode="bilinear", align_corners=False)[0, 0]
    med = sim.median()
    out["click_loose"] = (up(sim) > med + 0.25 * (1 - med)).float().flatten()
    ys_r, xs_r = region.nonzero(as_tuple=True)
    gen = torch.Generator().manual_seed(int(region.sum()))
    picks = [(py, px)] + [(int(ys_r[j]), int(xs_r[j])) for j in torch.randint(0, len(ys_r), (2,), generator=gen).tolist()]
    grown = torch.zeros(g * g, device=region.device)
    for yy, xx in picks:
        s2 = feat @ feat[(yy * gc // H) * gc + (xx * gc // H)]
        m2 = s2.median()
        grown = torch.maximum(grown, (up(s2) > m2 + 0.25 * (1 - m2)).float().flatten())
    out["click3"] = grown
    fc = feat - feat.mean(0, keepdim=True)
    pc = torch.linalg.svd(fc, full_matrices=False)[2][0]
    proj = (fc @ pc).view(gc, gc)
    border = torch.cat([proj[0], proj[-1], proj[:, 0], proj[:, -1]])
    if border.mean() > 0:                                             # border = background
        proj = -proj
    out["pca"] = (up(proj) > 0).float().flatten()
    cls = cap.cls.float().view(1, 1, gc, gc)
    sal = F.interpolate(cls, size=(g, g), mode="bilinear", align_corners=False)[0, 0].flatten()
    out["saliency"] = (sal >= torch.quantile(sal, 1 - cfg["saliency_top"])).float()
    return out


def main(cfg_path="configs/stage_b/q4_maps.yaml", resolutions=None):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    if resolutions:
        cfg["resolutions"] = [int(r) for r in resolutions.split(",")]
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"run_id": cfg["run_id"], "config": cfg, "git": git_hash(),
                                               "dirty": git_dirty()}, indent=1) + "\n")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EffModel(load_backbone(dev)).to(dev)
    if cfg.get("adapters"):                                          # Q10: trained adapters
        from .model import load_adapters
        load_adapters(model, cfg["adapters"])
    dtype = getattr(torch, cfg["dtype"])
    cheap_flops = dense_flops((CHEAP // 16) ** 2)
    for res in cfg["resolutions"]:
        g, N = res // 16, (res // 16) ** 2
        d_flops = dense_flops(N)
        ds = ImageList(cfg.get("image_list", "voc_val"), res)
        probe = torch.nn.Linear(384, CLASSES).to(dev)
        probe.load_state_dict(torch.load(Path(cfg["a0_runs"]) / f"probe_{res}.pt", map_location=dev))
        voc = [i for i, r in enumerate(ds.rows) if r[0] == "voc"][: cfg.get("max_images")]
        chunk_dir = out / f"chunks_{res}"
        chunk_dir.mkdir(exist_ok=True)
        for c0 in range(0, len(voc), cfg["chunk"]):
            ci = c0 // cfg["chunk"]
            if (chunk_dir / f"{ci:03d}.json").exists():
                continue
            try:
                os.close(os.open(chunk_dir / f"{ci:03d}.claim", os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
            rows = []
            for i in voc[c0:c0 + cfg["chunk"]]:
                label = mask_of(ds, i).to(dev).long()
                cls, cnt = torch.unique(label[(label != IGNORE) & (label != 0)], return_counts=True)
                if len(cls) == 0:
                    continue
                c = int(cls[cnt.argmax()])
                region = label == c
                x = ds[i][0][None].to(dev)
                maps = maps_for(model, x, region, g, cfg, dtype)
                valid = label != IGNORE

                def score(o):
                    pr = probe_logits(probe, o[:, N_PREFIX:], res).argmax(1)[0]
                    inr, outr = region & valid, (~region) & valid
                    return ((pr[inr] == label[inr]).float().mean().item(),
                            (pr[outr] == label[outr]).float().mean().item() if outr.any() else None)

                with torch.no_grad():
                    ref = model(x, dtype=dtype).final
                a_in, a_out = score(ref)
                rows.append({"image": i, "setting": "dense", "map": "none", "flops_ratio": 1.0, "acc_in": a_in,
                             "acc_out": a_out, "map_area": 1.0})
                for name in cfg["methods"][res]:
                    for src in (cfg["maps"] if name.split("_")[0] in ("itome", "gtome", "otome") else ["none"]):
                        m = from_name(name, g)
                        sts = []
                        with torch.no_grad():
                            o = model(x, structure=m, dtype=dtype, state_out=sts,
                                      interest=None if src == "none" else maps[src][None]).final
                        fl = image_flops(sts[0], ops_of(m), 0, N)["flops"]
                        extra = cheap_flops if src in ("click", "saliency", "click_loose", "click3", "pca") else 0.0
                        a_in, a_out = score(o)
                        rows.append({"image": i, "setting": name, "map": src, "flops_ratio": (fl + extra) / d_flops,
                                     "acc_in": a_in, "acc_out": a_out,
                                     "map_area": float(maps[src].mean()) if src != "none" else 1.0})
            (chunk_dir / f"{ci:03d}.json").write_text(json.dumps(rows))
            print(f"[{res}] chunk {ci}", flush=True)
        print(f"[{res}] done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
