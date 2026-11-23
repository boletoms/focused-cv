"""ADE20K for Q9 breadth (queue; docs/records/STAGE_B_ABLATION_QUEUE.md): a dataset (same transform as the Stage A
lists: resize the short side, centre crop), a linear probe trained on dense features (150 classes; ADE label 0 =
"other" is ignored), and the interest test: regions of any kind (sky, grass, walls, objects) as the region of interest.
Usage: python -m src.stage_b.ade probe <res> [training images] | eval [config]"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset

from src.ablate.runinfo import git_dirty, git_hash
from src.stage_a.data import image_transform, label_transform
from src.stage_a.evaluate import probe_logits, train_probe
from src.stage_a.model import N_PREFIX, load_backbone

from .flops import dense_flops, image_flops
from .methods import ops_of
from .model import EffModel
from .p4_maps import CHEAP, maps_for
from .timing import from_name

ROOT = Path("/work/data/ade20k/ADEChallengeData2016")
PROBES = Path("/work/stage_b/ade_probes")
CLASSES, IGNORE = 150, 255
CHEAP_MAPS = ("click", "saliency", "click_loose", "click3", "pca")


class ADE(Dataset):
    def __init__(self, split, size, limit=None, seed=0, exclude=None):
        """exclude: (limit, seed) of another sample of the same split to leave out (R2: tuning images disjoint from
        the probe's training images)."""
        imgs = sorted((ROOT / "images" / split).glob("*.jpg"))
        if exclude:
            drop = set(random.Random(exclude[1]).sample(imgs, exclude[0]))
            imgs = [p for p in imgs if p not in drop]
        if limit:
            imgs = sorted(random.Random(seed).sample(imgs, limit))
        self.imgs, self.split = imgs, split
        self.tf, self.ltf = image_transform(size), label_transform(size)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        x = self.tf(Image.open(p).convert("RGB"))
        a = np.asarray(self.ltf(Image.open(ROOT / "annotations" / self.split / (p.stem + ".png"))), dtype=np.int64)
        lab = np.where(a == 0, IGNORE, a - 1)                           # 0 = other -> ignore; classes 0..149
        return x, torch.from_numpy(lab), i


@torch.no_grad()
def features(model, ds, dev, dtype, batch=8):
    fs, ms = [], []
    for s in range(0, len(ds), batch):
        items = [ds[i] for i in range(s, min(len(ds), s + batch))]
        x = torch.stack([t[0] for t in items]).to(dev)
        fs.append(model(x, dtype=dtype).final[:, N_PREFIX:].to(torch.bfloat16))
        ms.append(torch.stack([t[1] for t in items]).to(torch.uint8).to(dev))     # 0..149 and 255 fit in uint8
    return torch.cat(fs), torch.cat(ms)


def probe(res, limit="2000"):
    res, limit = int(res), int(limit)
    dev = "cuda"
    model = EffModel(load_backbone(dev)).to(dev)
    ds = ADE("training", res, limit=limit, seed=0)                     # 896: 1,000 images (GPU memory)
    f, m = features(model, ds, dev, torch.bfloat16)
    cfg = {"lr": 0.001, "weight_decay": 0.0001, "epochs": 20, "batch": 8 if res <= 448 else 2,   # 896: logits memory
           "classes": CLASSES, "ignore": IGNORE}
    pr, losses = train_probe(f, m, res, cfg, 0)
    PROBES.mkdir(parents=True, exist_ok=True)
    torch.save(pr.state_dict(), PROBES / f"probe_{res}.pt")
    print(f"[{res}] probe loss {losses[0]:.3f} -> {losses[-1]:.3f}", flush=True)


def evaluate(cfg_path="configs/stage_b/q9_ade.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"run_id": cfg["run_id"], "config": cfg, "git": git_hash(),
                                               "dirty": git_dirty()}, indent=1) + "\n")
    dev = "cuda"
    model = EffModel(load_backbone(dev)).to(dev)
    dtype = torch.bfloat16
    cheap_flops = dense_flops((CHEAP // 16) ** 2)
    for res in cfg["resolutions"]:
        g, N = res // 16, (res // 16) ** 2
        d_flops = dense_flops(N)
        if cfg.get("split", "validation") == "training":       # R2 tuning: training images the probe never saw
            ds = ADE("training", res, limit=cfg["max_images"], seed=cfg.get("sample_seed", 2),
                     exclude=(2000 if res <= 448 else 1000, 0))
        else:
            ds = ADE("validation", res, limit=cfg.get("max_images"))
        pr = torch.nn.Linear(384, CLASSES).to(dev)
        pr.load_state_dict(torch.load(PROBES / f"probe_{res}.pt", map_location=dev))
        if "det" in cfg["maps"]:
            det = json.loads(Path(f"/work/stage_b/r11_boxes/boxes_{res}.json").read_text())
        chunk_dir = out / f"chunks_{res}"
        chunk_dir.mkdir(exist_ok=True)
        for c0 in range(0, len(ds), cfg["chunk"]):
            ci = c0 // cfg["chunk"]
            if (chunk_dir / f"{ci:03d}.json").exists():
                continue
            try:
                os.close(os.open(chunk_dir / f"{ci:03d}.claim", os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
            rows = []
            for i in range(c0, min(len(ds), c0 + cfg["chunk"])):
                x, label, _ = ds[i]
                x, label = x[None].to(dev), label.to(dev)
                valid = label != IGNORE
                cls, cnt = torch.unique(label[valid], return_counts=True)
                if len(cls) < 2:
                    continue
                regions = {"largest": int(cls[cnt.argmax()]), "random": int(random.Random(i).choice(cls.tolist()))}
                touch = lambda m: F.max_pool2d(m.float()[None, None], 16)[0, 0].flatten()

                def score(o, c):
                    p = probe_logits(pr, o[:, N_PREFIX:], res).argmax(1)[0]
                    inr, outr = (label == c) & valid, (label != c) & valid
                    lv, pv = label[valid], p[valid]               # full image: overall accuracy, per-class I and U
                    score.full = {"acc_all": (pv == lv).float().mean().item(),
                                  "iu": {int(k): [int(((pv == k) & (lv == k)).sum()), int(((pv == k) | (lv == k)).sum())]
                                         for k in torch.unique(torch.cat([lv, pv])).tolist()}}
                    return ((p[inr] == label[inr]).float().mean().item(),
                            (p[outr] == label[outr]).float().mean().item() if outr.any() else None)

                with torch.no_grad():
                    ref = model(x, dtype=dtype).final
                regions = {k: v for k, v in regions.items() if k in cfg.get("regions", regions)}
                for kind, c in regions.items():
                    a_in, a_out = score(ref, c)
                    dense_row = len(rows)
                    rows.append({"image": i, "region": kind, "class": c, "setting": "dense", "map": "none", "flops_ratio": 1.0,
                                 "acc_in": a_in, "acc_out": a_out, "area": float((label == c).float().mean()), **score.full})
                    reg = label == c
                    if set(cfg["maps"]) <= {"gt", "box", "det"}:
                        ys, xs = reg.nonzero(as_tuple=True)
                        bx = torch.zeros_like(reg)
                        bx[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                        maps = {"gt": touch(reg), "box": touch(bx)}
                        if "det" in cfg["maps"] and kind == "random":   # R11: text-grounded detector boxes
                            assert str(i) not in det or det[str(i)]["class"] == c
                            dm = torch.zeros_like(reg)
                            for x0, y0, x1, y1 in det[str(i)]["boxes"] if str(i) in det else []:
                                dm[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = True
                            maps["det"] = touch(dm)
                    else:                                       # N6: click maps from a cheap pass (cost added)
                        maps = maps_for(model, x, reg, g, cfg, dtype)
                    for name in cfg["methods"][res]:
                        for mp in (cfg["maps"] if name.split("_")[0] in ("itome", "ftome") else ["none"]):
                            if mp not in maps:                      # det exists for the random region only
                                continue
                            if mp == "det" and not maps["det"].any():   # no detection: dense fallback
                                rows.append({**rows[dense_row], "setting": name, "map": "det", "map_area": 0.0,
                                             "fallback": True})
                                continue
                            m = from_name(name, g)
                            sts = []
                            with torch.no_grad():
                                o = model(x, structure=m, dtype=dtype, state_out=sts,
                                          interest=None if mp == "none" else maps[mp][None]).final
                            a_in, a_out = score(o, c)
                            extra = cheap_flops if mp in CHEAP_MAPS else 0.0
                            rows.append({"image": i, "region": kind, "class": c, "setting": name, "map": mp,
                                         "flops_ratio": (image_flops(sts[0], ops_of(m), 0, N)["flops"] + extra) / d_flops,
                                         "acc_in": a_in, "acc_out": a_out, "area": float(reg.float().mean()),
                                         "map_area": 1.0 if mp == "none" else float(maps[mp].mean()), **score.full})
            (chunk_dir / f"{ci:03d}.json").write_text(json.dumps(rows))
            print(f"[{res}] chunk {ci}", flush=True)
        print(f"[{res}] done", flush=True)


if __name__ == "__main__":
    {"probe": probe, "eval": evaluate}[sys.argv[1]](*sys.argv[2:])
