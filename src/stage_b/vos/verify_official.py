"""R26 gate 1: the official Cutie weights (hkchengrex/Cutie release, Google Drive folder of docs/EVALUATION.md; MIT)
run unmodified (Cutie's own ResNet pixel encoder) on DAVIS 2017 val with the R10 protocol (480p, first-frame masks,
our J&F). Confirms the downloaded files are intact and behave as published before parts of them initialise R26.
Usage (ml-base:vos): python -m src.stage_b.vos.verify_official <weights.pth> [small|base] [out.json]"""
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


@torch.inference_mode()
def main(weights, size="small", out=None):
    import cutie.model.cutie as cm
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from cutie.inference.inference_core import InferenceCore
    from cutie.inference.utils.args_utils import get_dataset_cfg
    from src.stage_b.p2_davis import j_and_f
    with initialize_config_dir(version_base="1.3.2", config_dir=f"{CUTIE}/cutie/config", job_name="eval_config"):
        cfg = compose(config_name="eval_config", overrides=[f"model={size}", "dataset=d17-val"])
    with open_dict(cfg):
        cfg["weights"] = weights
    get_dataset_cfg(cfg)
    net = cm.CUTIE(cfg).cuda().eval()
    net.load_weights(torch.load(weights, map_location="cuda", weights_only=True))
    scores = []
    for seq in (DAVIS / "ImageSets/2017/val.txt").read_text().split():
        frames = sorted((DAVIS / "JPEGImages/480p" / seq).glob("*.jpg"))
        gt = [torch.from_numpy(np.array(Image.open(a))).long().cuda() for a in sorted((DAVIS / "Annotations/480p" / seq).glob("*.png"))]
        objects = sorted(set(torch.unique(gt[0]).tolist()) - {0})
        H, W = gt[0].shape
        tol = max(1, round(0.008 * (H ** 2 + W ** 2) ** 0.5))
        proc = InferenceCore(net, cfg=cfg)
        proc.max_internal_size = -1
        js, fs = {o: [] for o in objects}, {o: [] for o in objects}
        for t, fp in enumerate(frames):
            img = to_tensor(Image.open(fp).convert("RGB")).cuda().float()
            prob = proc.step(img, gt[0], objects=objects) if t == 0 else proc.step(img)
            pred = proc.output_prob_to_mask(prob)
            if 0 < t < len(gt):
                for o in objects:
                    j, f = j_and_f(pred == o, gt[t] == o, tol)
                    js[o].append(j)
                    fs[o].append(f)
        scores += [50 * (np.mean(js[o]) + np.mean(fs[o])) for o in objects]
        print(f"{seq}: running J&F {np.mean(scores):.2f}", flush=True)
    res = {"weights": weights, "size": size, "objects": len(scores), "jf": float(np.mean(scores))}
    print(res, flush=True)
    if out:
        Path(out).write_text(json.dumps(res) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
