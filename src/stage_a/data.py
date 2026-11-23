"""Stage A data: fixed image lists and resolution-parametric preprocessing (brief section 4).

Reuses the ImageNet-100 extraction and VOC layout from Core-1 (`core1/core1/data`); only the resolution changes.
Preprocessing: shorter side to the target size (bilinear, the DINOv3 image processor's resample), center crop,
DINOv3 mean and std. VOC labels get the same geometry with nearest interpolation.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # same values as core1/core1/backbone/load.py
WORK = Path("/work")
IN100 = WORK / "data/imagenet100"
VOC = WORK / "data/voc2012/VOCdevkit/VOC2012"
VOC_VOID = 255
LISTS = Path(__file__).resolve().parents[2] / "configs/stage_a/lists"


def image_transform(size: int):
    return T.Compose([T.Resize(size, interpolation=InterpolationMode.BILINEAR), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def label_transform(size: int):
    return T.Compose([T.Resize(size, interpolation=InterpolationMode.NEAREST), T.CenterCrop(size)])


class ImageList(Dataset):
    """Rows of a list file: `path<TAB>label`. Paths are relative to the dataset root; VOC rows carry no class label."""

    def __init__(self, name: str, size: int, with_mask: bool = False):
        self.rows = [l.split("\t") for l in (LISTS / f"{name}.txt").read_text().split("\n") if l]
        self.size, self.with_mask = size, with_mask
        self.tf, self.ltf = image_transform(size), label_transform(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        src, rel, lab = self.rows[i]
        root = IN100 if src == "in100" else VOC
        x = self.tf(Image.open(root / rel).convert("RGB"))
        if not self.with_mask:
            return x, int(lab), i
        stem = Path(rel).stem
        m = np.asarray(self.ltf(Image.open(VOC / "SegmentationClass" / f"{stem}.png")), dtype=np.uint8)
        return x, torch.from_numpy(m.copy()), i


def _in100_rows(split: str):
    man = json.loads((IN100 / f"{split}_manifest.json").read_text())
    labels = sorted({m["label"] for m in man})
    idx = {l: k for k, l in enumerate(labels)}  # ImageNet label -> 0..99
    return [("in100", m["path"], str(idx[m["label"]])) for m in man]


def _voc_rows(split: str):
    names = (VOC / "ImageSets/Segmentation" / f"{split}.txt").read_text().split()
    return [("voc", f"JPEGImages/{n}.jpg", "-1") for n in names]


def build_lists(seed: int = 0, bank_per_class: int = 100, n_calib: int = 256) -> dict:
    """Writes the fixed Stage A lists and returns their sha256. Deterministic for a seed."""
    rng = random.Random(seed)
    val, train = _in100_rows("val"), _in100_rows("train")
    voc_train, voc_val = _voc_rows("train"), _voc_rows("val")
    by_class = {}
    for r in sorted(train):
        by_class.setdefault(r[2], []).append(r)
    bank = [r for c in sorted(by_class, key=int) for r in sorted(rng.sample(by_class[c], bank_per_class))]
    calib = sorted(rng.sample(sorted(val), n_calib)) + sorted(rng.sample(voc_val, n_calib))
    lists = {"knn_bank": bank, "in100_val": sorted(val), "voc_train": voc_train, "voc_val": voc_val, "calibration": calib}
    LISTS.mkdir(parents=True, exist_ok=True)
    sums = {}
    for name, rows in lists.items():
        text = "".join("\t".join(r) + "\n" for r in rows)
        (LISTS / f"{name}.txt").write_text(text)
        sums[name] = {"n": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    (LISTS / "checksums.json").write_text(json.dumps({"seed": seed, "lists": sums}, indent=1) + "\n")
    return sums


if __name__ == "__main__":
    print(json.dumps(build_lists(), indent=1))
