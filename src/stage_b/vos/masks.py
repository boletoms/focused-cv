"""R40 plan section O (official metrics): prediction masks saved as indexed PNGs (the annotation's palette) for the
official DAVIS 2017 and LVOS toolkits. With VOS_SAVE_MASKS=1 the evaluators write `<out>/masks/<setting>/<sequence>/
<frame>.png` on every frame with an annotation (LVOS) or every frame (DAVIS); reads only, changes no prediction."""
import os
from pathlib import Path

import numpy as np
from PIL import Image

ON = os.environ.get("VOS_SAVE_MASKS") == "1"
SMALL = os.environ.get("VOS_SAVE_SMALL") == "1"     # every frame's prediction at the evaluation size (figures)


def palette(anno_path):
    return Image.open(anno_path).getpalette()


def save(out, setting, seq, frame, pred, pal):
    """pred: (H, W) integer tensor of object ids (as the scorer's), frame: the frame's file stem."""
    d = Path(out) / "masks" / setting / seq
    d.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(pred.cpu().numpy().astype(np.uint8), mode="P")
    if pal:
        im.putpalette(pal)
    im.save(d / f"{frame}.png")


def save_small(out, setting, seq, frames):
    """frames: [(t, (h, w) uint8 array)], every processed frame's prediction at the evaluation size (the map's input),
    for replaying the focus map's tiers offline (`vos/r40_replay.py`, figures). One file per video and setting."""
    d = Path(out) / "masks_small" / setting
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{seq}.npz", t=np.array([t for t, _ in frames]), small=np.stack([m for _, m in frames]))
