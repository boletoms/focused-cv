"""R40 plan section O (official metrics; reported): prediction masks saved by the evaluators (`VOS_SAVE_MASKS=1`,
`vos/masks.py`) scored by the official toolkits, used unchanged from /work/third_party/official_eval:
- LVOS (LingyiHongfd/lvos-evaluation @ 25cb02d, semi-supervised; each object scored on every fifth frame of its range
  in the meta file, without the range's first and last frame, as the LVOS challenge): LVOS V2 valid (140 videos; its
  frames and annotations equal LVOS V1 val's for the 50 V1 videos, byte for byte), the meta file restricted to the
  videos with saved masks; seen / unseen from the toolkit's `unseen_videos.txt`.
- DAVIS 2017 val (davisvideochallenge/davis2017-evaluation @ ac7c43f, semi-supervised; frames 1 to the second-to-last).
Per object the toolkits' per-object means of J and F, written to `<run dir>/official/<setting>.json` as
{"sequences": {sequence: {object id: [J, F]}}, "unseen": [...]}; the toolkit's own averages are not used (comparisons
pair objects, `r40_compare official`).
Usage (image ml-base:vos-eval, CPU): python -m src.stage_b.vos.official_eval <lvos|davis> <run dir under /work>
[workers] [setting ...] (default: every setting with saved masks)."""
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

TOOLS = Path("/work/third_party/official_eval")
LVOS_V2_VALID = Path("/work/data/lvos_v2/valid")
LVOS_META = Path("/work/data/lvos_v2/meta_jsons/valid_meta.json")
DAVIS_ROOT = Path("/work/data/davis/DAVIS")
_ev = None


def _lvos_view(run, setting, seqs):
    """A root the toolkit reads (`<root>/valid_meta.json`, `Annotations`, `JPEGImages`), restricted to `seqs`."""
    root = Path(run) / "official" / "lvos_root"
    root.mkdir(parents=True, exist_ok=True)
    for d in ("Annotations", "JPEGImages"):
        if not (root / d).exists():
            os.symlink(LVOS_V2_VALID / d, root / d)
    meta = json.loads(LVOS_META.read_text())
    meta["videos"] = {s: v for s, v in meta["videos"].items() if s in seqs}
    (root / "valid_meta.json").write_text(json.dumps(meta))
    return root


def _lvos_init(root):
    global _ev
    sys.path.insert(0, str(TOOLS / "lvos-evaluation"))
    os.chdir(TOOLS / "lvos-evaluation")                   # the toolkit reads ./unseen_videos.txt
    from lvos.evaluation import LVOSEvaluation
    _ev = LVOSEvaluation(lvos_root=str(root), task="semi-supervised", gt_set="valid")


def _lvos_seq(a):
    from lvos import utils
    from lvos.results import Results
    seq, res_dir = a
    j, f = _ev._evaluate_semisupervised(_ev.dataset.get_sequence(seq), Results(root_dir=res_dir), None, ("J", "F"))
    return seq, {o: [float(utils.db_statistics(j[o])[0]), float(utils.db_statistics(f[o])[0])] for o in j}


def lvos(run, workers=8, *settings):
    run = Path(run)
    for st in settings or sorted(p.name for p in (run / "masks").iterdir()):
        res = run / "masks" / st
        seqs = sorted(p.name for p in res.iterdir())
        root = _lvos_view(run, st, set(seqs))
        with get_context("spawn").Pool(int(workers), initializer=_lvos_init, initargs=(root,)) as pool:
            out = dict(pool.imap_unordered(_lvos_seq, [(s, str(res)) for s in seqs]))
        unseen = set((TOOLS / "lvos-evaluation" / "unseen_videos.txt").read_text().split())
        _write(run, st, out, sorted(s for s in seqs if s in unseen))


def _davis_one(a):
    run, st = a
    sys.path.insert(0, str(TOOLS / "davis2017-evaluation"))
    from davis2017.evaluation import DAVISEvaluation
    ev = DAVISEvaluation(davis_root=str(DAVIS_ROOT), task="semi-supervised", gt_set="val")
    m = ev.evaluate(str(Path(run) / "masks" / st))
    out = {}
    for key, jm in m["J"]["M_per_object"].items():
        seq, o = key.rsplit("_", 1)
        out.setdefault(seq, {})[o] = [float(jm), float(m["F"]["M_per_object"][key])]
    return st, out


def davis(run, workers=8, *settings):
    run = Path(run)
    sts = settings or sorted(p.name for p in (run / "masks").iterdir())
    with get_context("spawn").Pool(min(int(workers), len(sts))) as pool:
        for st, out in pool.imap_unordered(_davis_one, [(str(run), s) for s in sts]):
            _write(run, st, out, [])


def _write(run, st, out, unseen):
    d = run / "official"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{st}.json").write_text(json.dumps({"sequences": out, "unseen": unseen}))
    jf = [50 * (j + f) for s in out.values() for j, f in s.values()]
    print(f"{run.name} {st}: {len(out)} videos, {len(jf)} objects, official J&F {np.mean(jf):.2f}", flush=True)


if __name__ == "__main__":
    {"lvos": lvos, "davis": davis}[sys.argv[1]](*sys.argv[2:])
