"""Preprint figures from the result files (reported numbers only; no new measurement).
`flops`: J&F against encoder FLOPs, (a) held-out long videos (three heads: dense, ALGM 0.98, the method; 95% intervals
of the paired difference drawn on the method) and (b) DAVIS 2017 val at ViT-S with every baseline. Uses the official
scores from `r40_official.json` when present (`scorer=official`), else our scorer.
Writes `docs/paper/arxiv/figures/flops.pdf`.
`qualitative`: three LVOS videos (`fig_videos.txt`, run `24z_figures`): the method's tiers, replayed on CPU from the
run's own saved predictions (`r40_replay`; the reference path, which the timed run's decisions equal), with the method's
and ALGM 0.98's predicted masks; three frames per video, preferring annotated frames where one of the two has lost the object
(whichever happens more often in that video). Writes `docs/paper/arxiv/figures/qualitative.pdf`.
Usage (ml-base:vos, CPU): python -m src.stage_b.vos.paper_figures <flops [ours|official]|qualitative [preview.png]>"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Q = Path("reports/stage_b/queue")
OUT = Path("docs/paper/arxiv/figures")
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
OURS_C, ALGM_C, OTHER_C, PROT_C, CLOSED_C = "#c2410c", "#1d4ed8", "#6b7280", "#0f766e", "#7c3aed"


def _family(k):
    if k.startswith("titome"):
        return "Ours", OURS_C
    if k.startswith("ftome"):
        return "Closed-loop focus map", CLOSED_C
    if k.startswith("palgm"):
        return "Protected ALGM", PROT_C
    if k.startswith("algm"):
        return "ALGM", ALGM_C
    return "Other mergers", OTHER_C


def _label(k):
    p = k.split("_")
    return {"ailurus": f"AiluRus {p[1][1:]}", "algm": f"ALGM {p[1][1:]}", "tome": f"ToMe {p[1]}",
            "palgm": f"P-ALGM {p[1][1:]}", "ftome": f"closed-loop {p[1][1:]}", "pitome": f"PiToMe {p[1][1:]}",
            "expedite": f"Expedite {p[1][1:]}"}.get(p[0], k)


def flops(scorer="ours"):
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "serif"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
    det = json.loads((Q / "r40_final_fix_val_detail.json").read_text())
    d = det[next(iter(det))]["all"]
    pts = {"Dense": (1.0, d["dense_jf"], INK), "ALGM 0.98": (d["ref_flops"], d["ref_jf"], ALGM_C),
           "Ours": (d["flops"], d["jf"], OURS_C)}
    if scorer == "official" and (Q / "r40_official.json").exists():
        c = json.loads((Q / "r40_official.json").read_text())["claim"]
        pts = {"Dense": (1.0, c["all"]["dense_jf_official"], INK), "ALGM 0.98": (c["ref_flops"], c["all"]["ref_jf_official"], ALGM_C),
               "Ours": (c["flops"], c["all"]["jf_official"], OURS_C)}
        d = {"minus_ref": c["all"]["minus_ref_official"]}
    for name, (x, y, col) in pts.items():
        a.scatter([x], [y], s=110 if name == "Ours" else 30, color=col, zorder=3, marker="*" if name == "Ours" else "o",
                  edgecolor="white", linewidth=0.8)
        a.annotate(name, (x, y), textcoords="offset points", xytext=(7, -3 if name != "Dense" else 4), color=INK)
    m = d["minus_ref"]                                        # the paired difference against ALGM, as text (not a bar
    a.annotate(f"{m[0]:+.2f} over ALGM 0.98\n[{m[1]:+.2f}, {m[2]:+.2f}], paired",   # around the point: it is not the
               (pts["Ours"][0], pts["Ours"][1]), textcoords="offset points", xytext=(8, -26), color=OURS_C, fontsize=7)
    a.set_xlabel("Encoder FLOPs (share of dense)")
    a.set_ylabel("J&F" + (" (official)" if scorer == "official" else ""))
    a.set_title("(a) Held-out long videos (LVOS V2 valid, 3 heads)", fontsize=8, color=INK)
    a.set_xlim(0.3, 1.08)
    a.set_ylim(min(v[1] for v in pts.values()) - 0.5, max(v[1] for v in pts.values()) + 0.7)
    sh = json.loads((Q / "r40_final_fix_davis_all.json").read_text())["vits"]
    b.axhline(sh["dense"], color=GRID, lw=1, zorder=1)
    b.annotate("dense", (1.0, sh["dense"]), textcoords="offset points", xytext=(-22, 3), color=MUTED)
    seen = set()
    for k, v in list(sh["settings"].items()) + [("titome", sh["redesign"])]:
        fam, col = _family(k)
        b.scatter([v["flops"]], [v["jf"]], s=110 if fam == "Ours" else 22, color=col, marker="*" if fam == "Ours" else "o",
                  edgecolor="white", linewidth=0.6, zorder=3, label=None if fam in seen else fam)
        seen.add(fam)
        if fam == "Ours":
            b.annotate("Ours", (v["flops"], v["jf"]), textcoords="offset points", xytext=(5, 3), color=INK)
    b.set_xlabel("Encoder FLOPs (share of dense)")
    b.set_title("(b) Short clips (DAVIS 2017 val, ViT-S)", fontsize=8, color=INK)
    b.legend(frameon=False, fontsize=6.5, loc="lower right")
    for ax in (a, b):
        ax.grid(color=GRID, lw=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "flops.pdf")
    print(OUT / "flops.pdf")


def qualitative():
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode, resize, to_tensor
    from .eval_lvos import LVOS2, SIZE
    from .r28_compare import V
    from .r40_compare import S_FINAL
    from .r40_replay import replay
    root = LVOS2 / "valid"
    vids = (Q / "fig_videos.txt").read_text().split()
    fig, axes = plt.subplots(len(vids), 3, figsize=(6.8, 1.35 * len(vids) + 0.2), constrained_layout=True)
    for r, v in enumerate(vids):
        mine = np.load(V / "eval_fig_r40" / "masks_small" / S_FINAL / f"{v}.npz")
        base = np.load(V / "eval_fig_base" / "masks_small" / "algm_t0.98" / f"{v}.npz")
        pm, pb = dict(zip(mine["t"].tolist(), mine["small"])), dict(zip(base["t"].tolist(), base["small"]))
        frames = sorted((root / "JPEGImages" / v).glob("*.jpg"))
        annos = {q.stem for q in (root / "Annotations" / v).glob("*.png")}
        ts = [t for t in sorted(pm) if frames[t].stem in annos and t in pb]
        lost_b = [t for t in ts if not pb[t].any() and pm[t].any()]       # ALGM lost the object, the method kept it
        lost_m = [t for t in ts if not pm[t].any() and pb[t].any()]       # and the reverse
        lost = lost_b if len(lost_b) >= len(lost_m) else lost_m           # show the failure that dominates the video
        pool = lost if len(lost) >= 3 else ts
        pick = [pool[int(i)] for i in np.linspace(len(pool) * 0.15, len(pool) * 0.85, 3)]
        res = replay(root, v, preds=pm, keep=pick)["tiers"]
        for c, t in enumerate(pick):
            img = resize(to_tensor(Image.open(frames[t]).convert("RGB")), SIZE, InterpolationMode.BILINEAR, antialias=True)
            img = img.permute(1, 2, 0).numpy()
            ax = axes[r, c]
            ax.imshow(img)
            tier, (lh, lw, h, w) = res[t]
            if tier is not None:
                up = np.kron(tier, np.ones((16, 16)))[lh:lh + h, lw:lw + w]
                over = np.zeros((h, w, 4))
                over[up >= 1] = [0.85, 0.15, 0.1, 0.38]
                over[(up > 0) & (up < 1)] = [0.95, 0.6, 0.1, 0.32]
                ax.imshow(over)
            else:
                ax.text(0.02, 0.95, "fallback frame", transform=ax.transAxes, color="white", fontsize=6, va="top")
            ax.contour(pm[t] > 0, levels=[0.5], colors="white", linewidths=0.8)
            if pb[t].any():
                ax.contour(pb[t] > 0, levels=[0.5], colors="#38bdf8", linewidths=0.8, linestyles="dashed")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{v}, frame {t}", fontsize=6, color=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qualitative.pdf", dpi=200)
    if len(sys.argv) > 2:                                     # optional PNG preview path, for checking the layout
        fig.savefig(sys.argv[2], dpi=110)
    print(OUT / "qualitative.pdf")


if __name__ == "__main__":
    {"flops": lambda *a: flops(*a), "qualitative": lambda *a: qualitative()}[sys.argv[1]](*sys.argv[2:])
