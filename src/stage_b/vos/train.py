"""R10: train Cutie with the DINOv3 pixel encoder (dino_encoder.py) using Cutie's own training script and configs,
single GPU. Deviations from Cutie's recipe (budget, data available here): no static-image pre-training; main training
on DAVIS 2017 train and MOSE train (YouTube-VOS has no reachable mirror); `iters` iterations with Cutie's step schedule
scaled; batch size `batch`. Usage (in ml-base:vos): python -m src.stage_b.vos.train <exp_id> [iters] [batch] [dtem tag]
R21 (`dtem tag` given): start from the R10 weights, freeze everything except DTEM's metric layers (soft merging, ToMe
r64 schedule, in the encoder), learning rate 1e-3 with no backbone ratio; at the end the metric layers are written to
/work/stage_b/dtem/<tag>.pt.
R26 (keyword options after the positional ones): `init=<checkpoint>` starts from that checkpoint (e.g. the official
cutie-small-mega weights without the two stride-16 projections, `prepare_r26_init`); `only=proj` trains only those two
projections (`pix_feat_proj`, `key_proj.pix_feat_proj`), everything else frozen.
R31 (`compress=<setting>[~fallback]`): our method in the encoder during training, every other part as R10/R26. Each
frame's focus map is the previous frame's ground truth of the sampled objects (the first frame: its own), shifted by
up to one patch at random (uniform in [-16, 16] pixels per axis) and emptied with probability 0.15 (then the fallback
runs, as in a lost-map frame), so the head learns compressed features and imperfect maps (`maps_from_gt`).
R-25 protocol (`seed=<n>`): Cutie's `seed` (default 14159265, used by R26 and R31c); heads h2 and h3 repeat R31c's
recipe with seeds 2 and 3.
Resume (`resume=<checkpoint>`, R40 plan section L: the GPU paused at a checkpoint): Cutie's `checkpoint` (weights,
optimizer, scheduler, iteration); never together with `init=` (Cutie loads `weights` after the checkpoint).
Usage: python -m src.stage_b.vos.train <exp_id> [iters] [batch] [dtem tag|-] [init=...] [only=proj] [compress=...] [seed=...] [resume=...]"""
import os
import sys

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29571")
CUTIE = "/work/third_party/Cutie"
sys.path.insert(0, CUTIE)


R10 = "/work/stage_b/vos/r10_vits/r10_vits_main_training_last.pth"


PROJ = ("pix_feat_proj.", "key_proj.pix_feat_proj.")


def prepare_r26_init(src="/work/data/downloads/cutie_weights/cutie-small-mega.pth", dst="/work/stage_b/vos/r26_init.pth"):
    """Official Cutie weights minus what does not fit the DINOv3 encoder: the two 1x1 projections from the stride-16
    features (weights and biases; 256 input channels there, the ViT width here) and ResNet layer3 (replaced by the ViT)."""
    import torch
    sd = torch.load(src, map_location="cpu", weights_only=True)
    keep = {k: v for k, v in sd.items() if not k.startswith(PROJ) and not k.startswith("pixel_encoder.layer3.")}
    print(f"kept {len(keep)} of {len(sd)} tensors; dropped {sorted(set(sd) - set(keep))[:6]} ...", flush=True)
    torch.save(keep, dst)


SHIFT, EMPTY = 16, 0.15


def maps_from_gt(cls_gt, gen=None):
    """cls_gt (B, T, 1, H, W) object indices -> focus maps (B * T, H, W) bool: frame t gets frame t - 1's objects
    (frame 0 its own), shifted by a random offset in [-SHIFT, SHIFT] pixels per axis (zero fill), empty with
    probability EMPTY."""
    import torch
    g = cls_gt[:, :, 0] > 0
    prev = torch.cat([g[:, :1], g[:, :-1]], 1)
    B, T, H, W = prev.shape
    out = torch.zeros_like(prev)
    for b in range(B):
        for t in range(T):
            if float(torch.rand(1, generator=gen)) < EMPTY:
                continue
            dy, dx = (int(v) for v in torch.randint(-SHIFT, SHIFT + 1, (2,), generator=gen))
            ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
            xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
            out[b, t, yd, xd] = prev[b, t, ys, xs]
    return out.flatten(0, 1)


def main(exp_id, iters="30000", batch="8", dtem=None, *opts):
    iters, batch = int(iters), int(batch)
    dtem = None if dtem in (None, "-") else dtem
    kw = dict(o.split("=", 1) for o in opts)
    if kw.get("only") == "proj":
        import cutie.model.train_wrapper as tw
        orig0 = tw.CutieTrainWrapper.__init__

        def init_proj(self, *a, **k):                                   # before the trainer builds its optimizer
            orig0(self, *a, **k)
            for n, prm in self.named_parameters():
                prm.requires_grad_(n.startswith(PROJ))
        tw.CutieTrainWrapper.__init__ = init_proj
    if dtem:
        os.environ["VOS_DTEM_TRAIN"] = dtem
        import cutie.model.train_wrapper as tw
        orig = tw.CutieTrainWrapper.__init__

        def init(self, *a, **k):                                        # before the trainer builds its optimizer
            orig(self, *a, **k)
            for n, prm in self.named_parameters():
                prm.requires_grad_(n.startswith("pixel_encoder.dtem_heads."))
        tw.CutieTrainWrapper.__init__ = init
    if "compress" in kw:
        os.environ["VOS_COMPRESS_TRAIN"] = kw["compress"]
        import cutie.model.train_wrapper as tw
        orig_fwd = tw.CutieTrainWrapper.forward

        def forward(self, data):                                        # maps for this batch, then Cutie's forward
            self.pixel_encoder.train_maps = maps_from_gt(data["cls_gt"]).to(data["rgb"].device)
            return orig_fwd(self, data)
        tw.CutieTrainWrapper.forward = forward
    import cutie.model.big_modules as bm
    import cutie.model.cutie as cm

    from src.stage_b.vos.dino_encoder import VIT_WIDTH, DinoPixelEncoder
    bm.PixelEncoder = DinoPixelEncoder
    cm.PixelEncoder = DinoPixelEncoder
    out = f"/work/stage_b/vos/{exp_id}"
    os.makedirs(out, exist_ok=True)
    os.chdir(CUTIE)                                   # Cutie's configs use paths relative to its repository
    sys.argv = ["train.py", f"exp_id={exp_id}", "model=small", f"model.pixel_encoder.ms_dims=[{VIT_WIDTH},128,64]",
                "pre_training.enabled=False", "data=with-mose", "data.main_training.datasets=[DAVIS,MOSE]",
                f"data.main_training.num_iterations={iters}",
                f"data.main_training.lr_schedule_steps=[{int(iters * 0.8)},{int(iters * 0.92)}]",
                f"main_training.batch_size={batch}", "num_workers=8",
                "data.vos_datasets.base=/work/data",
                "data.vos_datasets.DAVIS.image_directory=davis/DAVIS/JPEGImages/480p",
                "data.vos_datasets.DAVIS.mask_directory=davis/DAVIS/Annotations/480p",
                "data.vos_datasets.MOSE.image_directory=mose/MOSE_release/train/JPEGImages",
                "data.vos_datasets.MOSE.mask_directory=mose/MOSE_release/train/Annotations",
                f"hydra.run.dir={out}"]
    if dtem:
        sys.argv += [f"weights={R10}", "main_training.learning_rate=1e-3", "main_training.backbone_lr_ratio=1.0"]
    assert not ("init" in kw and "resume" in kw), "resume without init= (Cutie would load the init weights over it)"
    if "init" in kw:
        sys.argv += [f"weights={kw['init']}"]
    if "resume" in kw:
        sys.argv += [f"checkpoint={kw['resume']}"]
    if "seed" in kw:                                  # R-25 protocol heads: Cutie's seed (initialisation, data order, augmentation)
        sys.argv += [f"seed={int(kw['seed'])}"]
    from cutie import train as T
    T.train()
    if dtem:
        import torch
        net = torch.load(f"{out}/{exp_id}_main_training_last.pth", map_location="cpu")
        pre = "pixel_encoder.dtem_heads."
        heads = {k[len(pre):]: v for k, v in net.items() if k.startswith(pre)}
        assert heads, "no DTEM metric layers in the checkpoint"
        os.makedirs("/work/stage_b/dtem", exist_ok=True)
        torch.save(heads, f"/work/stage_b/dtem/{dtem}.pt")


if __name__ == "__main__":
    if sys.argv[1:2] == ["prepare_r26_init"]:
        prepare_r26_init(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
