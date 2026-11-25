# Heads

Cutie (small) with the frozen DINOv3 ViT as its stride-16 pixel encoder; the ViT is frozen and dense during training.
Data: DAVIS 2017 train and MOSE train with Cutie's training script; we add no static-image or YouTube-VOS training,
but every head starts from the official `cutie-small-mega` weights, which their authors trained on static images, DAVIS,
YouTube-VOS, MOSE, BURST and OVIS, so the heads inherit that training.

First prepare the initial weights once (reads `cutie-small-mega.pth` from `/work/data/downloads/cutie_weights/`, keeps
every non-ViT tensor except the two new stride-16 projections):

```bash
IMAGE=ml-base:vos bash scripts/run.sh python -c "from src.stage_b.vos.train import prepare_r26_init; prepare_r26_init()"
```

| Head | Backbone | Recipe (`python -m src.stage_b.vos.train <id> <iterations> 2 - ...`) |
| --- | --- | --- |
| r26 | ViT-S/16 | `r26_s1 5000 2 - init=/work/stage_b/vos/r26_init.pth only=proj` (Cutie small weights, `prepare_r26_init`; only the two new stride-16 projections train), then `r26 40000 2 - init=<r26_s1 weights>` |
| r31c, r38_h2, r38_h3 | ViT-S/16 | fine-tuned from r26, 10,000 iterations, Cutie seeds default, 2, 3 (`seed=`) |
| r25 | ViT-B/16 | as r26 from the same Cutie small weights (`r25_s1`, then `r25`), `VOS_VIT=facebook/dinov3-vitb16-pretrain-lvd1689m` |
| r41_b1, r41_b2, r41_b3 | ViT-B/16 | fine-tuned from r25, 10,000 iterations, seeds default, 2, 3 |
| r34 | ViT-L/16 | as r26 (`r34_s1`, then `r34`), `VOS_VIT=facebook/dinov3-vitl16-pretrain-lvd1689m` |
