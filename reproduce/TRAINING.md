# Heads

Cutie (small) with the frozen DINOv3 ViT as its stride-16 pixel encoder; the ViT is frozen and dense during training.
Data: DAVIS 2017 train and MOSE train, Cutie's training script, no static-image pre-training, no YouTube-VOS.

| Head | Backbone | Recipe (`python -m src.stage_b.vos.train <id> <iterations> 2 - ...`) |
| --- | --- | --- |
| r26 | ViT-S/16 | `r26_s1 5000 2 - init=/work/stage_b/vos/r26_init.pth only=proj` (Cutie small weights, `prepare_r26_init`; only the two new stride-16 projections train), then `r26 40000 2 - init=<r26_s1 weights>` |
| r31c, r38_h2, r38_h3 | ViT-S/16 | fine-tuned from r26, 10,000 iterations, Cutie seeds default, 2, 3 (`seed=`) |
| r25 | ViT-B/16 | as r26 from the same Cutie small weights (`r25_s1`, then `r25`), `VOS_VIT=facebook/dinov3-vitb16-pretrain-lvd1689m` |
| r41_b1, r41_b2, r41_b3 | ViT-B/16 | fine-tuned from r25, 10,000 iterations, seeds default, 2, 3 |
| r34 | ViT-L/16 | as r26 (`r34_s1`, then `r34`), `VOS_VIT=facebook/dinov3-vitl16-pretrain-lvd1689m` |
