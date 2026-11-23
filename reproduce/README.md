# Reproducing the preprint's tables

Every command runs in Docker through `scripts/run.sh` (`IMAGE=ml-base:vos`); `W=/work/stage_b/vos`. Heads:
`TRAINING.md`. Setting names: `SETTINGS.md`. `GV` selects the timed implementation:

```bash
GV="VOS_GRAPH=2 TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor VOS_GRAPH_COMPILE=segment VOS_GRAPH_KERNEL=varlen VOS_GRAPH_OVERLAP=1 R40_INDUCTOR_OPTS=emulate_precision_casts=1"
S="titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1"
```

Several workers can share one GPU (`MPS=1`, an MPS daemon on the host; `scripts/lvos_launch_env.sh <weights> <out>
<settings> <split> <workers> <log tag> [VAR=value ...]` starts them); each worker takes the next unclaimed video.

## Table 1 and Figure 2a: held-out long videos (ViT-S, three heads)

```bash
for h in r31c r38_h2 r38_h3; do
  WH=$W/$h/${h}_main_training_last.pth
  bash scripts/lvos_launch_env.sh $WH $W/eval_om_${h}_base algm_t0.98 v2_valid 4 base_$h VOS_SAVE_MASKS=1
  bash scripts/lvos_launch_env.sh $WH $W/eval_om_${h}_r40 "$S" v2_valid 4 r40_$h VOS_SAVE_MASKS=1 VOS_SKIP_DENSE=1 $GV
  for r in base r40; do
    IMAGE=ml-base:vos bash scripts/run.sh env CUDA_VISIBLE_DEVICES= python -m src.stage_b.vos.official_eval lvos $W/eval_om_${h}_$r
  done
done
IMAGE=ml-base:vos bash scripts/run.sh env CUDA_VISIBLE_DEVICES= python -m src.stage_b.vos.r40_official claim
```

The pre-registered rule (superior if the lower 95% bound of the head-averaged paired difference against ALGM 0.98 is
above 0 at lower FLOPs, non-inferior if above -3.0) is evaluated on our scorer; the official scores are reported beside
it.

## Tables 3 and 4, Figure 2b: short clips (DAVIS 2017 val)

```bash
# ViT-S, head r26: every baseline (eager), then the method on the timed implementation
IMAGE=ml-base:vos MPS=1 bash scripts/run.sh env VOS_SAVE_MASKS=1 VOS_FAST=1 python -m src.stage_b.vos.eval \
    $W/r26/r26_main_training_last.pth $W/eval_om_davis_all "ailurus_m0.39_at3,ailurus_m0.5_at3,algm_t0.95,algm_t0.98,ftome_l0.3_t0.01_floor64_dil1_noprop_cls64,ftome_l0.7_t0.01_floor64_dil1_noprop_cls64,palgm_t0.77_dil1,palgm_t0.83_dil1,palgm_t0.91_dil1,tome_r64_from1_noprop,tome_r80_from1_noprop"
IMAGE=ml-base:vos MPS=1 bash scripts/run.sh env VOS_SAVE_MASKS=1 VOS_FAST=1 $GV python -m src.stage_b.vos.eval \
    $W/r26/r26_main_training_last.pth $W/eval_om_davis_r40 "$S"
# ViT-B: the same with VOS_VIT=facebook/dinov3-vitb16-pretrain-lvd1689m, head r25, the ViT-B setting (SETTINGS.md)
IMAGE=ml-base:vos bash scripts/run.sh env CUDA_VISIBLE_DEVICES= python -m src.stage_b.vos.official_eval davis $W/eval_om_davis_all
IMAGE=ml-base:vos bash scripts/run.sh env CUDA_VISIBLE_DEVICES= python -m src.stage_b.vos.r40_official davis
```

## ViT-B (three heads) and ViT-L

As Table 1 with `VOS_VIT=...vitb16...`, heads `r41_b1` to `r41_b3`, `algm_t0.95` and the ViT-B setting
(`r40_compare vitb3`, `r40_official vitb3`); ViT-L with `VOS_VIT=...vitl16...`, head `r34`, `algm_t0.95` and the ViT-L
setting (`r40_compare vitl`, `r40_official vitl`). At ViT-L run the method with `VOS_GRAPH_OVERLAP=0` in `GV` and one
process: the side-stream overlap has an unresolved race at ViT-L that can crash a run at the start of a video
(Appendix B). The overlap only reorders work.

## Table 2: where the gain comes from

Logged rows of the reference implementation (`VOS_DIAG=1`, no `GV`) for head r31c, then
`python -m src.stage_b.vos.r40_attr` (in the research repository; the summary is `results/r40_attr.json`).

## Table 5: wall clock

```bash
R="TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor R40_INDUCTOR_OPTS=emulate_precision_casts=1"
IMAGE=ml-base:vos bash scripts/run.sh env $R python -m src.stage_b.vos.r40_wallclock check vits L_csvo   # graphs equal eager
IMAGE=ml-base:vos bash scripts/run.sh env $R python -m src.stage_b.vos.r40_wallclock time vits dense L_csvo ALGM_csvo
IMAGE=ml-base:vos bash scripts/run.sh env $R R40_DET=1 python -m src.stage_b.vos.r40_wallclock time vits dense L_csvo ALGM_csvo
```

`vitb` and `vitl` with the matching `VOS_VIT`. Time on an idle GPU. The kernel breakdown against ALGM (Section 6): `python -m src.stage_b.vos.r40_wallclock parts vits L_csvo`
(likewise `vitb`, `vitl`).

## Implementation equivalence

`python -m src.stage_b.vos.r40_replay` (CPU) replays the ground-truth masks of the 140 long videos through the
reference and the timed implementations and counts differing decisions (`results/r40_replay_gt.json`).
