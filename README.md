# Focus maps for frozen ViTs in long-video object segmentation

Code for the preprint *Focus Only on What's Important: Training-Free Focus Maps for Frozen Vision Transformers in Long-Video Object Segmentation* A frozen DINOv3 ViT replaces the stride-16 pixel encoder of the Cutie video object segmenter.
Before the encoder runs on a frame, a **focus map** decides how much compute each patch gets:

- **Expectation (high tier, never merged).** Each tracked object is expected at its last mask moved by its centroid
  velocity and by a block-matching displacement found on the current frame's 4x4-pooled pixels (search radius 8
  patches), grown by one patch.
- **Surprise check.** Each patch outside the expectation compares its pooled pixels with the previous frame's, shifted
  by the integer camera motion (radius 4 patches), with a variance floor and a 4-pixel tolerance. Fired patches inside
  an object's ring or a lost object's search region join the high tier; other fired patches, the speed-scaled rings
  and the search regions form the **middle tier**, which the compared merger's own rules (ALGM) merge, unchanged.
- **Low tier.** Everything else is merged by ALGM's rules and then by a budget (nominal rate 0.7 per block, capped at
  one half by pairing, down to a floor of 64 or 128 tokens).
- **Fallback.** When every object is lost or the map is empty, the frame runs the baseline itself.

The focus policy is training-free: no part of it is learned and the ViT stays frozen (Cutie's heads are trained once
per backbone, from the official Cutie weights). On held-out long videos (LVOS V2 valid, 239 objects, three heads,
protocol pre-specified in `docs/R40_expectation_plan.md`) it is more accurate than ALGM
at a 16% lower encoder cost, and it is at dense level on short clips (DAVIS 2017 val). See the preprint for the
numbers, the ablations and the limitations.

```bash
git clone https://github.com/boletoms/focused-cv.git && cd focused-cv
```

## Repository layout

| Path | What it holds |
| --- | --- |
| `src/stage_b/expect.py` | The focus map: object states, expectation, surprise check, tiers (reference implementation) |
| `src/stage_b/expect_dev.py`, `frame_runner.py`, `tiered_graph.py` | The timed implementation: the same decisions on the GPU, compiled segments, CUDA graphs, variable-length attention |
| `src/stage_b/baselines.py`, `methods.py` | Token-reduction baselines (ToMe, ALGM, AiluRus, Expedite, PiToMe, G2TM) and their focus-map-aware variants |
| `src/stage_b/model.py`, `flops.py` | The ViT with method hooks; the FLOPs counter (every method's own extra work is counted) |
| `src/stage_b/vos/` | Cutie integration (`dino_encoder.py`), training (`train.py`), evaluators (`eval.py` for DAVIS, `eval_lvos.py` for LVOS), official scoring (`official_eval.py`), comparisons and timing |
| `src/stage_a/` | The DINOv3 loader and shared helpers |
| `tests/` | Unit and equivalence tests (CPU) |
| `docker/` | Image recipes (`Dockerfile` base, `Dockerfile.vos` for Cutie) |
| `scripts/` | `run.sh` (every command runs in Docker), launchers, `fetch_third_party.sh` |
| `reproduce/` | The command lines behind each table, and `SETTINGS.md` (the setting names) |
| `results/` | The summary result files the preprint's tables are built from |

## Setup

Requirements: Linux, an NVIDIA GPU (all results: one 16 GB consumer GPU), Docker with the NVIDIA container toolkit.

```bash
docker build -f docker/Dockerfile -t ml-base:dev docker
docker build -f docker/Dockerfile.vos -t ml-base:vos docker
# docker/requirements.lock.txt records the exact package versions of every reported run; install it in place of
# requirements.txt for an exact environment.
export WORK=$HOME/ml_work            # data, weights, caches and results (mounted as /work)
bash scripts/fetch_third_party.sh    # Cutie and the official evaluation toolkits at pinned commits
```

DINOv3 weights are downloaded from Hugging Face on first use. They are gated: accept the DINOv3 licence on the model
page and export `HF_TOKEN` in your shell (never write it to a file).

Datasets, under `$WORK/data` (each from its official source, under its own terms):

| Dataset | Path | Used for |
| --- | --- | --- |
| DAVIS 2017 trainval 480p | `davis/DAVIS` | Training (train) and short-clip evaluation (val) |
| MOSE train | `mose/MOSE_release/train` | Training |
| LVOS V1 val | `lvos/valid` | Long videos (the 50 V1 videos) |
| LVOS V2 train, valid (with `meta_jsons/`) | `lvos_v2/train`, `lvos_v2/valid` | Development (train); held-out test (valid, 140 videos) |
| Cutie weights (`cutie-small-mega.pth`) | `downloads/cutie_weights` | Initialising the heads |

## Running

Every command runs in Docker through `scripts/run.sh` (the repository is mounted at `/workspace`, `$WORK` at `/work`).

```bash
# tests (CPU)
IMAGE=ml-base:vos bash scripts/run.sh env CUDA_VISIBLE_DEVICES= python -m pytest -q tests

# train a head (ViT-S example; see reproduce/TRAINING.md for every head)
IMAGE=ml-base:vos bash scripts/run.sh python -m src.stage_b.vos.train r26_s1 5000 2 - init=/work/stage_b/vos/r26_init.pth only=proj
IMAGE=ml-base:vos bash scripts/run.sh python -m src.stage_b.vos.train r26 40000 2 - init=/work/stage_b/vos/r26_s1/r26_s1_main_training_last.pth

# evaluate on LVOS V2 valid: dense, ALGM 0.98 and the method, masks saved for the official toolkit
S="titome_t0.98_l0.7_floor64_dil1_cls64~algm_t0.98@exp-pix-mot-rl-r0-e0.06-ring-srch-f0.05-o1"
IMAGE=ml-base:vos bash scripts/run.sh env VOS_SAVE_MASKS=1 python -m src.stage_b.vos.eval_lvos \
    /work/stage_b/vos/r31c/r31c_main_training_last.pth /work/stage_b/vos/my_run "algm_t0.98,$S" - v2_valid
IMAGE=ml-base:vos bash scripts/run.sh env CUDA_VISIBLE_DEVICES= python -m src.stage_b.vos.official_eval lvos /work/stage_b/vos/my_run
```

The timed implementation is selected with `VOS_GRAPH=2 VOS_GRAPH_COMPILE=segment VOS_GRAPH_KERNEL=varlen
VOS_GRAPH_OVERLAP=1 R40_INDUCTOR_OPTS=emulate_precision_casts=1`. It takes the same decisions as the reference
implementation (`tests/stage_b/test_frame_runner.py`, `src/stage_b/vos/r40_replay.py`). Evaluations use deterministic
kernels. Attention runs in fp32 or bf16, never fp16.

`reproduce/README.md` lists, per table of the preprint, the runs and the comparison that produces it.

## Terminology

The paper calls the per-patch compute map the **focus map** and its high- and middle-tier patches the **focus areas**.
Some code identifiers keep the earlier name `interest` (for example `State.interest` and the `interest=` argument),
and setting names keep their original prefixes (`titome_...`), because they key the stored result files.

## Licence

The code is released under the Apache License 2.0 (`LICENSE`), except the parts derived from third-party code, which
keep their original licences. `NOTICE` lists them. In particular, the PiToMe port in `src/stage_b/baselines.py`
(marked in the file) follows PiToMe's CC BY-NC 4.0 licence and may not be used commercially (`LICENSES/PiToMe-CC-BY-NC-4.0.md` gives the attribution and the changes). Datasets, Cutie, DINOv3
weights and the evaluation toolkits are not redistributed here; their own licences apply.

## Citation

See `CITATION.cff`. The arXiv identifier is added once the preprint is public.

## Data terms
The datasets are not redistributed. LVOS (V1 and V2): annotations CC BY 4.0, evaluation toolkits BSD-3-Clause, data for non-commercial research only; the videos come from VOT-LT 2019 and LaSOT and are not the organizers' property
(terms on the LVOS competition page). DAVIS 2017 and MOSE keep their own licences. Download from the dataset authors; `reproduce/README.md` lists the paths the scripts expect.
