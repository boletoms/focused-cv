#!/bin/bash
# Third-party code at the commits the results were produced with, cloned into $WORK/third_party (mounted as
# /work/third_party; Cutie's path is fixed in src/stage_b/vos/eval.py and train.py). Runs in Docker (git in the image).
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-$HOME/ml_work}
mkdir -p "$WORK/third_party/official_eval"
IMAGE=${IMAGE:-ml-base:vos} bash scripts/run.sh env CUDA_VISIBLE_DEVICES= bash -c '
set -e
get() { [ -d "$2" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; echo "$2 @ $(git -C "$2" rev-parse --short HEAD)"; }
get https://github.com/hkchengrex/Cutie.git /work/third_party/Cutie ec5cdd4
get https://github.com/davisvideochallenge/davis2017-evaluation.git /work/third_party/official_eval/davis2017-evaluation ac7c43f
get https://github.com/LingyiHongfd/lvos-evaluation.git /work/third_party/official_eval/lvos-evaluation 25cb02d
# for tests/stage_b/test_r1_baselines.py (each port against the authors code)
get https://github.com/tue-mps/algm-segmenter.git /work/third_party/algm-segmenter 674eaa2
get https://github.com/caddyless/ailurus.git /work/third_party/ailurus 7765be2
get https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-Segmenter.git /work/third_party/Expedit-Segmenter 812e24b
get https://github.com/hchautran/PiToMe.git /work/third_party/PiToMe 550b5de
git clone -q -b segm https://github.com/vbercy/g2tm.git /work/third_party/g2tm 2>/dev/null || true; git -C /work/third_party/g2tm checkout -q 963093e
'
