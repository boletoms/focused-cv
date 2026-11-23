#!/bin/bash
# As lvos_launch_w.sh with extra environment settings for eval_lvos (arguments 7 on, e.g. VOS_SKIP_DENSE=1
# VOS_SEQS=/workspace/reports/stage_b/queue/r40_gate_subset.txt). Deterministic kernels are the default (R-29).
# usage: scripts/lvos_launch_env.sh <weights under /work> <out dir under /work> <settings> <split> <workers> <log tag> [VAR=value ...]
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVS=("${@:7}")
for k in $(seq 1 $5); do
  IMAGE=ml-base:vos MPS=1 scripts/run.sh env "${ENVS[@]}" python -m src.stage_b.vos.eval_lvos "$1" "$2" "$3" - "$4" > ~/ml_work/${6}_$k.log 2>&1 &
  sleep 3
done
wait
date +%H:%M
