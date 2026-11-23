#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE=${IMAGE:-ml-base:dev}
# /work: working drive (NVMe) for active data, caches, results. /data: archive drive (USB) for raw downloads, backups.
WORK=${WORK:-$HOME/ml_work}
TTY_FLAGS="-i"
[ -t 0 ] && [ -t 1 ] && TTY_FLAGS="-it"
NAME="ml-$(date +%H%M%S)-$$"
# MPS=1: share the host's CUDA MPS daemon (start it with `nvidia-cuda-mps-control -d`) so several GPU processes run kernels concurrently.
# TORCH_DISABLE_NATIVE_JIT=1 (default here): PyTorch's Python-dispatched Triton ops (sum, norm, topk, outer-product bmm) cost more than they save on the sweep's tiny tensors; replay about 19% faster (O-066).
MPS_FLAGS=()
if [ "${MPS:-0}" = "1" ]; then
  # Without the daemon, CUDA silently falls back to time-slicing and several ranks run slower than fewer (2026-10-29, after an outage).
  python3 -c "import socket; s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET); s.settimeout(5); s.connect('/tmp/nvidia-mps/control')" 2> /dev/null || { echo "run.sh: MPS=1 but the MPS daemon is not running (start it as root: nvidia-cuda-mps-control -d)" >&2; exit 3; }
  MPS_FLAGS=(-v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps)
fi
exec docker run --rm $TTY_FLAGS --name "$NAME" --gpus all \
  --user $(id -u):$(id -g) \
  --group-add "$(getent group docker | cut -d: -f3)" \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$ROOT":/workspace \
  -v "$WORK":/work \
  -v /mnt/data:/data \
  -e HF_HOME=/work/cache/hf \
  -e TORCH_HOME=/work/cache/torch \
  -e HF_TOKEN \
  -e HOME=/workspace/.home \
  -e USER="$(id -un)" \
  -e PYTORCH_CUDA_ALLOC_CONF \
  -e TORCH_DISABLE_NATIVE_JIT="${TORCH_DISABLE_NATIVE_JIT:-1}" \
  "${MPS_FLAGS[@]}" \
  -e IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')" \
  -w /workspace \
  "$IMAGE" "${@:-bash}"
