#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-fish-speech-jetson:orin}"
MODEL_DIR="${MODEL_DIR:-checkpoints/openaudio-s1-mini}"
CACHE_ROOT="${CACHE_ROOT:-compile-cache}"
TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-${CACHE_ROOT}/torch}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"

mkdir -p "${TORCH_CACHE_DIR}" "${TRITON_CACHE_DIR}"
rm -rf "${TORCH_CACHE_DIR}"/* "${TRITON_CACHE_DIR}"/*

log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" >&2
}

log "Prewarming torch.compile cache with ${IMAGE}"
docker run --rm \
  --runtime nvidia \
  --network host \
  -e USER=appuser \
  -e LOGNAME=appuser \
  -e LNAME=appuser \
  -e USERNAME=appuser \
  -e TORCHINDUCTOR_CACHE_DIR=/opt/torch-compile-cache \
  -e TRITON_CACHE_DIR=/opt/triton-cache \
  -v "$PWD/${TORCH_CACHE_DIR}:/opt/torch-compile-cache" \
  -v "$PWD/${TRITON_CACHE_DIR}:/opt/triton-cache" \
  --entrypoint python \
  "${IMAGE}" \
  tools/jetson/prewarm_compile_cache.py \
    --llama-checkpoint-path "/app/${MODEL_DIR}" \
    --decoder-checkpoint-path "/app/${MODEL_DIR}/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --device cuda \
    --half \
    --compile

touch "${TORCH_CACHE_DIR}/.compile-ready"
log "Compile cache ready at ${TORCH_CACHE_DIR}"
