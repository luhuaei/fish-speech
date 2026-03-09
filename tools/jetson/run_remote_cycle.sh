#!/bin/bash
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/home/nvidia/fish-speech-jetson}"
SSH_TARGET="${SSH_TARGET:-nvidia@192.168.1.230}"
PRECOMPILE="${PRECOMPILE:-0}"

log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" >&2
}

log "Syncing repo to ${SSH_TARGET}:${REMOTE_DIR}"
ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "mkdir -p ${REMOTE_DIR}"
tar \
  --exclude .git \
  --exclude __pycache__ \
  --exclude '.venv' \
  --exclude '.cache' \
  --exclude 'jetson-bench-results' \
  --exclude 'checkpoints/openaudio-s1-mini' \
  --exclude 'checkpoints/openaudio-s1-mini/**' \
  -cf - . | ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "cd ${REMOTE_DIR} && tar -xf -"

log "Ensuring checkpoint and references directories exist"
ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "cd ${REMOTE_DIR} && mkdir -p checkpoints/openaudio-s1-mini references"

log "Downloading model on Jetson when missing"
ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "bash -lc 'cd ${REMOTE_DIR} && if [ ! -f checkpoints/openaudio-s1-mini/model.pth ]; then python3 -m pip install --user huggingface_hub modelscope && FISH_SPEECH_MODEL_SOURCE=modelscope python3 tools/jetson/download_s1_mini.py; fi'"

log "Building image inside Docker on Jetson"
ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "cd ${REMOTE_DIR} && PRECOMPILE=0 docker compose -f compose.jetson.yml build"

if [ "${PRECOMPILE}" = "1" ]; then
  log "Prewarming compile cache in a GPU-enabled container"
  ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "cd ${REMOTE_DIR} && bash tools/jetson/prewarm_compile_cache_container.sh"

  log "Rebuilding image to bake compile cache"
  ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "cd ${REMOTE_DIR} && PRECOMPILE=0 docker compose -f compose.jetson.yml build"
fi

log "Starting service"
ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "cd ${REMOTE_DIR} && docker compose -f compose.jetson.yml up -d"

log "Waiting for health"
ssh -o StrictHostKeyChecking=no "${SSH_TARGET}" "bash -lc 'for i in {1..60}; do curl -fsS http://127.0.0.1:8080/v1/health >/dev/null && exit 0; sleep 5; done; exit 1'"

log "Service is healthy"
