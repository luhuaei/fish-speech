#!/bin/bash
set -euo pipefail

log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" >&2
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    log "Missing required file: $path"
    exit 1
  fi
}

COMPILE_MODE="${COMPILE:-auto}"
COMPILE_FLAG=""
COMPILE_MARKER="${TORCHINDUCTOR_CACHE_DIR:-/opt/torch-compile-cache}/.compile-ready"

require_file "${LLAMA_CHECKPOINT_PATH}/model.pth"
require_file "${LLAMA_CHECKPOINT_PATH}/config.json"
require_file "${LLAMA_CHECKPOINT_PATH}/tokenizer.tiktoken"
require_file "${LLAMA_CHECKPOINT_PATH}/special_tokens.json"
require_file "${DECODER_CHECKPOINT_PATH}"

mkdir -p /app/references "${TORCHINDUCTOR_CACHE_DIR:-/opt/torch-compile-cache}" "${TRITON_CACHE_DIR:-/opt/triton-cache}"

case "${COMPILE_MODE}" in
  1|true|TRUE|yes|YES)
    COMPILE_FLAG="--compile"
    ;;
  auto|AUTO|"")
    if [ -f "${COMPILE_MARKER}" ]; then
      COMPILE_FLAG="--compile"
    fi
    ;;
  0|false|FALSE|no|NO)
    COMPILE_FLAG=""
    ;;
  *)
    log "Unsupported COMPILE value: ${COMPILE_MODE}"
    exit 1
    ;;
esac

log "Starting Fish Speech Jetson API server"
log "Listen: ${API_SERVER_NAME}:${API_SERVER_PORT}"
log "Compile mode: ${COMPILE_MODE}"
log "Inductor cache: ${TORCHINDUCTOR_CACHE_DIR:-/opt/torch-compile-cache}"
log "Triton cache: ${TRITON_CACHE_DIR:-/opt/triton-cache}"

exec python tools/api_server.py \
  --listen "${API_SERVER_NAME}:${API_SERVER_PORT}" \
  --llama-checkpoint-path "${LLAMA_CHECKPOINT_PATH}" \
  --decoder-checkpoint-path "${DECODER_CHECKPOINT_PATH}" \
  --decoder-config-name "${DECODER_CONFIG_NAME:-modded_dac_vq}" \
  --device cuda \
  --half \
  ${COMPILE_FLAG}
