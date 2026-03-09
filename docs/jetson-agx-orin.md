# Jetson AGX Orin 64G deployment

## Scope

This guide targets `Jetson AGX Orin 64G` with `JetPack 6.1 / L4T R36.4.3` and the Jetson-focused server image defined in `docker/Dockerfile.jetson`.

## Prepare the model

Download the built-in `OpenAudio S1-mini` checkpoint into the repo before building the image:

```bash
python3 tools/jetson/download_s1_mini.py
```

Expected files under `checkpoints/openaudio-s1-mini`:

- `model.pth`
- `codec.pth`
- `config.json`
- `tokenizer.tiktoken`
- `special_tokens.json`

## Build the image in Docker

The build happens inside Docker. The host only provides the source tree, Docker daemon, and local model files.

```bash
docker compose -f compose.jetson.yml build
```

Useful build-time knobs:

- `PRECOMPILE=1` enables build-time `torch.compile` warm-up and bakes the cache into the final image when it succeeds.
- `PIP_INDEX_URL=http://wa.lan:10608/simple` keeps dependency resolution on the local mirror first.
- `PIP_EXTRA_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126` is the fallback Jetson package index.
- Jetson wheels for `torchvision` and `triton` are pulled from `http://wa.lan:10608/simple` and baked into the image.

Example:

```bash
PRECOMPILE=1 docker compose -f compose.jetson.yml build
```

If you need GPU-backed `torch.compile` artifacts baked into the image, use the two-phase helper on the Jetson host:

```bash
PRECOMPILE=1 REMOTE_DIR=/home/nvidia/fish-speech-jetson-ms ./tools/jetson/run_remote_cycle.sh
```

This flow builds the image once, runs `tools/jetson/prewarm_compile_cache_container.sh` in a GPU-enabled container, and rebuilds the image with `compile-cache/` copied into `/opt/torch-compile-cache` and `/opt/triton-cache`.

## Run the server

```bash
docker compose -f compose.jetson.yml up -d
```

The server listens on `http://127.0.0.1:8080` by default.

Mounted paths:

- `./references -> /app/references`

Optional override mount if you want to replace the built-in model:

```yaml
volumes:
  - ./references:/app/references
  - ./checkpoints:/app/checkpoints
```

## Compile cache behavior

When build-time precompile succeeds, the image contains:

- `TORCHINDUCTOR_CACHE_DIR=/opt/torch-compile-cache`
- `TRITON_CACHE_DIR=/opt/triton-cache`

At runtime, `COMPILE=auto` enables `--compile` only when the baked cache marker exists. You can override this behavior:

- `COMPILE=auto` uses baked cache when available
- `COMPILE=1` forces compile mode
- `COMPILE=0` disables compile mode

## HTTP endpoints

Existing endpoints remain available:

- `POST /v1/tts`
- `POST /v1/vqgan/encode`
- `POST /v1/vqgan/decode`
- `POST /v1/references/add`
- `GET /v1/health`

OpenAI-compatible endpoint:

- `POST /v1/audio/speech`

Example:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o speech.mp3 \
  -d '{
    "model": "openaudio-s1-mini",
    "input": "Hello from Jetson.",
    "response_format": "mp3"
  }'
```

## Troubleshooting

- If build fails on missing checkpoint files, re-run `python3 tools/jetson/download_s1_mini.py`.
- If compile warm-up fails, the image still builds and falls back to non-compiled startup.
- If you want to verify the service is healthy, call `curl http://127.0.0.1:8080/v1/health`.
