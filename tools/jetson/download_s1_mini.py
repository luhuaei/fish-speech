#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

FILES = [
    "model.pth",
    "codec.pth",
    "config.json",
    "special_tokens.json",
    "tokenizer.tiktoken",
    "README.md",
]
LOCAL_DIR = Path("checkpoints/openaudio-s1-mini")
HF_REPO = "fishaudio/openaudio-s1-mini"
MODELSCOPE_REPO = "fishaudio/openaudio-s1-mini"


def download_from_hf() -> None:
    from huggingface_hub import hf_hub_download

    for filename in FILES:
        path = LOCAL_DIR / filename
        if path.exists():
            print(f"skip {filename}")
            continue
        print(f"huggingface download {filename}")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=str(LOCAL_DIR),
            resume_download=True,
        )


def download_from_modelscope() -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    print("modelscope snapshot download start")
    snapshot_download(model_id=MODELSCOPE_REPO, local_dir=str(LOCAL_DIR), revision="master")


def validate() -> None:
    missing = [filename for filename in FILES[:-1] if not (LOCAL_DIR / filename).exists()]
    if missing:
        raise RuntimeError(f"missing required files: {missing}")


if __name__ == "__main__":
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        source = os.environ.get("FISH_SPEECH_MODEL_SOURCE", "auto")
        if source in {"hf", "huggingface", "auto"}:
            download_from_hf()
            validate()
        elif source in {"modelscope", "ms"}:
            download_from_modelscope()
            validate()
        else:
            raise RuntimeError(f"unsupported source: {source}")
    except Exception as exc:
        print(f"primary download failed: {exc}")
        print("falling back to ModelScope")
        download_from_modelscope()
        validate()
