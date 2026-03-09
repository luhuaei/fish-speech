from argparse import ArgumentParser
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.server.model_manager import ModelManager


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--llama-checkpoint-path", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint-path", type=Path, required=True)
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ModelManager(
        mode="tts",
        device=args.device,
        half=args.half,
        compile=args.compile,
        llama_checkpoint_path=str(args.llama_checkpoint_path),
        decoder_checkpoint_path=str(args.decoder_checkpoint_path),
        decoder_config_name=args.decoder_config_name,
    )
