#!/usr/bin/env python3
"""Export the generator weights of a training checkpoint into a small release file.

Accepts checkpoints written by train.py, original PyTorch-Lightning SFR-Net checkpoints and LaMa checkpoints.

Example:
    python scripts/export_checkpoint.py experiments/sfrnet_istd_plus/checkpoints/best.pth \
        checkpoints/sfrnet_istd_plus.pth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_generator, extract_generator_state_dict, load_checkpoint_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()

    checkpoint = load_checkpoint_file(args.src)
    state = extract_generator_state_dict(checkpoint)
    build_generator().load_state_dict(state, strict=True)  # validate before writing

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save({'generator': {k: v.clone() for k, v in state.items()}}, args.dst)
    print(f'Wrote {len(state)} generator tensors to {args.dst}')


if __name__ == '__main__':
    main()
