#!/usr/bin/env python3
"""Train SFR-Net.

Examples:
    python train.py --config configs/sfrnet_istd_plus.yaml
    python train.py --config configs/sfrnet_istd_plus.yaml data.batch_size=2 trainer.out_dir=experiments/debug
    python train.py --config configs/sfrnet_istd_plus.yaml --resume experiments/sfrnet_istd_plus/checkpoints/last.pth
"""

import argparse
import logging

import torch
from omegaconf import OmegaConf

from training import Trainer
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None, help='checkpoint written by this script (last.pth / best.pth)')
    parser.add_argument('overrides', nargs='*', help='OmegaConf dotlist overrides, e.g. data.batch_size=2')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Config:\n%s', OmegaConf.to_yaml(cfg))

    seed_everything(cfg.trainer.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    trainer = Trainer(cfg)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == '__main__':
    main()
