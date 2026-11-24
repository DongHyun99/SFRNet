#!/usr/bin/env python3
"""Run SFR-Net on a folder of shadow images.

Example:
    python predict.py --checkpoint checkpoints/sfrnet_istd_plus.pth \
        --input_dir ISTD+/test/test_A --mask_dir ISTD+/test/test_B --output_dir results/istd_plus
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ShadowTripletDataset
from inference import SFRNetPredictor, linear_color_restoration, to_uint8
from utils import save_rgb


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='SFR-Net checkpoint (.pth release file or .ckpt)')
    parser.add_argument('--input_dir', required=True, help='directory with shadow images')
    parser.add_argument('--mask_dir', required=True, help='directory with shadow masks (same file stems)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--postprocess', choices=['linear', 'none'], default='linear',
                        help='"linear": color restoration by linear regression on the non-shadow region (default, '
                             'final SFR-Net output); "none": raw composited output')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='also save the raw generator output (predicted/) and the composited output (inpainted/)')
    parser.add_argument('--erode_iterations', type=int, default=1,
                        help='3x3 erosion iterations applied to the mask fed to the generator')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = ShadowTripletDataset(args.input_dir, args.mask_dir, pad_to_modulo=8)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    predictor = SFRNetPredictor(args.checkpoint, device=args.device, erode_iterations=args.erode_iterations)

    for batch in tqdm(loader, desc='SFR-Net'):
        out = predictor(batch['image'], batch['mask'])
        height, width = batch['shape'][0].tolist()
        name = batch['name'][0]
        predicted = to_uint8(out['predicted'][0])[:height, :width]
        inpainted = to_uint8(out['inpainted'][0])[:height, :width]

        if args.postprocess == 'linear':
            shadow = to_uint8(batch['image'][0])[:height, :width]
            mask = batch['mask'][0, 0, :height, :width].numpy()
            final = linear_color_restoration(shadow, predicted, mask)
        else:
            final = inpainted

        save_rgb(os.path.join(args.output_dir, f'{name}.png'), final)
        if args.save_intermediate:
            save_rgb(os.path.join(args.output_dir, 'predicted', f'{name}.png'), predicted)
            save_rgb(os.path.join(args.output_dir, 'inpainted', f'{name}.png'), inpainted)

    print(f'Saved {len(dataset)} results to {args.output_dir}')


if __name__ == '__main__':
    main()
