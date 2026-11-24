#!/usr/bin/env python3
"""Evaluate shadow removal results (RMSE in LAB, PSNR / SSIM in RGB) per region.

Example:
    python evaluate.py --result_dir results/istd_plus \
        --gt_dir ISTD+/test/test_C --mask_dir ISTD+/test/test_B
"""

import argparse
import csv
import os
from functools import partial
from multiprocessing import Pool

import cv2

from data import list_images
from evaluation import aggregate, format_table, image_metrics
from utils import load_mask_uint8, load_rgb_uint8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--mask_dir', required=True)
    parser.add_argument('--resize', type=int, nargs=2, metavar=('W', 'H'), default=None,
                        help='resize results, ground truth and masks before evaluation (e.g. 256 256)')
    parser.add_argument('--csv', default=None, help='optional path for per-image metrics')
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


def _stem_index(directory):
    return {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(directory)}


def _evaluate_one(paths, resize):
    result_path, gt_path, mask_path = paths
    pred, gt, mask = load_rgb_uint8(result_path), load_rgb_uint8(gt_path), load_mask_uint8(mask_path)
    if resize is not None:
        size = tuple(resize)
        pred = cv2.resize(pred, size, interpolation=cv2.INTER_CUBIC)
        gt = cv2.resize(gt, size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image_metrics(pred, gt, mask)


def main():
    args = parse_args()
    results, gts, masks = _stem_index(args.result_dir), _stem_index(args.gt_dir), _stem_index(args.mask_dir)
    names = sorted(results)
    missing = [n for n in names if n not in gts or n not in masks]
    if missing:
        raise FileNotFoundError(f'{len(missing)} results lack ground truth or mask, e.g. {missing[:5]}')

    jobs = [(results[n], gts[n], masks[n]) for n in names]
    with Pool(args.workers) as pool:
        per_image = pool.map(partial(_evaluate_one, resize=args.resize), jobs)

    print(f'Evaluated {len(names)} images from {args.result_dir}')
    print(format_table(aggregate(per_image)))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'rmse_all', 'psnr_shadow', 'psnr_non_shadow', 'psnr_all',
                             'ssim_shadow', 'ssim_non_shadow', 'ssim_all'])
            for name, s in zip(names, per_image):
                writer.writerow([name, s['rmse_all_per_image'], s['psnr_shadow'], s['psnr_non_shadow'],
                                 s['psnr_all'], s['ssim_shadow'], s['ssim_non_shadow'], s['ssim_all']])


if __name__ == '__main__':
    main()
