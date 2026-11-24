"""Shadow removal metrics.

Python re-implementation of the MATLAB evaluation protocol that is standard for ISTD / ISTD+ / SRD
(e.g. ShadowFormer / DeS3 ``evaluate_*.m``):

* RMSE: mean absolute error in CIELAB (summed over L, a, b) -- the quantity reported as "RMSE" in the
  shadow removal literature. ``all`` is averaged per image; ``shadow`` / ``non_shadow`` are averaged per pixel
  over the whole test set (total error / total region pixels).
* PSNR / SSIM: computed in RGB on [0, 1] images. Region scores are computed on full-size images with the
  other region zeroed out (``psnr(x .* M, y .* M)``), as in the MATLAB scripts.

Numbers can differ slightly from MATLAB (sRGB->Lab conversion, SSIM filtering); use the MATLAB scripts for
camera-ready tables.
"""

import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

REGIONS = ('shadow', 'non_shadow', 'all')


def _psnr(x, y):
    mse = np.mean((x - y) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(1.0 / mse))


def _ssim(x, y):
    # Gaussian-weighted SSIM (sigma 1.5), averaged over RGB channels.
    return float(structural_similarity(x, y, data_range=1.0, channel_axis=-1, gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False))


def image_metrics(pred, gt, mask):
    """Per-image statistics.

    Args:
        pred, gt: (H, W, 3) uint8 RGB.
        mask:     (H, W) shadow mask, shadow pixels != 0.
    """
    if pred.shape != gt.shape:
        raise ValueError(f'Prediction shape {pred.shape} != ground-truth shape {gt.shape}')
    pred = pred.astype(np.float64) / 255.0
    gt = gt.astype(np.float64) / 255.0
    shadow = (mask != 0)[..., None].astype(np.float64)
    region_masks = dict(shadow=shadow, non_shadow=1.0 - shadow, all=np.ones_like(shadow))

    stats = {}
    lab_dist = np.abs(rgb2lab(pred) - rgb2lab(gt))
    height, width = mask.shape
    for region, m in region_masks.items():
        stats[f'psnr_{region}'] = _psnr(pred * m, gt * m)
        stats[f'ssim_{region}'] = _ssim(pred * m, gt * m)
        stats[f'lab_sum_{region}'] = float((lab_dist * m).sum())
        stats[f'pixels_{region}'] = float(m.sum())
    stats['rmse_all_per_image'] = stats['lab_sum_all'] / (height * width)
    return stats


def aggregate(per_image):
    """Aggregates a list of ``image_metrics`` outputs into the reported table values."""
    if not per_image:
        raise ValueError('No images to aggregate')
    result = {}
    for region in REGIONS:
        result[f'psnr_{region}'] = float(np.mean([s[f'psnr_{region}'] for s in per_image]))
        result[f'ssim_{region}'] = float(np.mean([s[f'ssim_{region}'] for s in per_image]))
    for region in ('shadow', 'non_shadow'):
        total = sum(s[f'lab_sum_{region}'] for s in per_image)
        pixels = sum(s[f'pixels_{region}'] for s in per_image)
        result[f'rmse_{region}'] = total / pixels if pixels > 0 else float('nan')
    result['rmse_all'] = float(np.mean([s['rmse_all_per_image'] for s in per_image]))
    return result


def format_table(result):
    header = f"{'':12s}{'RMSE':>10s}{'SSIM':>10s}{'PSNR':>10s}"
    rows = [header]
    for region, label in (('shadow', 'Shadow'), ('non_shadow', 'Non-Shadow'), ('all', 'All')):
        rows.append(f"{label:12s}{result[f'rmse_{region}']:10.4f}{result[f'ssim_{region}']:10.4f}"
                    f"{result[f'psnr_{region}']:10.4f}")
    return '\n'.join(rows)
