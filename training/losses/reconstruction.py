# Mix loss: MS-SSIM + Gaussian-weighted L1 (Zhao et al., "Loss Functions for Image Restoration with Neural
# Networks"), paper Eq. 8-11. Adapted from the implementation by Yunpeng Li (Tianjin University).

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(size, sigma):
    coords = torch.arange(size, dtype=torch.float) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return torch.outer(g, g)


class MSSSIML1Loss(nn.Module):
    """Per-pixel ``compensation * (alpha * L_ms-ssim + (1 - alpha) * G_sigma_M * L1)``, shape (B, H, W)."""

    def __init__(self, gaussian_sigmas=(0.5, 1.0, 2.0, 4.0, 8.0), data_range=1.0, k=(0.01, 0.03), alpha=0.025,
                 compensation=200.0):
        super().__init__()
        self.data_range = data_range
        self.c1 = (k[0] * data_range) ** 2
        self.c2 = (k[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        # Channel order r0, g0, b0, r1, g1, b1, ..., rM, gM, bM.
        kernels = [_gaussian_kernel_2d(filter_size, s) for s in gaussian_sigmas for _ in range(3)]
        self.register_buffer('g_masks', torch.stack(kernels)[:, None], persistent=False)

    def forward(self, x, y):
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)
        mux2, muy2, muxy = mux * mux, muy * muy, mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        luminance = (2 * muxy + self.c1) / (mux2 + muy2 + self.c1)
        contrast_structure = (2 * sigmaxy + self.c2) / (sigmax2 + sigmay2 + self.c2)
        l_m = luminance[:, -1] * luminance[:, -2] * luminance[:, -3]
        loss_ms_ssim = 1 - l_m * contrast_structure.prod(dim=1)

        loss_l1 = F.l1_loss(x, y, reduction='none')
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks[-3:], groups=3, padding=self.pad).mean(1)

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.data_range
        return self.compensation * loss_mix


def masked_mix_loss(criterion, pred, target, mask, weight_known, weight_missing):
    """Mix loss weighted per pixel: ``weight_missing`` inside the mask, ``weight_known`` outside."""
    per_pixel = criterion(pred, target)[:, None]  # (B, 1, H, W)
    pixel_weights = mask * weight_missing + (1 - mask) * weight_known
    return (pixel_weights * per_pixel).mean()
