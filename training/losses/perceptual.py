# High receptive field perceptual loss (HRFPL, paper Eq. 5) and discriminator feature-matching loss.
# Adapted from LaMa (https://github.com/advimman/lama, Apache-2.0).

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ade20k_resnet import load_ade20k_resnet50dilated

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


class ResNetPL(nn.Module):
    def __init__(self, weights_dir, weight=30.0):
        super().__init__()
        # Frozen weights. BatchNorm follows the parent module's train/eval mode, as in the original training code.
        self.impl = load_ade20k_resnet50dilated(weights_dir)
        self.impl.requires_grad_(False)
        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)
        losses = [F.mse_loss(p, t) for p, t in zip(self.impl(pred), self.impl(target))]
        return torch.stack(losses).sum() * self.weight


def feature_matching_loss(fake_features, real_features):
    return torch.stack([F.mse_loss(f, r) for f, r in zip(fake_features, real_features)]).mean()
