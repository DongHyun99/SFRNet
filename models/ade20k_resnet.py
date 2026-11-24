# Dilated ResNet-50 encoder trained on ADE20k scene parsing, used by the
# high receptive field perceptual loss (HRFPL).
# Adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch (BSD-3-Clause)
# via LaMa (https://github.com/advimman/lama, Apache-2.0).

import os
from functools import partial

import torch
import torch.nn as nn


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        return self.relu(out)


class ResNetDilated(nn.Module):
    """ResNet-50 (deep stem) with dilation in layer3/layer4 (output stride 8)."""

    def __init__(self, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 128
        self.conv1 = _conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = _conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.layer3.apply(partial(self._nostride_dilate, dilate=2))
        self.layer4.apply(partial(self._nostride_dilate, dilate=4))

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    @staticmethod
    def _nostride_dilate(m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            elif m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        """Returns the feature maps of layer1..layer4."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        feats = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            feats.append(x)
        return feats


ADE20K_ENCODER_RELPATH = os.path.join('ade20k', 'ade20k-resnet50dilated-ppm_deepsup', 'encoder_epoch_20.pth')


def load_ade20k_resnet50dilated(weights_dir):
    """Loads ``<weights_dir>/ade20k/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth``."""
    path = os.path.join(weights_dir, ADE20K_ENCODER_RELPATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'ADE20k encoder weights not found at {path}. See README "Pretrained weights".')
    state = torch.load(path, map_location='cpu')
    # The released encoder was trained with SynchronizedBatchNorm and still carries its unused
    # ``_tmp_running_{mean,var}`` accumulators; drop only those so every real parameter loads strictly.
    state = {k: v for k, v in state.items() if '._tmp_running_' not in k}
    model = ResNetDilated()
    model.load_state_dict(state)
    return model
