# Pix2PixHD-style patch discriminator.
# Adapted from LaMa (https://github.com/advimman/lama, Apache-2.0).

import numpy as np
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """Returns ``(patch_logits, [intermediate activations])``; activations feed the feature-matching loss."""

    def __init__(self, input_nc=3, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence.append([nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                             norm_layer(nf),
                             nn.LeakyReLU(0.2, True)])

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence.append([nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                         norm_layer(nf),
                         nn.LeakyReLU(0.2, True)])
        sequence.append([nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)])

        for n, layers in enumerate(sequence):
            setattr(self, f'model{n}', nn.Sequential(*layers))

    def forward(self, x):
        activations = [x]
        for n in range(self.n_layers + 2):
            activations.append(getattr(self, f'model{n}')(activations[-1]))
        activations = activations[1:]
        return activations[-1], activations[:-1]
