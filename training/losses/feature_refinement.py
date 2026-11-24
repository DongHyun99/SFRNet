# Feature refinement loss L_fr (paper Eq. 12): feature-level knowledge distillation from the guiding
# network G_I (reconstructs I_gt from I_gt and an empty mask) to the guided network G_S.

import torch
import torch.nn as nn
import torch.nn.functional as F

# FFC residual blocks are generator.model[5:23]; the paper's three stages (Fig. 6) use six blocks each.
STAGE_BLOCKS = {
    'shallow': range(5, 11),
    'middle': range(11, 17),
    'deep': range(17, 23),
}


def refinement_layers(stages):
    """Names of the hooked activations (local and global branch of each block's second FFC) for ``stages``."""
    layers = []
    for stage in stages:
        if stage not in STAGE_BLOCKS:
            raise ValueError(f'Unknown stage "{stage}", expected one of {list(STAGE_BLOCKS)}')
        for block in STAGE_BLOCKS[stage]:
            layers += [f'{block}.conv2.act_g', f'{block}.conv2.act_l']
    return layers


class FeatureRecorder:
    """Stores the outputs of named sub-modules of ``generator.model`` on every forward pass."""

    def __init__(self, generator, layers):
        modules = dict(generator.model.named_modules())
        self.features = {}
        self.handles = [modules[name].register_forward_hook(self._hook(name)) for name in layers]

    def _hook(self, name):
        def fn(_module, _inputs, output):
            self.features[name] = output
        return fn

    def pop(self):
        features, self.features = self.features, {}
        return features


def feature_refinement_loss(student_features, teacher_features, layers):
    """Mean over layers of the per-element MSE between G_S and G_I activations."""
    return torch.stack([F.mse_loss(student_features[name], teacher_features[name]) for name in layers]).mean()


class GuidingNetwork(nn.Module):
    """Frozen G_I. Its BatchNorm layers follow the parent module's train/eval mode, as in the original code."""

    def __init__(self, generator, layers):
        super().__init__()
        self.generator = generator.requires_grad_(False)
        self.recorder = FeatureRecorder(self.generator, layers)

    @torch.no_grad()
    def forward(self, target):
        empty_mask = torch.zeros_like(target[:, :1])
        self.generator(torch.cat([target, empty_mask], dim=1))
        return self.recorder.pop()
