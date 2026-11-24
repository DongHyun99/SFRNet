# Non-saturating adversarial loss with R1 gradient penalty (paper Eq. 2-4).
# Adapted from LaMa (https://github.com/advimman/lama, Apache-2.0).

import torch
import torch.nn.functional as F


def _resize_mask(mask, shape):
    return F.interpolate(mask, size=shape, mode='nearest') if mask.shape[-2:] != shape else mask


class NonSaturatingWithR1:
    """Patches inside the shadow mask are "fake" for the discriminator, patches outside are "real"."""

    def __init__(self, weight=10.0, gp_coef=0.001):
        self.weight = weight
        self.gp_coef = gp_coef

    def generator_loss(self, discr_fake_pred):
        return F.softplus(-discr_fake_pred).mean() * self.weight

    def discriminator_loss(self, real_batch, discr_real_pred, discr_fake_pred, mask):
        """``real_batch`` must have ``requires_grad=True`` before the discriminator forward pass (R1 penalty)."""
        real_loss = F.softplus(-discr_real_pred)
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean() * self.gp_coef

        mask = _resize_mask(mask, discr_fake_pred.shape[-2:])
        fake_loss = F.softplus(discr_fake_pred) * mask + F.softplus(-discr_fake_pred) * (1 - mask)

        loss = (real_loss + grad_penalty + fake_loss).mean()
        metrics = dict(discr_real_out=discr_real_pred.mean().detach(),
                       discr_fake_out=discr_fake_pred.mean().detach(),
                       discr_real_gp=grad_penalty.detach())
        return loss, metrics
