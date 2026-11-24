import torch
import torch.nn as nn

from models import load_generator
from models.discriminator import NLayerDiscriminator
from training.losses import (FeatureRecorder, GuidingNetwork, MSSSIML1Loss, NonSaturatingWithR1, ResNetPL,
                             feature_matching_loss, feature_refinement_loss, masked_mix_loss, refinement_layers)


class SFRNet(nn.Module):
    """SFR-Net training graph: guided network G_S, frozen guiding network G_I, discriminator D and losses.

    L = adv * L_Adv + hrfpl * L_HRFPL + fm * L_fm + mix * L_mix + fr * L_fr + gp * R1   (paper Eq. 1)

    G_S and G_I are both initialized from the pretrained LaMa generator.
    Frozen networks (G_I, the HRFPL encoder) follow this module's train/eval mode, i.e. their BatchNorm
    layers use batch statistics during training, exactly as in the training code used for the paper.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.generator = load_generator(cfg.model.init_checkpoint)
        self.discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=4)

        lcfg = cfg.losses
        self.layers = refinement_layers(lcfg.feature_refinement.stages)
        self.student_recorder = FeatureRecorder(self.generator, self.layers)
        self.guiding = GuidingNetwork(load_generator(cfg.model.guiding_checkpoint), self.layers)

        self.mix_loss = MSSSIML1Loss(alpha=lcfg.mix.alpha)
        self.adversarial = NonSaturatingWithR1(weight=lcfg.adversarial.weight, gp_coef=lcfg.adversarial.gp_coef)
        self.hrfpl = ResNetPL(cfg.model.ade20k_weights_dir, weight=lcfg.hrfpl.weight)

    def forward(self, image, mask):
        """Returns the raw generator output and the composited result (non-shadow pixels from the input)."""
        predicted = self.generator(torch.cat([image, mask], dim=1))
        inpainted = mask * predicted + (1 - mask) * image
        return predicted, inpainted

    def generator_step(self, batch):
        image, mask, target = batch['image'], batch['mask'], batch['target']
        lcfg = self.cfg.losses

        predicted, _ = self(image, mask)
        student_features = self.student_recorder.pop()

        loss_mix = masked_mix_loss(self.mix_loss, predicted, target, mask,
                                   lcfg.mix.weight_known, lcfg.mix.weight_missing)

        discr_real_pred, discr_real_features = self.discriminator(target)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted)
        loss_adv = self.adversarial.generator_loss(discr_fake_pred)
        loss_fm = feature_matching_loss(discr_fake_features, discr_real_features) * lcfg.feature_matching.weight
        loss_hrfpl = self.hrfpl(predicted, target)

        teacher_features = self.guiding(target)
        loss_fr = feature_refinement_loss(student_features, teacher_features, self.layers) \
            * lcfg.feature_refinement.weight

        total = loss_mix + loss_adv + loss_fm + loss_hrfpl + loss_fr
        metrics = dict(gen_mix=loss_mix, gen_adv=loss_adv, gen_fm=loss_fm, gen_hrfpl=loss_hrfpl, gen_fr=loss_fr,
                       gen_total=total)
        return total, {k: v.detach() for k, v in metrics.items()}

    def discriminator_step(self, batch):
        image, mask, target = batch['image'], batch['mask'], batch['target']
        with torch.no_grad():
            predicted, _ = self(image, mask)
        self.student_recorder.pop()

        real = target.detach().requires_grad_(True)
        discr_real_pred, _ = self.discriminator(real)
        discr_fake_pred, _ = self.discriminator(predicted)
        loss, metrics = self.adversarial.discriminator_loss(real, discr_real_pred, discr_fake_pred, mask)
        metrics['discr_adv'] = loss.detach()
        return loss, metrics
