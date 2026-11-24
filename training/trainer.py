import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import ShadowTripletDataset
from evaluation import aggregate, format_table, image_metrics
from inference import erode_mask, to_uint8
from training.model import SFRNet
from utils import save_rgb

LOGGER = logging.getLogger(__name__)


def make_optimizer(parameters, kind, lr):
    if kind == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    if kind == 'lion':
        from lion_pytorch import Lion  # optional dependency: pip install lion-pytorch
        return Lion(parameters, lr=lr)
    raise ValueError(f'Unknown optimizer "{kind}" (expected "adam" or "lion")')


def _set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def _optimizer_step(optimizer, loss, clip_val):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if clip_val:
        params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, clip_val)
    optimizer.step()


class Trainer:
    def __init__(self, cfg):
        if not cfg.trainer.monitor.startswith('rmse_'):
            raise ValueError(f'trainer.monitor must be an RMSE metric (lower is better), got {cfg.trainer.monitor}')
        self.cfg = cfg
        self.device = torch.device(cfg.trainer.device)
        self.out_dir = cfg.trainer.out_dir
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)
        OmegaConf.save(cfg, os.path.join(self.out_dir, 'config.yaml'))

        self.model = SFRNet(cfg).to(self.device)
        ocfg = cfg.optimizers
        self.opt_g = make_optimizer(self.model.generator.parameters(), ocfg.generator.kind, ocfg.generator.lr)
        self.opt_d = make_optimizer(self.model.discriminator.parameters(), ocfg.discriminator.kind,
                                    ocfg.discriminator.lr)

        dcfg = cfg.data
        train_set = ShadowTripletDataset(dcfg.train.shadow_dir, dcfg.train.mask_dir, dcfg.train.gt_dir,
                                         crop_size=dcfg.train.crop_size, hflip=dcfg.train.hflip)
        self.train_loader = DataLoader(train_set, batch_size=dcfg.batch_size, shuffle=True,
                                       num_workers=dcfg.num_workers, drop_last=True, pin_memory=True)
        self.val_set = None
        if dcfg.get('val') is not None:
            self.val_set = ShadowTripletDataset(dcfg.val.shadow_dir, dcfg.val.mask_dir, dcfg.val.gt_dir,
                                                pad_to_modulo=8)

        self.writer = SummaryWriter(os.path.join(self.out_dir, 'tb'))
        self.epoch = 0
        self.global_step = 0
        self.best_score = float('inf')

    # ------------------------------------------------------------------ checkpoints
    def save(self, name):
        state = dict(generator=self.model.generator.state_dict(),
                     discriminator=self.model.discriminator.state_dict(),
                     opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict(),
                     epoch=self.epoch, global_step=self.global_step, best_score=self.best_score,
                     config=OmegaConf.to_container(self.cfg, resolve=True))
        torch.save(state, os.path.join(self.out_dir, 'checkpoints', name))

    def resume(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.model.generator.load_state_dict(state['generator'])
        self.model.discriminator.load_state_dict(state['discriminator'])
        self.opt_g.load_state_dict(state['opt_g'])
        self.opt_d.load_state_dict(state['opt_d'])
        self.epoch = state['epoch'] + 1
        self.global_step = state['global_step']
        self.best_score = state['best_score']
        LOGGER.info(f'Resumed from {path} (next epoch {self.epoch}, step {self.global_step})')

    # ------------------------------------------------------------------ training
    def train_epoch(self):
        tcfg = self.cfg.trainer
        self.model.train()
        progress = tqdm(self.train_loader, desc=f'epoch {self.epoch}', dynamic_ncols=True)
        for batch in progress:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}

            _set_requires_grad(self.model.generator, True)
            _set_requires_grad(self.model.discriminator, False)
            loss_g, metrics = self.model.generator_step(batch)
            _optimizer_step(self.opt_g, loss_g, tcfg.gradient_clip_val)

            _set_requires_grad(self.model.generator, False)
            _set_requires_grad(self.model.discriminator, True)
            loss_d, d_metrics = self.model.discriminator_step(batch)
            _optimizer_step(self.opt_d, loss_d, tcfg.gradient_clip_val)
            metrics.update(d_metrics)

            if not (torch.isfinite(loss_g) and torch.isfinite(loss_d)):
                raise FloatingPointError(f'Non-finite loss at step {self.global_step}: {metrics}')

            if self.global_step % tcfg.log_every_n_steps == 0:
                for k, v in metrics.items():
                    self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)
                progress.set_postfix(g=f'{loss_g.item():.3f}', d=f'{loss_d.item():.3f}',
                                     fr=f"{metrics['gen_fr'].item():.4f}")
            self.global_step += 1

    @torch.no_grad()
    def validate(self):
        vcfg = self.cfg.data.val
        generator = self.model.generator
        generator.eval()
        loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=self.cfg.data.num_workers)
        sample_dir = os.path.join(self.out_dir, 'samples', f'epoch_{self.epoch:03d}')
        per_image = []
        for i, batch in enumerate(loader):
            if vcfg.max_images is not None and i >= vcfg.max_images:
                break
            image = batch['image'].to(self.device)
            mask = (batch['mask'].to(self.device) > 0).float()
            predicted = generator(torch.cat([image, erode_mask(mask, 1)], dim=1))
            inpainted = mask * predicted + (1 - mask) * image

            height, width = batch['shape'][0].tolist()
            result = to_uint8(inpainted[0])[:height, :width]
            gt = to_uint8(batch['target'][0])[:height, :width]
            per_image.append(image_metrics(result, gt, batch['mask'][0, 0, :height, :width].numpy()))
            if i < vcfg.num_samples:
                save_rgb(os.path.join(sample_dir, f"{batch['name'][0]}.png"),
                         np.concatenate([to_uint8(image[0])[:height, :width], result, gt], axis=1))
        self.model.train()

        result = aggregate(per_image)
        for k, v in result.items():
            self.writer.add_scalar(f'val/{k}', v, self.global_step)
        LOGGER.info(f'Validation after epoch {self.epoch} ({len(per_image)} images):\n{format_table(result)}')
        return result

    def fit(self):
        tcfg = self.cfg.trainer
        while self.epoch < tcfg.max_epochs:
            self.train_epoch()
            if self.val_set is not None and (self.epoch + 1) % tcfg.val_every_n_epochs == 0:
                score = self.validate()[tcfg.monitor]
                if score < self.best_score:
                    self.best_score = score
                    self.save('best.pth')
                    LOGGER.info(f'New best {tcfg.monitor}={score:.4f} at epoch {self.epoch}')
            self.save('last.pth')
            self.epoch += 1
        self.writer.close()
