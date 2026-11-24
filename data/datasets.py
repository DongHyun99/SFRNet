import glob
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_images(directory):
    files = [f for f in glob.glob(os.path.join(directory, '*')) if f.lower().endswith(IMG_EXTENSIONS)]
    if not files:
        raise FileNotFoundError(f'No images found in {directory}')
    return sorted(files)


def _index_by_stem(directory):
    index = {}
    for path in list_images(directory):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in index:
            raise ValueError(f'Duplicate image stem "{stem}" in {directory}')
        index[stem] = path
    return index


def read_rgb(path):
    """Reads an image as float32 RGB in [0, 1], shape (3, H, W)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'Failed to read image {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0


def read_mask(path):
    """Reads a shadow mask as float32 in [0, 1], shape (1, H, W)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f'Failed to read mask {path}')
    return mask[None].astype(np.float32) / 255.0


def ceil_modulo(x, mod):
    return x if x % mod == 0 else (x // mod + 1) * mod


def pad_to_modulo(img, mod):
    """Symmetric padding of a (C, H, W) array so that H and W are multiples of ``mod``."""
    _, height, width = img.shape
    return np.pad(img, ((0, 0), (0, ceil_modulo(height, mod) - height), (0, ceil_modulo(width, mod) - width)),
                  mode='symmetric')


class ShadowTripletDataset(Dataset):
    """Shadow image / shadow mask / shadow-free image triplets matched by file stem.

    ``gt_dir`` is optional (inference on unpaired images). Every sample contains
    ``image`` (3, H, W), ``mask`` (1, H, W), ``name`` and, if available, ``target`` (3, H, W), all in [0, 1].

    Training options:
        crop_size: random crop size (None keeps the full image).
        hflip: random horizontal flip.
    Evaluation options:
        pad_to_modulo: symmetric padding so the generator sees sizes divisible by this value;
            ``shape`` (original H, W) is returned to crop the prediction back.
    """

    def __init__(self, shadow_dir, mask_dir, gt_dir=None, crop_size=None, hflip=False, pad_to_modulo=None):
        shadows = _index_by_stem(shadow_dir)
        masks = _index_by_stem(mask_dir)
        gts = _index_by_stem(gt_dir) if gt_dir is not None else None

        self.names = sorted(shadows)
        missing = [n for n in self.names if n not in masks or (gts is not None and n not in gts)]
        if missing:
            raise FileNotFoundError(f'{len(missing)} shadow images lack a mask or ground truth, e.g. {missing[:5]}')

        self.shadow_paths = [shadows[n] for n in self.names]
        self.mask_paths = [masks[n] for n in self.names]
        self.gt_paths = [gts[n] for n in self.names] if gts is not None else None
        self.crop_size = crop_size
        self.hflip = hflip
        self.pad_to_modulo = pad_to_modulo

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        sample = dict(image=read_rgb(self.shadow_paths[i]), mask=read_mask(self.mask_paths[i]))
        if self.gt_paths is not None:
            sample['target'] = read_rgb(self.gt_paths[i])

        _, height, width = sample['image'].shape
        for key, value in sample.items():
            if value.shape[1:] != (height, width):
                raise ValueError(f'{key} of "{self.names[i]}" has size {value.shape[1:]}, expected {(height, width)}')

        if self.crop_size is not None:
            if height < self.crop_size or width < self.crop_size:
                raise ValueError(f'"{self.names[i]}" ({height}x{width}) is smaller than crop_size={self.crop_size}')
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            sample = {k: v[:, top:top + self.crop_size, left:left + self.crop_size] for k, v in sample.items()}
        if self.hflip and random.random() < 0.5:
            sample = {k: v[:, :, ::-1] for k, v in sample.items()}

        if self.pad_to_modulo is not None and self.pad_to_modulo > 1:
            sample = {k: pad_to_modulo(v, self.pad_to_modulo) for k, v in sample.items()}
            sample['shape'] = np.array([height, width])

        sample = {k: np.ascontiguousarray(v) for k, v in sample.items()}
        sample['name'] = self.names[i]
        return sample
