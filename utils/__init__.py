import os
import random

import cv2
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_rgb(path, image):
    """Saves an (H, W, 3) uint8 RGB array."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise IOError(f'Failed to write {path}')


def load_rgb_uint8(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f'Failed to read image {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask_uint8(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f'Failed to read mask {path}')
    return mask
