import numpy as np
import torch
import torch.nn.functional as F

from models import load_generator


def erode_mask(mask, iterations):
    """Binary erosion with a 3x3 square kernel (pixels outside the image do not erode, as in cv2.erode)."""
    for _ in range(iterations):
        mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return mask


def to_uint8(tensor):
    """(3, H, W) float tensor in [0, 1] -> (H, W, 3) uint8 RGB array."""
    return np.clip(tensor.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)


class SFRNetPredictor:
    """Shadow removal with a trained SFR-Net guided network G_S.

    The generator receives the full shadow image concatenated with the shadow mask. The mask fed to
    the generator is eroded by ``erode_iterations`` (3x3 kernel); the output is composited with the
    original mask, so non-shadow pixels are copied from the input image.
    """

    def __init__(self, checkpoint, device='cuda', erode_iterations=1):
        self.device = torch.device(device)
        self.generator = load_generator(checkpoint).to(self.device).eval()
        self.erode_iterations = erode_iterations

    @torch.no_grad()
    def __call__(self, image, mask):
        """
        Args:
            image: (B, 3, H, W) float tensor in [0, 1]; H and W must be multiples of 8.
            mask:  (B, 1, H, W) tensor, shadow pixels > 0.
        Returns:
            dict with ``inpainted`` (composited result) and ``predicted`` (raw generator output).
        """
        image = image.to(self.device)
        mask = (mask.to(self.device) > 0).float()
        model_mask = erode_mask(mask, self.erode_iterations)
        predicted = self.generator(torch.cat([image, model_mask], dim=1))
        inpainted = mask * predicted + (1 - mask) * image
        return dict(inpainted=inpainted, predicted=predicted)
