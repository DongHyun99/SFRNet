import numpy as np


def linear_color_restoration(shadow_image, result, mask):
    """Per-image color restoration by per-channel linear regression (paper Eq. 13-14).

    For each color channel c, fits ``I_S[c] = b0 + b1 * I_sf[c]`` by least squares on the
    non-shadow pixels (mask == 0) and applies the fitted mapping to every pixel of ``result``.

    Args:
        shadow_image: (H, W, 3) uint8 input shadow image I_S.
        result:       (H, W, 3) uint8 shadow-removed image I_sf.
        mask:         (H, W) shadow mask, shadow pixels > 0.
    Returns:
        (H, W, 3) uint8 color-restored image I_sf-pp.
    """
    source = result.astype(np.float64) / 255.0
    target = shadow_image.astype(np.float64) / 255.0
    non_shadow = mask == 0
    if non_shadow.sum() < 2:
        return result.copy()

    restored = np.empty_like(source)
    for c in range(3):
        x = source[..., c][non_shadow]
        y = target[..., c][non_shadow]
        x_mean = x.mean()
        var = ((x - x_mean) ** 2).sum()
        b1 = ((x - x_mean) * (y - y.mean())).sum() / var if var > 0 else 1.0
        b0 = y.mean() - b1 * x_mean
        restored[..., c] = b0 + b1 * source[..., c]
    return np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8)
