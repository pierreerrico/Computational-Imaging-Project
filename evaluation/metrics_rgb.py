# evaluation/metrics_rgb.py

import torch
from skimage.metrics import structural_similarity as ssim

from IPPy.utilities.metrics import PSNR, RE, RMSE


def SSIM_RGB(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    """
    RGB SSIM computed by averaging SSIM over RGB channels and batch.

    Expected shape:
        [N, 3, H, W]

    Expected range:
        [0, 1]
    """
    if x_pred.shape != x_true.shape:
        raise ValueError(f"Shape mismatch: {x_pred.shape} vs {x_true.shape}")

    if x_pred.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W], got {x_pred.shape}")

    x_pred = x_pred.detach().cpu().clamp(0, 1)
    x_true = x_true.detach().cpu().clamp(0, 1)

    n, c, h, w = x_pred.shape

    values = []

    for i in range(n):
        channel_values = []

        for ch in range(c):
            channel_values.append(
                ssim(
                    x_pred[i, ch].numpy(),
                    x_true[i, ch].numpy(),
                    data_range=1,
                )
            )

        values.append(sum(channel_values) / len(channel_values))

    return sum(values) / len(values)


def compute_rgb_metrics(x_pred: torch.Tensor, x_true: torch.Tensor) -> dict:
    """
    Compute RGB-compatible restoration metrics.
    """
    x_pred = x_pred.clamp(0, 1)
    x_true = x_true.clamp(0, 1)

    return {
        "PSNR": PSNR(x_pred, x_true),
        "SSIM": SSIM_RGB(x_pred, x_true),
        "RE": RE(x_pred, x_true),
        "RMSE": RMSE(x_pred, x_true),
    }