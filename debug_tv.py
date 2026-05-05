# debug_tv.py

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from IPPy.utilities import get_device

from config import IMAGE_SIZE, MOTION_BLUR
from degradation import create_blur_operator
from dataset import create_restoration_loader
from methods.tv import tv_reconstruction
from evaluation.metrics_rgb import compute_rgb_metrics


def main():
    device = get_device()
    print("Device:", device)

    loader = create_restoration_loader(
        split="train",
        max_samples=4,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        fixed_noise_level=0.01,
    )

    batch = next(iter(loader))

    clean = batch["clean"].to(device)
    degraded = batch["degraded"].to(device)

    K = create_blur_operator(
        image_size=IMAGE_SIZE,
        kernel_size=MOTION_BLUR["kernel_size"],
        theta=MOTION_BLUR["theta"],
    )

    reconstructed = tv_reconstruction(
        degraded=degraded,
        K=K,
        lambda_tv=0.005,
        num_iters=300,
        lr=0.05,
        verbose=True,
    )

    degraded_metrics = compute_rgb_metrics(degraded, clean)
    reconstructed_metrics = compute_rgb_metrics(reconstructed, clean)

    print("\nDegraded metrics:")
    for k, v in degraded_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nTV reconstruction metrics:")
    for k, v in reconstructed_metrics.items():
        print(f"{k}: {v:.4f}")

    out_dir = Path("debug_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = torch.cat(
        [
            clean.cpu(),
            degraded.cpu(),
            reconstructed.cpu(),
        ],
        dim=0,
    )

    grid = make_grid(
        comparison,
        nrow=clean.shape[0],
        normalize=True,
    )

    out_file = out_dir / "tv_comparison.png"
    save_image(grid, out_file)

    print("\nSaved:", out_file)


if __name__ == "__main__":
    main()