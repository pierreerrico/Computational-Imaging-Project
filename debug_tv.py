# debug_tv.py

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from config import MOTION_BLUR
from degradation import motion_blur_kernel
from dataset import create_restoration_loader
from methods.tv import tv_reconstruction


def main():
    loader = create_restoration_loader(
        split="train",
        max_samples=4,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        fixed_noise_level=0.01,
    )

    batch = next(iter(loader))

    clean = batch["clean"]
    degraded = batch["degraded"]

    blur_kernel = motion_blur_kernel(
        kernel_size=MOTION_BLUR["kernel_size"],
        theta=MOTION_BLUR["theta"],
    )

    reconstructed = tv_reconstruction(
        degraded=degraded,
        blur_kernel=blur_kernel,
        lambda_tv=0.005,
        num_iters=300,
        lr=0.05,
        verbose=True,
    )

    out_dir = Path("debug_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = torch.cat(
        [
            clean,
            degraded,
            reconstructed,
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

    print("Saved:", out_file)


if __name__ == "__main__":
    main()