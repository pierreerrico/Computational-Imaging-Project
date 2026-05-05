# debug_degradation.py

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from dataset import create_restoration_loader


def main():
    loader = create_restoration_loader(
        split="train",
        max_samples=16,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        fixed_noise_level=0.01,
    )

    batch = next(iter(loader))

    clean = batch["clean"]
    degraded = batch["degraded"]

    print("Clean shape:", clean.shape)
    print("Degraded shape:", degraded.shape)

    print("Clean range:", clean.min().item(), clean.max().item())
    print("Degraded range:", degraded.min().item(), degraded.max().item())

    print("Sigma:", batch["sigma"])
    print("Kernel size:", batch["kernel_size"])
    print("Theta:", batch["theta"])
    print("Idx:", batch["idx"])

    out_dir = Path("debug_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(8, clean.shape[0])

    comparison = torch.cat(
        [
            clean[:n],
            degraded[:n],
        ],
        dim=0,
    )

    grid = make_grid(
        comparison,
        nrow=n,
        normalize=True,
    )

    out_file = out_dir / "comparison.png"
    save_image(grid, out_file)

    print("Saved:", out_file)


if __name__ == "__main__":
    main()