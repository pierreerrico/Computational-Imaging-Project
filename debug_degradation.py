from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from degradation_dataset import create_imagenet_degradation_loader


def main():
    loader = create_imagenet_degradation_loader(
        split="train[:16]",
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
    print("Sigma:", batch["sigma"])
    print("Kernel size:", batch["kernel_size"])
    print("Theta:", batch["theta"])
    print("Idx:", batch["idx"])

    out_dir = Path("debug_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(8, clean.shape[0])

    grid = make_grid(
        torch.cat([clean[:n], degraded[:n]], dim=0),
        nrow=n,
        normalize=True
    )

    save_image(grid, out_dir / "clean_vs_degraded.png")

    print("Saved:", out_dir / "clean_vs_degraded.png")


if __name__ == "__main__":
    main()