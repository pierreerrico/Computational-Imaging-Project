# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datasets import load_dataset

from config import (
    DATASET_NAME,
    IMAGE_SIZE,
    NOISE_LEVELS,
    MOTION_BLUR,
    SEED,
)

from degradation import (
    motion_blur_kernel,
    degrade_image,
)


class ImageNetRestorationDataset(Dataset):
    """
    PyTorch dataset for image restoration.

    Each sample contains:
    - clean image x
    - degraded observation y
    - degradation parameters

    The degradation is deterministic with respect to idx and seed.
    """

    def __init__(
        self,
        hf_dataset,
        image_size: int = IMAGE_SIZE,
        noise_levels: list[float] = NOISE_LEVELS,
        motion_blur_config: dict = MOTION_BLUR,
        fixed_noise_level: float | None = None,
        seed: int = SEED,
    ):
        self.dataset = hf_dataset
        self.image_size = image_size
        self.noise_levels = noise_levels
        self.motion_blur_config = motion_blur_config
        self.fixed_noise_level = fixed_noise_level
        self.seed = seed

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.blur_kernel = motion_blur_kernel(
            kernel_size=motion_blur_config["kernel_size"],
            theta=motion_blur_config["theta"],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        clean = self.transform(sample["image"].convert("RGB"))

        if self.fixed_noise_level is None:
            sigma = self.noise_levels[idx % len(self.noise_levels)]
        else:
            sigma = self.fixed_noise_level

        degraded = degrade_image(
            clean=clean,
            kernel=self.blur_kernel,
            sigma=sigma,
            seed=self.seed + idx,
        )

        return {
            "clean": clean,
            "degraded": degraded,
            "label": sample.get("label", -1),
            "sigma": sigma,
            "kernel_size": self.motion_blur_config["kernel_size"],
            "theta": self.motion_blur_config["theta"],
            "idx": idx,
        }


def load_hf_imagenet(
    dataset_name: str = DATASET_NAME,
    split: str = "train",
    max_samples: int | None = None,
):
    """
    Load Hugging Face ImageNet dataset.

    Important:
    use split='train', split='validation', or split='test'.
    Do not use split='train[:16]' here if the full split has already been cached.
    Use max_samples instead.
    """
    hf_dataset = load_dataset(
        dataset_name,
        split=split,
    )

    if max_samples is not None:
        max_samples = min(max_samples, len(hf_dataset))
        hf_dataset = hf_dataset.select(range(max_samples))

    return hf_dataset


def create_restoration_dataset(
    split: str = "train",
    max_samples: int | None = None,
    fixed_noise_level: float | None = None,
    seed: int = SEED,
) -> ImageNetRestorationDataset:
    hf_dataset = load_hf_imagenet(
        split=split,
        max_samples=max_samples,
    )

    dataset = ImageNetRestorationDataset(
        hf_dataset=hf_dataset,
        fixed_noise_level=fixed_noise_level,
        seed=seed,
    )

    return dataset


def create_restoration_loader(
    split: str = "train",
    max_samples: int | None = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    fixed_noise_level: float | None = None,
    seed: int = SEED,
) -> DataLoader:
    dataset = create_restoration_dataset(
        split=split,
        max_samples=max_samples,
        fixed_noise_level=fixed_noise_level,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader