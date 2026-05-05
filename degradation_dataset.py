import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datasets import load_dataset


NOISE_LEVELS = [0.005, 0.01, 0.05, 0.1]

MOTION_BLUR_CONFIG = {
    "kernel_size": 9,
    "theta": 45
}


def motion_blur_kernel(kernel_size: int, theta: float):
    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)

    center = kernel_size // 2
    theta_rad = torch.tensor(theta * torch.pi / 180.0)

    dx = torch.cos(theta_rad)
    dy = torch.sin(theta_rad)

    for i in range(kernel_size):
        offset = i - center
        x = int(round(center + offset * dx.item()))
        y = int(round(center + offset * dy.item()))

        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    kernel = kernel / kernel.sum()
    return kernel


def apply_motion_blur(img: torch.Tensor, kernel: torch.Tensor):
    c, h, w = img.shape
    k = kernel.shape[0]

    kernel = kernel.to(img.device)
    kernel = kernel.view(1, 1, k, k)
    kernel = kernel.repeat(c, 1, 1, 1)

    img = img.unsqueeze(0)

    degraded = F.conv2d(
        img,
        kernel,
        padding=k // 2,
        groups=c
    )

    return degraded.squeeze(0)


class ImageNetDegradationDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        noise_levels=NOISE_LEVELS,
        motion_blur_config=MOTION_BLUR_CONFIG,
        fixed_noise_level=None,
        seed=42
    ):
        self.dataset = hf_dataset
        self.noise_levels = noise_levels
        self.motion_blur_config = motion_blur_config
        self.fixed_noise_level = fixed_noise_level
        self.seed = seed

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        self.blur_kernel = motion_blur_kernel(
            kernel_size=motion_blur_config["kernel_size"],
            theta=motion_blur_config["theta"]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        x = self.transform(sample["image"].convert("RGB"))

        y = apply_motion_blur(x, self.blur_kernel)

        if self.fixed_noise_level is None:
            sigma = self.noise_levels[idx % len(self.noise_levels)]
        else:
            sigma = self.fixed_noise_level

        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)

        noise = torch.randn(
            y.shape,
            generator=generator,
            dtype=y.dtype
        ) * sigma

        y = torch.clamp(y + noise, 0.0, 1.0)

        return {
            "clean": x,
            "degraded": y,
            "label": sample.get("label", -1),
            "sigma": sigma,
            "kernel_size": self.motion_blur_config["kernel_size"],
            "theta": self.motion_blur_config["theta"],
            "idx": idx
        }


def create_imagenet_degradation_loader(
    split="train",
    batch_size=16,
    shuffle=True,
    num_workers=4,
    fixed_noise_level=None,
    seed=42
):
    ds = load_dataset("benjamin-paine/imagenet-1k-256x256")

    hf_dataset = ds[split]

    dataset = ImageNetDegradationDataset(
        hf_dataset=hf_dataset,
        fixed_noise_level=fixed_noise_level,
        seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader