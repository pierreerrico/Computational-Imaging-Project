# degradation.py

import torch

from IPPy.utilities import gaussian_noise

from operators_rgb import RGBBlurring


def motion_blur_kernel(kernel_size: int, theta: float) -> torch.Tensor:
    """
    Create a normalized 2D motion blur kernel.

    theta is expressed in degrees.
    """
    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)

    center = kernel_size // 2
    theta_rad = torch.tensor(theta * torch.pi / 180.0)

    dx = torch.cos(theta_rad).item()
    dy = torch.sin(theta_rad).item()

    for i in range(kernel_size):
        offset = i - center
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))

        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    kernel_sum = kernel.sum()

    if kernel_sum == 0:
        raise ValueError("Generated motion blur kernel is empty.")

    return kernel / kernel_sum


def create_blur_operator(
    image_size: int,
    kernel_size: int,
    theta: float,
) -> RGBBlurring:
    kernel = motion_blur_kernel(
        kernel_size=kernel_size,
        theta=theta,
    )

    return RGBBlurring(
        img_shape=(image_size, image_size),
        kernel=kernel,
    )


def add_relative_gaussian_noise(
    y: torch.Tensor,
    noise_level: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Add IPPy-style relative Gaussian noise.

    IPPy gaussian_noise returns noise e such that:

        ||e|| = noise_level * ||y||

    We then return clamp(y + e, 0, 1).
    """
    if seed is not None:
        torch.manual_seed(seed)

    noise = gaussian_noise(y, noise_level)
    return torch.clamp(y + noise, 0.0, 1.0)


def degrade_image(
    clean: torch.Tensor,
    blur_operator: RGBBlurring,
    noise_level: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Apply:

        clean -> motion blur -> Gaussian noise

    clean shape:
        [3, H, W]

    output shape:
        [3, H, W]
    """
    if clean.ndim != 3:
        raise ValueError(f"Expected clean image [C, H, W], got {clean.shape}")

    clean_batch = clean.unsqueeze(0)

    blurred = blur_operator(clean_batch)
    degraded = add_relative_gaussian_noise(
        blurred,
        noise_level=noise_level,
        seed=seed,
    )

    return degraded.squeeze(0)