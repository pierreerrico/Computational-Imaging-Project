# degradation.py

import torch
import torch.nn.functional as F


def motion_blur_kernel(kernel_size: int, theta: float) -> torch.Tensor:
    """
    Create a 2D motion blur kernel.

    Parameters
    ----------
    kernel_size:
        Size of the square kernel.
    theta:
        Motion angle in degrees.

    Returns
    -------
    torch.Tensor
        Kernel of shape [kernel_size, kernel_size].
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


def apply_motion_blur(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Apply motion blur to a single image tensor.

    Parameters
    ----------
    img:
        Tensor of shape [C, H, W] in range [0, 1].
    kernel:
        Tensor of shape [K, K].

    Returns
    -------
    torch.Tensor
        Blurred image of shape [C, H, W].
    """
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor [C, H, W], got shape {img.shape}.")

    c, h, w = img.shape
    k = kernel.shape[0]

    kernel = kernel.to(img.device)
    kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)

    img = img.unsqueeze(0)

    blurred = F.conv2d(
        img,
        kernel,
        padding=k // 2,
        groups=c,
    )

    return blurred.squeeze(0)


def add_gaussian_noise(
    img: torch.Tensor,
    sigma: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Add deterministic Gaussian noise if seed is provided.
    """
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        noise = torch.randn(
            img.shape,
            generator=generator,
            dtype=img.dtype,
            device=img.device,
        ) * sigma
    else:
        noise = torch.randn_like(img) * sigma

    return torch.clamp(img + noise, 0.0, 1.0)


def degrade_image(
    clean: torch.Tensor,
    kernel: torch.Tensor,
    sigma: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Apply motion blur and Gaussian noise.

    clean -> motion blur -> additive Gaussian noise -> clamp [0, 1]
    """
    degraded = apply_motion_blur(clean, kernel)
    degraded = add_gaussian_noise(degraded, sigma=sigma, seed=seed)

    return degraded