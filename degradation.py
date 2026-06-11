# degradation.py

from dataclasses import dataclass, field
import random
import torch

from IPPy.utilities import gaussian_noise
from IPPy import operators


@dataclass
class DegradationParameters:
    image_size: int = 256
    kernel_type: str | None = "motion"
    kernel_size: int = 9
    motion_angle: float = 45.0
    noise_levels: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.05, 0.1]
    )


def create_blur_operator(params: DegradationParameters):
    if params.kernel_type is None:
        return None

    return operators.Blurring(
        img_shape=(params.image_size, params.image_size),
        kernel_type=params.kernel_type,
        kernel_size=params.kernel_size,
        motion_angle=params.motion_angle,
    )


def add_noise(
    y: torch.Tensor,
    noise_level: float,
    seed: int | None = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    noise = gaussian_noise(y, noise_level)
    return torch.clamp(y + noise, 0.0, 1.0)


class ImageDegradation:
    def __init__(
        self,
        params: DegradationParameters | None = None,
    ) -> None:
        self.params = params or DegradationParameters()
        self.operator = create_blur_operator(self.params)

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.ndim != 4:
            raise ValueError(
                f"Expected image with shape [3,H,W] or [B,3,H,W], got {image.shape}"
            )

        if image.shape[1] != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels, got {image.shape[1]} channels"
            )

        if image.shape[-2:] != (
            self.params.image_size,
            self.params.image_size,
        ):
            raise ValueError(
                f"Expected image size {self.params.image_size}x"
                f"{self.params.image_size}, got {image.shape[-2:]}"
            )

        clean = image.float().clamp(0.0, 1.0)

        if self.operator is None:
            blurred = clean
        else:
            channels = [
                clean[:, i:i + 1, :, :]
                for i in range(3)
            ]

            blurred = torch.cat(
                [
                    self.operator(channel)
                    for channel in channels
                ],
                dim=1,
            )

        noise_level = random.choice(self.params.noise_levels)

        degraded = add_noise(
            blurred,
            noise_level=noise_level,
        )

        return degraded