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


class RGBBlurOperator:
    """
    Optimized RGB Blur Operator wrapper.
    Processes batches by unstacking the color channels along the batch dimension
    to apply K fully in parallel across a single batch matrix execution.
    """
    def __init__(self, K) -> None:
        self.K = K

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {x.shape}")
        
        B, C, H, W = x.shape
        
        # Merge batch and channel dimensions: [B, C, H, W] -> [B * C, 1, H, W]
        # This executes K over all channels simultaneously in a single operation
        x_flat = x.reshape(B * C, 1, H, W)
        out_flat = self.K(x_flat)
        
        # Reshape back to the original layout
        return out_flat.reshape(B, C, H, W)

    def T(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {y.shape}")
        
        B, C, H, W = y.shape
        
        # Merge batch and channel dimensions for the Adjoint step
        y_flat = y.reshape(B * C, 1, H, W)
        out_flat = self.K.T(y_flat)
        
        return out_flat.reshape(B, C, H, W)


class ImageDegradation:
    def __init__(
        self,
        params: DegradationParameters | None = None,
    ) -> None:
        self.params = params or DegradationParameters()
        # Wrap our underlying single-channel IPPy operator inside our batch-optimized wrapper
        base_operator = create_blur_operator(self.params)
        self.operator = RGBBlurOperator(base_operator) if base_operator is not None else None

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        # Detect layout cleanly
        is_3d = image.ndim == 3
        if is_3d:
            image = image.unsqueeze(0)  # [3, H, W] -> [1, 3, H, W]

        if image.ndim != 4:
            raise ValueError(
                f"Expected image with shape [3,H,W] or [B,3,H,W], got {image.shape}"
            )

        if image.shape[1] != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels, got {image.shape[1]} channels"
            )

        if image.shape[-2:] != (self.params.image_size, self.params.image_size):
            raise ValueError(
                f"Expected image size {self.params.image_size}x"
                f"{self.params.image_size}, got {image.shape[-2:]}"
            )

        clean = image.float().clamp(0.0, 1.0)

        # Apply parallel operator directly without list slices
        if self.operator is None:
            blurred = clean
        else:
            blurred = self.operator(clean)

        # Sample and inject noise profile
        noise_level = random.choice(self.params.noise_levels)
        degraded = add_noise(
            blurred,
            noise_level=noise_level,
        )

        # Match structural output back exactly to your dataset requirements
        return degraded.squeeze(0) if is_3d else degraded