# operators_rgb.py

import torch
import torch.nn.functional as F

from IPPy.operators import Operator


class RGBBlurring(Operator):
    """
    RGB blurring operator compatible with the IPPy Operator interface.

    Input/output shape:
        [N, 3, H, W]
    """

    def __init__(self, img_shape: tuple[int, int], kernel: torch.Tensor):
        super().__init__()

        self.nx, self.ny = img_shape
        self.mx, self.my = img_shape

        if kernel.ndim != 2:
            raise ValueError(f"Expected kernel shape [K, K], got {kernel.shape}")

        self.kernel = kernel.float()

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [N, C, H, W], got {x.shape}")

        n, c, h, w = x.shape
        k = self.kernel.shape[0]

        kernel = self.kernel.to(x.device)
        kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)

        return F.conv2d(
            x,
            kernel,
            padding=k // 2,
            groups=c,
        )

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 4:
            raise ValueError(f"Expected input [N, C, H, W], got {y.shape}")

        n, c, h, w = y.shape
        k = self.kernel.shape[0]

        kernel = torch.flip(self.kernel, dims=[0, 1]).to(y.device)
        kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)

        return F.conv2d(
            y,
            kernel,
            padding=k // 2,
            groups=c,
        )