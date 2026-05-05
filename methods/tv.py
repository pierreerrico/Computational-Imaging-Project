# methods/tv.py

import torch

from degradation import apply_motion_blur


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic Total Variation.

    x shape: [B, C, H, W]
    """
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()

    return tv_h + tv_w


def tv_reconstruction(
    degraded: torch.Tensor,
    blur_kernel: torch.Tensor,
    lambda_tv: float = 0.01,
    num_iters: int = 300,
    lr: float = 0.05,
    clamp: bool = True,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Reconstruct clean image from degraded observation using TV regularization.

    Minimizes:

        || A(x) - y ||^2 + lambda * TV(x)

    where:
        y = degraded image
        A = motion blur operator
        x = reconstructed image
    """

    device = degraded.device
    blur_kernel = blur_kernel.to(device)

    # inizializzazione: parto dall'immagine degradata
    x = degraded.clone().detach()
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    for it in range(num_iters):
        optimizer.zero_grad()

        blurred_x = torch.stack([
            apply_motion_blur(img, blur_kernel)
            for img in x
        ], dim=0)

        data_term = torch.mean((blurred_x - degraded) ** 2)
        tv_term = total_variation(x)

        loss = data_term + lambda_tv * tv_term

        loss.backward()
        optimizer.step()

        if clamp:
            with torch.no_grad():
                x.clamp_(0.0, 1.0)

        if verbose and (it % 50 == 0 or it == num_iters - 1):
            print(
                f"[TV] iter {it:04d} | "
                f"loss={loss.item():.6f} | "
                f"data={data_term.item():.6f} | "
                f"tv={tv_term.item():.6f}"
            )

    return x.detach()